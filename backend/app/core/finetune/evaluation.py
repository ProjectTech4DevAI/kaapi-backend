import difflib
import logging
import time
import uuid
from typing import Set

import openai
import pandas as pd
from openai import OpenAI
from sklearn.metrics import matthews_corrcoef

from app.core.cloud import AmazonCloudStorage
from app.core.finetune.preprocessing import DataPreprocessor
from app.utils import handle_openai_error

logger = logging.getLogger(__name__)


class ModelEvaluator:
    max_latency = 90
    retries = 3
    normalization_cutoff = 0.7

    def __init__(
        self,
        fine_tuned_model: str,
        test_data_s3_object: str,
        storage: AmazonCloudStorage,
        system_prompt: str,
        client: OpenAI,
    ):
        self.fine_tuned_model = fine_tuned_model
        self.test_data_s3_object = test_data_s3_object
        self.storage = storage
        self.system_instruction = system_prompt
        self.client = client

        self.allowed_labels: Set[str] = set()
        self.y_true: list[str] = []
        self.prompts: list[str] = []

        logger.info(f"ModelEvaluator initialized with model: {fine_tuned_model}")

    def load_labels_and_prompts(self) -> None:
        """
        Load prompts (X) and labels (y) from an S3-hosted CSV via storage.stream.
        Expects:
          - one of: 'query' | 'question' | 'message'
          - 'label'
        """
        logger.info(
            f"[ModelEvaluator.load_labels_and_prompts] Loading CSV from: "
            f"{self.test_data_s3_object}"
        )
        file_obj = self.storage.stream(self.test_data_s3_object)
        try:
            df = pd.read_csv(file_obj)
            df.columns = [c.strip().lower() for c in df.columns]

            possible_query_columns = ["query", "question", "message"]
            query_col = next(
                (c for c in possible_query_columns if c in df.columns), None
            )
            label_col = "label" if "label" in df.columns else None

            if not query_col or not label_col:
                logger.warning(
                    "[ModelEvaluator.load_labels_and_prompts] CSV must "
                    "contain a 'label' column and one of: "
                    f"{possible_query_columns}"
                )
                raise ValueError(
                    f"CSV must contain a 'label' column and one of: "
                    f"{possible_query_columns}"
                )

            prompts = df[query_col].astype(str).tolist()
            labels = df[label_col].astype(str).str.strip().str.lower().tolist()

            self.prompts = prompts
            self.y_true = labels
            self.allowed_labels = set(labels)

            self.query_col = query_col
            self.label_col = label_col

            logger.info(
                "[ModelEvaluator.load_labels_and_prompts] "
                f"Loaded {len(self.prompts)} prompts and "
                f"{len(self.y_true)} labels; "
                f"query_col={query_col}, label_col={label_col}, "
                f"allowed_labels={self.allowed_labels}"
            )
        except Exception as e:
            logger.error(
                f"[ModelEvaluator.load_labels_and_prompts] "
                f"Failed to load/parse test CSV: {e}",
                exc_info=True,
            )
            raise
        finally:
            file_obj.close()

    def normalize_prediction(self, text: str) -> str:
        logger.debug(f"[normalize_prediction] Normalizing prediction: {text}")
        t = (text or "").strip().lower()

        if t in self.allowed_labels:
            return t

        closest = difflib.get_close_matches(
            t, self.allowed_labels, n=1, cutoff=self.normalization_cutoff
        )
        if closest:
            return closest[0]

        logger.warning(
            f"[normalize_prediction] No close match found for '{t}'. "
            f"Using default label '{next(iter(self.allowed_labels))}'."
        )
        return next(iter(self.allowed_labels))

    def generate_predictions(self) -> tuple[list[str], str]:
        logger.info(
            f"[generate_predictions] Generating predictions for "
            f"{len(self.prompts)} prompts."
        )
        start_preds = time.time()
        predictions = []
        total_prompts = len(self.prompts)

        for idx, prompt in enumerate(self.prompts, 1):
            attempt = 0
            while attempt < self.retries:
                start_time = time.time()
                logger.info(
                    f"[generate_predictions] Processing prompt "
                    f"{idx}/{total_prompts} "
                    f"(Attempt {attempt + 1}/{self.retries})"
                )

                try:
                    response = self.client.responses.create(
                        model=self.fine_tuned_model,
                        instructions=self.system_instruction,
                        input=prompt,
                    )

                    elapsed_time = time.time() - start_time
                    if elapsed_time > self.max_latency:
                        logger.warning(
                            f"[generate_predictions] Timeout exceeded for "
                            f"prompt {idx}/{total_prompts}. Retrying..."
                        )
                        continue

                    raw = response.output_text or ""
                    prediction = self.normalize_prediction(raw)
                    predictions.append(prediction)
                    break

                except openai.OpenAIError as e:
                    error_msg = handle_openai_error(e)
                    logger.error(
                        f"[generate_predictions] OpenAI API error at prompt "
                        f"{idx}/{total_prompts}: {error_msg}"
                    )
                    attempt += 1
                    if attempt == self.retries:
                        predictions.append("openai_error")
                        logger.error(
                            f"[generate_predictions] Maximum retries reached "
                            f"for prompt {idx}/{total_prompts}. "
                            f"Appending 'openai_error'."
                        )
                    else:
                        logger.info(
                            f"[generate_predictions] Retrying prompt "
                            f"{idx}/{total_prompts} after OpenAI error "
                            f"({attempt}/{self.retries})."
                        )

        total_elapsed = time.time() - start_preds
        logger.info(
            f"[generate_predictions] Finished {total_prompts} prompts in "
            f"{total_elapsed:.2f}s | Generated {len(predictions)} "
            f"predictions."
        )

        prediction_data = pd.DataFrame(
            {
                "prompt": self.prompts,
                "true_label": self.y_true,
                "prediction": predictions,
            }
        )

        unique_id = uuid.uuid4().hex
        filename = f"predictions_{self.fine_tuned_model}_{unique_id}.csv"
        prediction_data_s3_object = DataPreprocessor.upload_csv_to_s3(
            self.storage, prediction_data, filename
        )
        self.prediction_data_s3_object = prediction_data_s3_object

        logger.info(
            f"[generate_predictions] Predictions CSV uploaded to S3 | "
            f"url={prediction_data_s3_object}"
        )

        return predictions, prediction_data_s3_object

    def evaluate(self) -> dict:
        """Evaluate using the predictions CSV previously uploaded to S3."""
        if not getattr(self, "prediction_data_s3_object", None):
            raise RuntimeError(
                "[evaluate] predictions_s3_object not set. "
                "Call generate_predictions() first."
            )

        logger.info(
            f"[evaluate] Streaming predictions CSV from: "
            f"{self.prediction_data_s3_object}"
        )
        prediction_obj = self.storage.stream(self.prediction_data_s3_object)
        try:
            df = pd.read_csv(prediction_obj)
        finally:
            prediction_obj.close()

        if "true_label" not in df.columns or "prediction" not in df.columns:
            raise ValueError(
                "[evaluate] prediction data CSV must contain 'true_label' "
                "and 'prediction' columns."
            )

        y_true = df["true_label"].astype(str).str.strip().str.lower().tolist()
        y_pred = df["prediction"].astype(str).str.strip().str.lower().tolist()

        try:
            mcc_score = round(matthews_corrcoef(y_true, y_pred), 4)
            logger.info(f"[evaluate] Computed MCC={mcc_score}")
            return {"mcc_score": mcc_score}
        except Exception as e:
            logger.error(f"[evaluate] Evaluation failed: {e}", exc_info=True)
            raise

    def run(self) -> dict:
        """Run the full evaluation process.

        Load data, generate predictions, and evaluate results.
        """
        try:
            self.load_labels_and_prompts()
            predictions, prediction_data_s3_object = self.generate_predictions()
            evaluation_results = self.evaluate()
            logger.info("[evaluate] Model evaluation completed successfully.")
            return {
                "evaluation_score": evaluation_results,
                "prediction_data_s3_object": prediction_data_s3_object,
            }
        except Exception as e:
            logger.error(f"[evaluate] Error in running ModelEvaluator: {str(e)}")
            raise
