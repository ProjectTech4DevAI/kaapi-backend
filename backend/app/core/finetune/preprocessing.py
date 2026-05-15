import os
import json
import uuid
import tempfile
import logging
import io

from pathlib import Path
from fastapi import UploadFile
from starlette.datastructures import Headers
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataPreprocessor:
    RANDOM_STATE = 42

    def __init__(self, document, storage, split_ratio: float, system_message: str):
        self.document = document
        self.storage = storage
        self.split_ratio = split_ratio
        self.query_col = None
        self.label_col = None
        self.generated_files = []

        self.system_message = {"role": "system", "content": system_message.strip()}

    @staticmethod
    def upload_csv_to_s3(storage, df, filename: str) -> str:
        """
        Uploads a DataFrame as CSV to S3 using the provided storage instance.
        """
        logger.info(
            f"[upload_csv_to_s3] Preparing to upload '{filename}' to s3 | rows={len(df)}, cols={len(df.columns)}"
        )

        buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
        buf.seek(0)

        headers = Headers({"content-type": "text/csv"})
        upload = UploadFile(
            filename=filename,
            file=buf,
            headers=headers,
        )

        try:
            dest = storage.put(upload, file_path=Path("datasets") / filename)
            logger.info(
                f"[upload_csv_to_s3] Upload successful | filename='{filename}', s3_object='{dest}'"
            )
            return str(dest)
        except Exception as err:
            logger.error(
                f"[upload_csv_to_s3] Upload failed | filename='{filename}', error='{err}'",
                exc_info=True,
            )
            raise

    def _save_to_jsonl(self, data, filename):
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            for record in data:
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        self.generated_files.append(file_path)
        return file_path

    def cleanup(self):
        for path in self.generated_files:
            if os.path.exists(path):
                os.remove(path)
        self.generated_files = []

    def _modify_data_format(self, data):
        modified_data = []
        for record in data:
            user_message = {"role": "user", "content": record[self.query_col]}
            assistant_message = {"role": "assistant", "content": record[self.label_col]}
            modified_record = {
                "messages": [self.system_message, user_message, assistant_message]
            }
            modified_data.append(modified_record)
        return modified_data

    def _load_dataframe(self):
        logger.info(f"Loading CSV from: {self.document.object_store_url}")
        f_obj = self.storage.stream(self.document.object_store_url)
        try:
            f_obj.name = self.document.fname
            df = pd.read_csv(f_obj)
            df.columns = [col.strip().lower() for col in df.columns]

            possible_query_columns = ["query", "question", "message"]
            self.query_col = next(
                (col for col in possible_query_columns if col in df.columns), None
            )
            self.label_col = "label" if "label" in df.columns else None

            if not self.query_col or not self.label_col:
                logger.warning(
                    f"[DataPreprocessor] Dataset does not contain a 'label' column and one of: {possible_query_columns}"
                )
                raise ValueError(
                    f"CSV must contain a 'label' column and one of: {possible_query_columns}"
                )

            logger.info(
                f"[DataPreprocessor] Identified columns - query_col={self.query_col}, label_col={self.label_col}"
            )
            return df
        finally:
            f_obj.close()

    def process(self):
        logger.info("Starting data preprocessing")
        df = self._load_dataframe()

        train_data, test_data = train_test_split(
            df,
            test_size=1 - self.split_ratio,
            stratify=df[self.label_col],
            random_state=self.RANDOM_STATE,
        )

        logger.info(
            f"[DataPreprocessor] Data split complete: train_size={len(train_data)}, test_size={len(test_data)}"
        )

        train_dict = train_data.to_dict(orient="records")
        train_jsonl = self._modify_data_format(train_dict)

        unique_id = uuid.uuid4().hex
        train_percentage = int(self.split_ratio * 100)  # train %
        test_percentage = (
            100 - train_percentage
        )  # remaining % for test (since we used 1 - ratio earlier for test size)

        train_csv_name = f"train_split_{train_percentage}_{unique_id}.csv"
        test_csv_name = f"test_split_{test_percentage}_{unique_id}.csv"
        train_jsonl_name = f"train_data_{train_percentage}_{unique_id}.jsonl"

        train_csv_object = self.upload_csv_to_s3(
            self.storage, train_data, train_csv_name
        )
        test_csv_object = self.upload_csv_to_s3(self.storage, test_data, test_csv_name)

        train_jsonl_path = self._save_to_jsonl(train_jsonl, train_jsonl_name)

        return {
            "train_csv_s3_object": train_csv_object,
            "test_csv_s3_object": test_csv_object,
            "train_jsonl_temp_filepath": train_jsonl_path,
        }
