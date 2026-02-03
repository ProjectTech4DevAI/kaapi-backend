import os
import csv
import json
import logging
import time
from datetime import datetime
from typing import List, Protocol
from dataclasses import dataclass

import requests
import typer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)

cli = typer.Typer(help=__doc__)

HEADERS = {
    "accept": "application/json",
    "X-API-KEY": os.getenv("LOCAL_CREDENTIALS_API_KEY"),
    "Content-Type": "application/json",
}

PAYLOAD = {
    "project_id": 1,
}


@dataclass
class AssistantDatasetConfig:
    assistant_id: str
    filename: str
    query_column: str


@dataclass
class ResponsesDatasetConfig:
    model: str
    vector_store_ids: list[str]
    instructions: str
    filename: str
    query_column: str


def load_instructions(filename: str) -> str:
    with open(os.path.join(os.path.dirname(__file__), "data", filename), "r") as file:
        return file.read()


SERVICES = {
    "assistants": {
        "endpoint": "http://localhost:8000/api/v1/threads/sync",
        "datasets": {
            "kunji": AssistantDatasetConfig(
                assistant_id="asst_fz7oIQ2goRLfrP1mWceBcjje",
                filename="glific_kunji_test_queries.csv",
                query_column="prompt_text",
            ),
            "sneha": AssistantDatasetConfig(
                assistant_id="asst_U34ZORFHMrY6a8JqrFLzyUuy",
                filename="sneha_goldens.csv",
                query_column="Question",
            ),
        },
    },
    "responses": {
        "endpoint": "http://localhost:8000/api/v1/responses/sync",
        "datasets": {
            "kunji": ResponsesDatasetConfig(
                model="gpt-4o",
                vector_store_ids=["vs_QtyGjYxyM0OUKd6y7Z1pnfKU"],
                instructions=load_instructions("kunji_instructions.txt"),
                filename="glific_kunji_test_queries.csv",
                query_column="prompt_text",
            ),
            "sneha": ResponsesDatasetConfig(
                model="gpt-4o-mini",
                vector_store_ids=["vs_67a9de6638888191beb37c06f84e1a88"],
                instructions=load_instructions("sneha_instructions.txt"),
                filename="sneha_goldens.csv",
                query_column="Question",
            ),
        },
    },
}


@dataclass
class BenchItem:
    question: str
    answer: str
    duration: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_estimate_usd: float
    model: str


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    GPT_4o_MINI_2024_07_18_COSTING = {
        "input": 0.15,
        "cached_input": 0.075,
        "output": 0.60,
    }

    GPT_4o_2024_08_06_COSTING = {
        "input": 2.50,
        "cached_input": 1.25,
        "output": 10.00,
    }

    usd_per_1m = {
        "gpt-4o": GPT_4o_2024_08_06_COSTING,
        "gpt-4o-2024-08-06": GPT_4o_2024_08_06_COSTING,
        "gpt-4o-mini": GPT_4o_MINI_2024_07_18_COSTING,
        "gpt-4o-mini-2024-07-18": GPT_4o_MINI_2024_07_18_COSTING,
        # Extend with more models as needed: https://platform.openai.com/docs/pricing
    }

    pricing = usd_per_1m.get(model.lower())
    if not pricing:
        logging.warning(f"No pricing found for model '{model}'. Returning cost = 0.")
        return 0.0

    # We don't care about cached_input for now, this just to be mindful of upper bound cost to run benchmark
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost


def output_csv(items: List[BenchItem]):
    filename = f"bench_results_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    file_exists = os.path.exists(filename)
    with open(filename, "a") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "question",
                    "answer",
                    "duration",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "cost_estimate_usd",
                    "model",
                ]
            )
        for item in items:
            writer.writerow(
                [
                    item.question,
                    item.answer,
                    item.duration,
                    item.input_tokens,
                    item.output_tokens,
                    item.total_tokens,
                    item.cost_estimate_usd,
                    item.model,
                ]
            )

    return filename


def send_benchmark_request(
    prompt: str,
    i: int,
    total: int,
    endpoint: str,
    build_payload: callable,
) -> BenchItem:
    """
    Send a benchmark request to the specified endpoint.

    Args:
        prompt: The prompt to send
        i: Current index
        total: Total number of requests
        endpoint: API endpoint to call
        build_payload: Function that builds the payload given the prompt
    """
    local_payload = build_payload(prompt)

    start = time.perf_counter()
    response = requests.post(endpoint, headers=HEADERS, json=local_payload)
    end = time.perf_counter()
    duration = end - start

    if response.status_code == 200:
        result = response.json()
        result = result["data"]
        diagnostics = result["diagnostics"]
        return BenchItem(
            question=prompt,
            answer=result["message"],
            duration=duration,
            input_tokens=diagnostics["input_tokens"],
            output_tokens=diagnostics["output_tokens"],
            total_tokens=diagnostics["total_tokens"],
            cost_estimate_usd=estimate_cost(
                diagnostics["model"],
                diagnostics["input_tokens"],
                diagnostics["output_tokens"],
            ),
            model=diagnostics["model"],
        )
    else:
        typer.echo(response.text)
        typer.echo(f"[{i+1}/{total}] FAILED - Status: {response.status_code}")
        raise Exception(f"Request failed with status code {response.status_code}")


class DatasetConfig(Protocol):
    filename: str
    query_column: str


def load_and_dedupe_csv(
    dataset_config: DatasetConfig, count: int | None = None
) -> List[dict]:
    """Load and deduplicate CSV data for benchmarking."""
    csv_file_path = os.path.join(
        os.path.dirname(__file__), "data", dataset_config.filename
    )
    with open(csv_file_path, "r") as file:
        csv_reader = csv.DictReader(file)
        csv_data = list(csv_reader)

        if count:
            csv_data = csv_data[:count]

        # dedupe csv_data by query value
        seen_prompts = set()
        csv_data = [
            row
            for row in csv_data
            if row[dataset_config.query_column] not in seen_prompts
            and not seen_prompts.add(row[dataset_config.query_column])
        ]

        return csv_data


def calculate_statistics(results: List[BenchItem]) -> dict:
    """Calculate statistics from benchmark results."""
    total_runs = len(results)
    avg_duration = sum(item.duration for item in results) / total_runs
    total_input_tokens = sum(item.input_tokens for item in results)
    total_output_tokens = sum(item.output_tokens for item in results)
    total_duration = sum(item.duration for item in results)
    model = results[0].model
    total_cost = estimate_cost(model, total_input_tokens, total_output_tokens)

    return {
        "avg_duration": avg_duration,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_duration": total_duration,
        "model": model,
        "total_cost": total_cost,
        "total_runs": total_runs,
    }


def print_statistics(stats: dict):
    """Print benchmark statistics."""
    typer.echo(
        f"\nMean duration: {stats['avg_duration']:.3f}s over {stats['total_runs']} runs."
    )
    typer.echo(f"Total duration: {stats['total_duration']:.3f}s")
    typer.echo(
        f"Total tokens used: input={stats['total_input_tokens']}, output={stats['total_output_tokens']}"
    )
    typer.echo(
        f"Estimated cost for {stats['total_runs']} runs on {stats['model']}: ${stats['total_cost']:.6f}"
    )


@cli.command()
def assistants(
    count: int = typer.Option(
        None,
        help="Number of queries to process. If not set, will process all CSV rows.",
    ),
    dataset: str = typer.Option(
        "kunji", help="Dataset to use for benchmarking (kunji or sneha)."
    ),
):
    """
    Runs test queries on OpenAI Assistant API based /threads/sync endpoint to measure latency and cost.

    How to run the benchmark: in backend/ run `uv run ai-cli bench assistants --count 100 --dataset kunji`

    Note that increasing the number of workers beyond 1 can impact accuracy of duration.
    """
    dataset_config: AssistantDatasetConfig = SERVICES["assistants"]["datasets"][dataset]
    csv_data = load_and_dedupe_csv(dataset_config, count)

    typer.echo(f"Total deduped queries: {len(csv_data)}")
    total = len(csv_data)

    def build_assistant_payload(prompt: str) -> dict:
        return {
            **PAYLOAD,
            "question": prompt,
            "assistant_id": dataset_config.assistant_id,
        }

    results = []
    for i, row in enumerate(tqdm(csv_data, total=total)):
        result = send_benchmark_request(
            row[dataset_config.query_column],
            i,
            total,
            SERVICES["assistants"]["endpoint"],
            build_assistant_payload,
        )
        if result:
            results.append(result)

    stats = calculate_statistics(results)
    filename = output_csv(results)
    typer.echo(f"Results saved to {filename}")
    print_statistics(stats)


@cli.command()
def responses(
    count: int = typer.Option(
        None,
        help="Number of queries to process. If not set, will process all CSV rows.",
    ),
    dataset: str = typer.Option(
        "kunji", help="Dataset to use for benchmarking (kunji or sneha)."
    ),
):
    """
    Benchmarks Responses API-based /responses/sync endpoint for a given dataset.
    """
    dataset_config: ResponsesDatasetConfig = SERVICES["responses"]["datasets"][dataset]
    csv_data = load_and_dedupe_csv(dataset_config, count)

    typer.echo(f"Total deduped queries: {len(csv_data)}")
    total = len(csv_data)

    def build_responses_payload(prompt: str) -> dict:
        return {
            **PAYLOAD,
            "question": prompt,
            "model": dataset_config.model,
            "vector_store_ids": dataset_config.vector_store_ids,
            "instructions": dataset_config.instructions,
        }

    results = []
    for i, row in enumerate(tqdm(csv_data, total=total)):
        result = send_benchmark_request(
            row[dataset_config.query_column],
            i,
            total,
            SERVICES["responses"]["endpoint"],
            build_responses_payload,
        )
        if result:
            results.append(result)

    stats = calculate_statistics(results)
    filename = output_csv(results)
    typer.echo(f"Results saved to {filename}")
    print_statistics(stats)
