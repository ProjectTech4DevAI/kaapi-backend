"""
Analysis tool for multimodal load test results

Analyzes metrics from concurrency.py output to identify:
- Resource saturation patterns (CPU, Memory)
- Latency distribution by test type
- Real-time factor analysis for STT
- Error patterns and failure modes
- ECS task exhaustion indicators

Usage:
    python analyze.py load_test_results.json
    python analyze.py load_test_results.json --format markdown
    python analyze.py load_test_results.json --format csv --output report.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from datetime import datetime

# ECS Resource Limits (from concurrency.py)
ECS_MEMORY_MB = 4096
ECS_MEMORY_PRESSURE_THRESHOLD = int(ECS_MEMORY_MB * 0.8)  # 3276 MB
ECS_CPU_COUNT = 2


def load_results(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(file_path, "r") as f:
        return json.load(f)


def calculate_percentiles(
    values: List[float], percentiles: List[int] = [50, 90, 95, 99]
) -> Dict[str, float]:
    """Calculate percentile statistics"""
    if not values:
        return {f"p{p}": 0.0 for p in percentiles}

    return {f"p{p}": float(np.percentile(values, p)) for p in percentiles}


def analyze_latency(df: pd.DataFrame, test_type: str) -> Dict[str, Any]:
    """Analyze latency metrics by test type"""
    type_df = df[df["test_type"] == test_type]

    if len(type_df) == 0:
        return {}

    successful = type_df[type_df["success"] == True]
    latencies = successful["latency_ms"].tolist()

    if not latencies:
        return {"count": len(type_df), "success_count": 0, "success_rate": 0.0}

    stats = {
        "count": len(type_df),
        "success_count": len(successful),
        "success_rate": len(successful) / len(type_df) * 100,
        "latency_min": float(min(latencies)),
        "latency_max": float(max(latencies)),
        "latency_mean": float(np.mean(latencies)),
        "latency_median": float(np.median(latencies)),
        "latency_std": float(np.std(latencies)),
        **calculate_percentiles(latencies),
    }

    # TTFB analysis if available
    ttfb_values = successful["ttfb_ms"].dropna().tolist()
    if ttfb_values:
        stats["ttfb_mean"] = float(np.mean(ttfb_values))
        stats["ttfb_median"] = float(np.median(ttfb_values))
        stats.update(
            {f"ttfb_{k}": v for k, v in calculate_percentiles(ttfb_values).items()}
        )

    return stats


def analyze_stt_real_time_factor(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze real-time factor for STT requests"""
    stt_df = df[(df["test_type"] == "stt") & (df["success"] == True)]

    if len(stt_df) == 0:
        return {}

    rtf_values = stt_df["real_time_factor"].dropna().tolist()

    if not rtf_values:
        return {}

    return {
        "rtf_min": float(min(rtf_values)),
        "rtf_max": float(max(rtf_values)),
        "rtf_mean": float(np.mean(rtf_values)),
        "rtf_median": float(np.median(rtf_values)),
        **calculate_percentiles(rtf_values, [50, 90, 95, 99]),
        "rtf_below_1x": sum(1 for x in rtf_values if x < 1.0),
        "rtf_above_2x": sum(1 for x in rtf_values if x > 2.0),
    }


def analyze_resource_saturation(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze resource usage and identify saturation patterns"""
    memory_values = df["worker_memory_mb"].dropna().tolist()
    cpu_values = df["cpu_percent"].dropna().tolist()

    memory_pressure_events = df[df["memory_pressure"] == True]
    high_cpu_events = df[df["cpu_percent"] > 80]

    analysis = {
        "memory": {
            "min_mb": float(min(memory_values)) if memory_values else 0,
            "max_mb": float(max(memory_values)) if memory_values else 0,
            "mean_mb": float(np.mean(memory_values)) if memory_values else 0,
            **{
                f"memory_{k}": v
                for k, v in calculate_percentiles(memory_values).items()
            },
            "pressure_threshold_mb": ECS_MEMORY_PRESSURE_THRESHOLD,
            "pressure_events": len(memory_pressure_events),
            "pressure_rate": len(memory_pressure_events) / len(df) * 100
            if len(df) > 0
            else 0,
        },
        "cpu": {
            "min_percent": float(min(cpu_values)) if cpu_values else 0,
            "max_percent": float(max(cpu_values)) if cpu_values else 0,
            "mean_percent": float(np.mean(cpu_values)) if cpu_values else 0,
            **{f"cpu_{k}": v for k, v in calculate_percentiles(cpu_values).items()},
            "high_cpu_events_80pct": len(high_cpu_events),
            "high_cpu_rate": len(high_cpu_events) / len(df) * 100 if len(df) > 0 else 0,
        },
    }

    # Identify saturation warnings
    warnings = []
    if analysis["memory"]["pressure_rate"] > 10:
        warnings.append(
            f"⚠️  HIGH MEMORY PRESSURE: {analysis['memory']['pressure_rate']:.1f}% of requests exceeded {ECS_MEMORY_PRESSURE_THRESHOLD}MB"
        )

    if analysis["memory"]["max_mb"] > ECS_MEMORY_MB * 0.95:
        warnings.append(
            f"⚠️  CRITICAL: Memory usage reached {analysis['memory']['max_mb']:.0f}MB (95% of ECS limit)"
        )

    if analysis["cpu"]["high_cpu_rate"] > 20:
        warnings.append(
            f"⚠️  HIGH CPU USAGE: {analysis['cpu']['high_cpu_rate']:.1f}% of requests had >80% CPU"
        )

    analysis["warnings"] = warnings

    return analysis


def analyze_queue_wait(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze queue wait times (Celery/RabbitMQ backpressure indicator)"""
    queue_wait_values = df["queue_wait_ms"].dropna().tolist()

    if not queue_wait_values:
        return {}

    high_wait = [x for x in queue_wait_values if x > 1000]  # >1s wait

    return {
        "min_ms": float(min(queue_wait_values)),
        "max_ms": float(max(queue_wait_values)),
        "mean_ms": float(np.mean(queue_wait_values)),
        **calculate_percentiles(queue_wait_values),
        "high_wait_count": len(high_wait),
        "high_wait_rate": len(high_wait) / len(queue_wait_values) * 100,
    }


def analyze_errors(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze error patterns"""
    failed = df[df["success"] == False]

    if len(failed) == 0:
        return {"total_failures": 0, "failure_rate": 0.0, "error_types": {}}

    error_types = failed["error_type"].value_counts().to_dict()

    return {
        "total_failures": len(failed),
        "failure_rate": len(failed) / len(df) * 100,
        "error_types": error_types,
    }


def analyze_concurrency(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze concurrency levels during test"""
    concurrency_values = df["concurrency_level"].tolist()

    return {
        "min_workers": int(min(concurrency_values)) if concurrency_values else 0,
        "max_workers": int(max(concurrency_values)) if concurrency_values else 0,
        "mean_workers": float(np.mean(concurrency_values)) if concurrency_values else 0,
        **{
            f"concurrency_{k}": v
            for k, v in calculate_percentiles(concurrency_values).items()
        },
    }


def generate_report(data: Dict[str, Any], format: str = "text") -> str:
    """Generate formatted analysis report"""
    summary = data.get("summary", {})
    analysis = data.get("analysis", {})

    if format == "json":
        return json.dumps(data, indent=2)

    # Text/Markdown format
    report = []

    report.append("=" * 80)
    report.append("LOAD TEST ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Test Summary
    report.append("## TEST SUMMARY")
    report.append("-" * 80)
    report.append(f"Test Type:        {summary.get('test_type', 'N/A')}")
    report.append(f"Total Requests:   {summary.get('total_requests', 0)}")
    report.append(
        f"Successful:       {summary.get('successful', 0)} ({summary.get('successful', 0) / summary.get('total_requests', 1) * 100:.1f}%)"
    )
    report.append(
        f"Failed:           {summary.get('failed', 0)} ({summary.get('failed', 0) / summary.get('total_requests', 1) * 100:.1f}%)"
    )
    report.append(
        f"Memory Pressure:  {summary.get('memory_pressure_events', 0)} events"
    )
    report.append("")

    # Latency Analysis by Test Type
    for test_type in ["text", "stt", "tts"]:
        latency = analysis.get("latency", {}).get(test_type, {})
        if latency:
            report.append(f"## LATENCY ANALYSIS - {test_type.upper()}")
            report.append("-" * 80)
            report.append(f"Total Requests:   {latency.get('count', 0)}")
            report.append(f"Success Rate:     {latency.get('success_rate', 0):.1f}%")
            report.append(f"Latency Min:      {latency.get('latency_min', 0):.0f}ms")
            report.append(f"Latency Max:      {latency.get('latency_max', 0):.0f}ms")
            report.append(f"Latency Mean:     {latency.get('latency_mean', 0):.0f}ms")
            report.append(f"Latency Median:   {latency.get('latency_median', 0):.0f}ms")
            report.append(f"Latency p90:      {latency.get('p90', 0):.0f}ms")
            report.append(f"Latency p95:      {latency.get('p95', 0):.0f}ms")
            report.append(f"Latency p99:      {latency.get('p99', 0):.0f}ms")

            if "ttfb_mean" in latency:
                report.append(f"\nTime to First Byte:")
                report.append(f"  TTFB Mean:      {latency.get('ttfb_mean', 0):.0f}ms")
                report.append(f"  TTFB p90:       {latency.get('ttfb_p90', 0):.0f}ms")
                report.append(f"  TTFB p99:       {latency.get('ttfb_p99', 0):.0f}ms")

            report.append("")

    # STT Real-Time Factor
    rtf = analysis.get("real_time_factor", {})
    if rtf:
        report.append("## REAL-TIME FACTOR (STT)")
        report.append("-" * 80)
        report.append(f"RTF Min:          {rtf.get('rtf_min', 0):.2f}x")
        report.append(f"RTF Max:          {rtf.get('rtf_max', 0):.2f}x")
        report.append(f"RTF Mean:         {rtf.get('rtf_mean', 0):.2f}x")
        report.append(f"RTF Median:       {rtf.get('rtf_median', 0):.2f}x")
        report.append(f"RTF p90:          {rtf.get('p90', 0):.2f}x")
        report.append(f"RTF p95:          {rtf.get('p95', 0):.2f}x")
        report.append(f"RTF p99:          {rtf.get('p99', 0):.2f}x")
        report.append(
            f"RTF < 1x:         {rtf.get('rtf_below_1x', 0)} requests (faster than real-time)"
        )
        report.append(
            f"RTF > 2x:         {rtf.get('rtf_above_2x', 0)} requests (slower than 2x real-time)"
        )
        report.append("")

    # Resource Saturation
    resources = analysis.get("resources", {})
    if resources:
        report.append("## RESOURCE SATURATION ANALYSIS")
        report.append("-" * 80)

        # Memory
        mem = resources.get("memory", {})
        report.append("Memory Usage:")
        report.append(f"  Min:            {mem.get('min_mb', 0):.0f} MB")
        report.append(f"  Max:            {mem.get('max_mb', 0):.0f} MB")
        report.append(f"  Mean:           {mem.get('mean_mb', 0):.0f} MB")
        report.append(f"  p90:            {mem.get('memory_p90', 0):.0f} MB")
        report.append(f"  p99:            {mem.get('memory_p99', 0):.0f} MB")
        report.append(
            f"  Threshold:      {mem.get('pressure_threshold_mb', 0)} MB (80% of {ECS_MEMORY_MB}MB)"
        )
        report.append(
            f"  Pressure Events: {mem.get('pressure_events', 0)} ({mem.get('pressure_rate', 0):.1f}%)"
        )

        # CPU
        cpu = resources.get("cpu", {})
        report.append(f"\nCPU Usage:")
        report.append(f"  Min:            {cpu.get('min_percent', 0):.1f}%")
        report.append(f"  Max:            {cpu.get('max_percent', 0):.1f}%")
        report.append(f"  Mean:           {cpu.get('mean_percent', 0):.1f}%")
        report.append(f"  p90:            {cpu.get('cpu_p90', 0):.1f}%")
        report.append(f"  p99:            {cpu.get('cpu_p99', 0):.1f}%")
        report.append(
            f"  High CPU (>80%): {cpu.get('high_cpu_events_80pct', 0)} ({cpu.get('high_cpu_rate', 0):.1f}%)"
        )

        # Warnings
        warnings = resources.get("warnings", [])
        if warnings:
            report.append("\n⚠️  RESOURCE WARNINGS:")
            for warning in warnings:
                report.append(f"  {warning}")

        report.append("")

    # Queue Wait Time
    queue = analysis.get("queue_wait", {})
    if queue:
        report.append("## QUEUE WAIT TIME (Celery/RabbitMQ Backpressure)")
        report.append("-" * 80)
        report.append(f"Min:              {queue.get('min_ms', 0):.0f}ms")
        report.append(f"Max:              {queue.get('max_ms', 0):.0f}ms")
        report.append(f"Mean:             {queue.get('mean_ms', 0):.0f}ms")
        report.append(f"p90:              {queue.get('p90', 0):.0f}ms")
        report.append(f"p99:              {queue.get('p99', 0):.0f}ms")
        report.append(
            f"High Wait (>1s):  {queue.get('high_wait_count', 0)} ({queue.get('high_wait_rate', 0):.1f}%)"
        )

        if queue.get("high_wait_rate", 0) > 10:
            report.append("\n⚠️  HIGH QUEUE WAIT: Celery workers may be saturated")

        report.append("")

    # Concurrency
    conc = analysis.get("concurrency", {})
    if conc:
        report.append("## CONCURRENCY LEVELS")
        report.append("-" * 80)
        report.append(f"Min Workers:      {conc.get('min_workers', 0)}")
        report.append(f"Max Workers:      {conc.get('max_workers', 0)}")
        report.append(f"Mean Workers:     {conc.get('mean_workers', 0):.1f}")
        report.append(f"p90:              {conc.get('concurrency_p90', 0):.0f}")
        report.append(f"p99:              {conc.get('concurrency_p99', 0):.0f}")
        report.append("")

    # Error Analysis
    errors = analysis.get("errors", {})
    if errors and errors.get("total_failures", 0) > 0:
        report.append("## ERROR ANALYSIS")
        report.append("-" * 80)
        report.append(f"Total Failures:   {errors.get('total_failures', 0)}")
        report.append(f"Failure Rate:     {errors.get('failure_rate', 0):.1f}%")
        report.append("\nError Types:")
        for error_type, count in errors.get("error_types", {}).items():
            report.append(f"  {error_type}: {count}")
        report.append("")

    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Analyze multimodal load test results")
    parser.add_argument(
        "input_file", type=str, help="Input JSON file from concurrency.py"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument("--output", type=str, help="Output file (default: stdout)")

    args = parser.parse_args()

    # Load results
    if not Path(args.input_file).exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    data = load_results(args.input_file)
    results = data.get("results", [])

    if not results:
        print("Error: No results found in input file")
        return 1

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Perform analysis
    analysis = {
        "summary": data.get("summary", {}),
        "analysis": {
            "latency": {
                "text": analyze_latency(df, "text"),
                "stt": analyze_latency(df, "stt"),
                "tts": analyze_latency(df, "tts"),
            },
            "real_time_factor": analyze_stt_real_time_factor(df),
            "resources": analyze_resource_saturation(df),
            "queue_wait": analyze_queue_wait(df),
            "concurrency": analyze_concurrency(df),
            "errors": analyze_errors(df),
        },
    }

    # Generate report
    report = generate_report(analysis, format=args.format)

    # Output
    if args.output:
        if args.format == "csv":
            # Export detailed metrics to CSV
            df.to_csv(args.output, index=False)
            print(f"Detailed metrics exported to {args.output}")
        else:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Analysis report saved to {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
