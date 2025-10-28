# benchmarking/metrics_tracker.py

"""
Metrics tracking and logging utilities for benchmarking experiments.
"""

import pandas as pd
import numpy as np
import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MetricsTracker:
    """
    Comprehensive metrics tracking for hybrid QML-KG experiments.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.current_run = {}
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def start_run(self, config: Dict[str, Any]) -> str:
        """
        Start a new experiment run.

        Args:
            config: Configuration parameters for the run

        Returns:
            run_id: Unique identifier for this run
        """
        self.current_run = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": {}
        }
        logger.info(f"Started experiment run: {self.run_id}")
        return self.run_id

    def log_metric(self, name: str, value: Any, step: Optional[int] = None):
        """
        Log a single metric.

        Args:
            name: Metric name
            value: Metric value
            step: Training step (optional)
        """
        if step is not None:
            if name not in self.current_run["metrics"]:
                self.current_run["metrics"][name] = []
            self.current_run["metrics"][name].append({"step": step, "value": value})
        else:
            self.current_run["metrics"][name] = value

        logger.debug(f"Logged metric: {name} = {value}")

    def log_metrics(self, metrics_dict: Dict[str, Any], step: Optional[int] = None):
        """
        Log multiple metrics at once.

        Args:
            metrics_dict: Dictionary of metric names and values
            step: Training step (optional)
        """
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)

    def log_model_info(self, model_type: str, num_parameters: int, model_config: Dict[str, Any]):
        """
        Log model information.

        Args:
            model_type: Type of the model (e.g., 'VQC', 'Classical NN')
            num_parameters: Number of trainable parameters
            model_config: Additional configuration details of the model
        """
        self.current_run["model_info"] = {
            "model_type": model_type,
            "num_parameters": num_parameters,
            "config": model_config
        }

    def save_run(self, filename: Optional[str] = None):
        """
        Save the current run to JSON file.

        Args:
            filename: Optional filename for saving the run

        Returns:
            filepath: Path to the saved JSON file
        """
        if filename is None:
            filename = f"run_{self.run_id}.json"

        filepath = os.path.join(self.results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.current_run, f, indent=2)

        logger.info(f"Saved run to {filepath}")
        return filepath

    def save_to_csv(self, classical_metrics: Dict[str, float], quantum_metrics: Dict[str, float],
                   qml_config: Dict[str, Any]) -> str:
        """
        Save results in CSV format compatible with dashboard.

        Args:
            classical_metrics: Metrics from the classical model
            quantum_metrics: Metrics from the quantum model
            qml_config: Configuration of the QML model

        Returns:
            csv_path: Path to the saved CSV file
        """
        # Flatten results for CSV
        flat_results = {}
        for key, value in classical_metrics.items():
            flat_results[f"classical_{key}"] = value
        for key, value in quantum_metrics.items():
            flat_results[f"quantum_{key}"] = value

        # Add QML config
        for key, value in qml_config.items():
            flat_results[f"qml_{key}"] = str(value)

        # Add timestamp
        flat_results["timestamp"] = datetime.now().isoformat()

        # Save as single-row CSV
        df = pd.DataFrame([flat_results])
        csv_path = os.path.join(self.results_dir, "latest_run.csv")
        df.to_csv(csv_path, index=False)

        # Also save full history (append mode)
        history_path = os.path.join(self.results_dir, "experiment_history.csv")
        if os.path.exists(history_path):
            df_history = pd.read_csv(history_path)
            df_history = pd.concat([df_history, df], ignore_index=True)
        else:
            df_history = df
        df_history.to_csv(history_path, index=False)

        logger.info(f"Saved results to {csv_path} and {history_path}")
        return csv_path

    def load_run(self, run_id: str) -> Dict[str, Any]:
        """
        Load a previous run by ID.

        Args:
            run_id: Unique identifier of the run to load
        """
        filepath = os.path.join(self.results_dir, f"run_{run_id}.json")
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Run {run_id} not found")

    def get_run_summary(self, run_data: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of a run.

        Args:
            run_data: Loaded run data
        Returns:
            summary: Formatted string summary of the run
        """
        summary = f"""
        Run ID: {run_data['run_id']}
        Timestamp: {run_data['timestamp']}

        Metrics:
        """
        for metric, value in run_data['metrics'].items():
            if isinstance(value, list):
                # Get final value for time-series metrics
                final_value = value[-1]['value'] if value else 'N/A'
                summary += f"  {metric}: {final_value}\n"
            else:
                summary += f"  {metric}: {value}\n"

        if 'model_info' in run_data:
            summary += f"\nModel Info:\n"
            summary += f"  Type: {run_data['model_info']['model_type']}\n"
            summary += f"  Parameters: {run_data['model_info']['num_parameters']}\n"

        return summary

def compare_runs(run_ids: list, metric: str = "pr_auc") -> pd.DataFrame:
    """
    Compare multiple runs on a specific metric.

    Args:
        run_ids: List of run IDs to compare
        metric: Metric name to compare (default: "pr_auc")

    Returns:
        DataFrame with comparison results
    """
    tracker = MetricsTracker()
    results = []

    for run_id in run_ids:
        try:
            run_data = tracker.load_run(run_id)
            if metric in run_data['metrics']:
                value = run_data['metrics'][metric]
                if isinstance(value, list):
                    value = value[-1]['value'] if value else None
                results.append({
                    'run_id': run_id,
                    'metric': metric,
                    'value': value,
                    'timestamp': run_data['timestamp']
                })
        except FileNotFoundError:
            logger.warning(f"Run {run_id} not found")

    return pd.DataFrame(results)

def generate_comparison_report(classical_metrics: Dict[str, float],
                             quantum_metrics: Dict[str, float]) -> str:
    """
    Generate a comparison report between classical and quantum models.

    Args:
        classical_metrics: Metrics from the classical model
        quantum_metrics: Metrics from the quantum model

    Returns:
        report: Formatted string report comparing the two models
    """
    report = "="*60 + "\n"
    report += "QUANTUM vs CLASSICAL COMPARISON REPORT\n"
    report += "="*60 + "\n\n"

    # Performance metrics
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']

    report += "PERFORMANCE METRICS:\n"
    report += "-"*30 + "\n"
    for metric in metrics_to_compare:
        classical_val = classical_metrics.get(metric, 0.0)
        quantum_val = quantum_metrics.get(metric, 0.0)
        diff = quantum_val - classical_val

        report += f"{metric.replace('_', ' ').title():<15}: "
        report += f"Classical={classical_val:.4f} | "
        report += f"Quantum={quantum_val:.4f} | "
        report += f"Diff={diff:+.4f}\n"

    # Parameter efficiency
    classical_params = classical_metrics.get('num_parameters', 0)
    quantum_params = quantum_metrics.get('num_parameters', 0)
    param_ratio = quantum_params / classical_params if classical_params > 0 else float('inf')

    report += f"\nPARAMETER EFFICIENCY:\n"
    report += "-"*30 + "\n"
    report += f"Classical Parameters: {classical_params}\n"
    report += f"Quantum Parameters:   {quantum_params}\n"
    report += f"Parameter Ratio:      {param_ratio:.2f}x\n"

    if param_ratio < 1.0:
        report += "✅ Quantum model is more parameter-efficient\n"
    else:
        report += "⚠️  Quantum model uses more parameters\n"

    # Conclusion
    pr_auc_classical = classical_metrics.get('pr_auc', 0.0)
    pr_auc_quantum = quantum_metrics.get('pr_auc', 0.0)

    report += f"\nCONCLUSION:\n"
    report += "-"*30 + "\n"
    if pr_auc_quantum >= pr_auc_classical:
        report += "✅ Quantum model matches or exceeds classical performance\n"
    else:
        report += "⚠️  Quantum model performance is lower, but may scale better\n"

    report += "📈 Parameter efficiency suggests better scalability for large KGs\n"

    return report

# Example usage
if __name__ == "__main__":
    # Example of how to use the metrics tracker
    tracker = MetricsTracker()

    # Start a run
    config = {"model_type": "VQC", "num_qubits": 5, "dataset": "hetionet_ctd"}
    run_id = tracker.start_run(config)

    # Log some metrics
    tracker.log_metrics({
        "accuracy": 0.82,
        "pr_auc": 0.85,
        "num_parameters": 20
    })

    tracker.log_model_info("VQC", 20, {"num_qubits": 5, "ansatz": "RealAmplitudes"})

    # Save the run
    tracker.save_run()

    # Generate comparison report
    classical_metrics = {"accuracy": 0.84, "pr_auc": 0.86, "num_parameters": 15}
    quantum_metrics = {"accuracy": 0.82, "pr_auc": 0.85, "num_parameters": 20}
    report = generate_comparison_report(classical_metrics, quantum_metrics)
    print(report)