"""Plot regret vs program length for multiple Qwen models on the same figure.

Usage:
  python -m neural_networks_solomonoff_induction.analysis.plot_multiple_models_regret \\
    --input_files qwen2_5_0_5b_results.npz qwen2_5_3b_results.npz qwen2_5_7b_results.npz qwen2_5_32b_results.npz \\
    --labels "Qwen2.5-0.5B" "Qwen2.5-3B" "Qwen2.5-7B" "Qwen2.5-32B" \\
    --use_short_program_length \\
    --output_path regret_vs_complexity_comparison.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _finite_mask(x: np.ndarray) -> np.ndarray:
  """Boolean mask of finite entries (not NaN/inf)."""
  return np.isfinite(x)


def _linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
  """Returns (slope, intercept) of least-squares fit y ≈ slope * x + intercept."""
  mask = _finite_mask(x) & _finite_mask(y)
  if not np.any(mask):
    raise ValueError("No finite data points for regression.")
  x_m = x[mask]
  y_m = y[mask]
  if x_m.size < 2:
    raise ValueError("Need at least two points for regression.")
  slope, intercept = np.polyfit(x_m, y_m, deg=1)
  return float(slope), float(intercept)


def load_regret_data(
    npz_path: str,
    use_short_program_length: bool = True,
    remove_outliers: bool = False,
) -> tuple[np.ndarray, np.ndarray, str]:
  """Load regret and program length data from .npz file.
  
  Args:
    npz_path: Path to .npz file produced by llm_brainphoque_binary_qwen.py
    use_short_program_length: If True, use short_program_lengths; else program_lengths
    remove_outliers: If True, filter out programs with unusually high regret (IQR method)
    
  Returns:
    Tuple of (L, R_T, model_name) where:
      L: Program lengths (description length)
      R_T: Total regret in bits at final horizon
      model_name: Model name from metadata
  """
  data = np.load(npz_path, allow_pickle=True)
  
  if use_short_program_length:
    L = data["short_program_lengths"].astype(np.float64)
  else:
    L = data["program_lengths"].astype(np.float64)
  
  # Recompute total regret from log_losses if not present
  if "total_regret_bits" in data:
    R_T = data["total_regret_bits"].astype(np.float64)
  else:
    log_losses = data["log_losses"]  # (N, T), nats
    ln2 = math.log(2.0)
    finite_log_losses = np.where(np.isfinite(log_losses), log_losses, 0.0)
    cumulative_regret_bits = np.cumsum(finite_log_losses, axis=1) / ln2
    R_T = cumulative_regret_bits[:, -1]
  
  # Filter outliers if requested
  if remove_outliers:
    Q1 = np.percentile(R_T, 25)
    Q3 = np.percentile(R_T, 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    outlier_mask = R_T <= upper_bound
    L = L[outlier_mask]
    R_T = R_T[outlier_mask]
  
  metadata = data.get("metadata", None)
  if metadata is not None:
    if hasattr(metadata, "item"):
      metadata = metadata.item()
    model_name = metadata.get("model_name", Path(npz_path).stem)
  else:
    model_name = Path(npz_path).stem
  
  return L, R_T, model_name


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Plot regret vs program length for multiple models on the same figure."
  )
  parser.add_argument(
      "--input_files",
      nargs="+",
      required=True,
      help="Paths to .npz files produced by llm_brainphoque_binary_qwen.py",
  )
  parser.add_argument(
      "--labels",
      nargs="+",
      help="Labels for each model (default: model names from metadata or filenames)",
  )
  parser.add_argument(
      "--use_short_program_length",
      action="store_true",
      help="Use short_program_lengths as description length (MDL-style)",
  )
  parser.add_argument(
      "--output_path",
      type=str,
      default="regret_vs_complexity_comparison.png",
      help="Output path for the plot (default: regret_vs_complexity_comparison.png)",
  )
  parser.add_argument(
      "--title",
      type=str,
      default=None,
      help="Plot title (default: auto-generated)",
  )
  parser.add_argument(
      "--remove_outliers",
      action="store_true",
      help="Remove outlier programs with unusually high regret (IQR method)",
  )
  parser.add_argument(
      "--figsize",
      type=float,
      nargs=2,
      default=[10, 7],
      metavar=("WIDTH", "HEIGHT"),
      help="Figure size in inches (default: 10 7)",
  )
  
  args = parser.parse_args()
  
  if args.labels and len(args.labels) != len(args.input_files):
    parser.error("Number of labels must match number of input files")
  
  fig, ax = plt.subplots(figsize=tuple(args.figsize))
  
  colors = plt.cm.tab10(np.linspace(0, 1, len(args.input_files)))
  
  for i, (input_file, color) in enumerate(zip(args.input_files, colors)):
    try:
      L, R_T, model_name = load_regret_data(
          input_file, 
          args.use_short_program_length,
          args.remove_outliers
      )
      
      label = args.labels[i] if args.labels else model_name
      
      # Compute regression
      slope, intercept = _linear_regression(L, R_T)
      corr = np.corrcoef(L, R_T)[0, 1]
      
      # Plot scatter
      ax.scatter(L, R_T, alpha=0.6, label=label, color=color, s=30)
      
      # Plot regression line
      x_line = np.linspace(L.min(), L.max(), 100)
      y_line = slope * x_line + intercept
      ax.plot(
          x_line, y_line, 
          color=color, 
          linestyle='--', 
          linewidth=2,
          alpha=0.8,
          label=f"{label} fit (α={slope:.3f}, r={corr:.3f})"
      )
      
      print(f"{label}: α={slope:.3f} bits/token, intercept={intercept:.1f} bits, corr={corr:.3f}")
      
    except Exception as e:
      print(f"Warning: Failed to load {input_file}: {e}")
      continue
  
  ax.set_xlabel(
      "Description length L(p) (short_program tokens)" if args.use_short_program_length
      else "Program length L(p) (tokens)",
      fontsize=12
  )
  ax.set_ylabel("Total regret R_T(p) (bits)", fontsize=12)
  
  if args.title:
    title = args.title
  else:
    title = "Regret vs program length\n(linear fit: R_T ≈ α L + b)"
  ax.set_title(title, fontsize=14)
  
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=9, loc='upper left')
  plt.tight_layout()
  
  fig.savefig(args.output_path, dpi=150)
  print(f"Saved comparison plot to {args.output_path}")


if __name__ == "__main__":
  main()

