"""Plot token loss over position for multiple Qwen models on the same figure.

Usage:
  python -m neural_networks_solomonoff_induction.analysis.plot_multiple_models_token_loss \\
    --input_files qwen2_5_0_5b_results.npz qwen2_5_3b_results.npz qwen2_5_7b_results.npz \\
    --labels "Qwen2.5-0.5B" "Qwen2.5-3B" "Qwen2.5-7B" \\
    --smooth_window 6 \\
    --output_path token_loss_comparison.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_and_smooth_loss(
    npz_path: str,
    smooth_window: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
  """Load log losses from .npz and return smoothed average loss over positions.
  
  Args:
    npz_path: Path to .npz file produced by llm_BF_binary_qwen.py
    smooth_window: Size of moving average window for smoothing
    
  Returns:
    Tuple of (positions, smoothed_loss) where positions is 0-indexed array
    and smoothed_loss is average token loss in bits, smoothed over smooth_window positions.
  """
  data = np.load(npz_path, allow_pickle=True)
  
  log_losses = data["log_losses"]  # (N, T), nats; position 0 may be NaN
  sequence_length = log_losses.shape[1]
  
  # Compute average loss over sequences, ignoring NaNs
  avg_log_loss_by_position = np.nanmean(log_losses, axis=0)
  avg_log_loss_by_position = np.asarray(avg_log_loss_by_position, dtype=float)
  
  # Convert from nats to bits (divide by ln(2))
  ln2 = math.log(2.0)
  avg_log_loss_by_position = avg_log_loss_by_position / ln2
  
  # Smooth over smooth_window positions using a moving average
  kernel_size = smooth_window
  kernel = np.ones(kernel_size) / kernel_size
  # Pad with edge values to maintain length
  padded = np.pad(avg_log_loss_by_position, (kernel_size//2, kernel_size//2), mode='edge')
  smoothed_loss = np.convolve(padded, kernel, mode='valid')
  # Ensure same length
  smoothed_loss = smoothed_loss[:sequence_length]
  
  positions = np.arange(sequence_length)
  return positions, smoothed_loss


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Plot token loss over position for multiple models on the same figure."
  )
  parser.add_argument(
      "--input_files",
      nargs="+",
      required=True,
      help="Paths to .npz files produced by llm_BF_binary_qwen.py",
  )
  parser.add_argument(
      "--labels",
      nargs="+",
      help="Labels for each model (default: filenames without extension)",
  )
  parser.add_argument(
      "--smooth_window",
      type=int,
      default=6,
      help="Size of moving average window for smoothing (default: 6)",
  )
  parser.add_argument(
      "--output_path",
      type=str,
      default="token_loss_comparison.png",
      help="Output path for the plot (default: token_loss_comparison.png)",
  )
  parser.add_argument(
      "--title",
      type=str,
      default=None,
      help="Plot title (default: auto-generated)",
  )
  parser.add_argument(
      "--log_scale",
      action="store_true",
      help="Use log scale for y-axis",
  )
  parser.add_argument(
      "--figsize",
      type=float,
      nargs=2,
      default=[10, 6],
      metavar=("WIDTH", "HEIGHT"),
      help="Figure size in inches (default: 10 6)",
  )
  
  args = parser.parse_args()
  
  if args.labels and len(args.labels) != len(args.input_files):
    parser.error("Number of labels must match number of input files")
  
  if not args.labels:
    args.labels = [Path(f).stem for f in args.input_files]
  
  fig, ax = plt.subplots(figsize=tuple(args.figsize))
  
  for input_file, label in zip(args.input_files, args.labels):
    try:
      positions, smoothed_loss = load_and_smooth_loss(input_file, args.smooth_window)
      ax.plot(positions, smoothed_loss, marker="o", markersize=2, label=label, linewidth=1.5)
    except Exception as e:
      print(f"Warning: Failed to load {input_file}: {e}")
      continue
  
  ax.set_xlabel("Position t", fontsize=12)
  ax.set_ylabel("Average token loss (bits)", fontsize=12)
  
  if args.log_scale:
    ax.set_yscale("log")
  
  if args.title:
    title = args.title
  else:
    title = f"Average token loss over position\n(smoothed over {args.smooth_window} positions)"
    if args.log_scale:
      title += " [log scale]"
  ax.set_title(title, fontsize=14)
  
  ax.grid(True, alpha=0.3)
  ax.legend(fontsize=10)
  plt.tight_layout()
  
  fig.savefig(args.output_path, dpi=150)
  print(f"Saved comparison plot to {args.output_path}")


if __name__ == "__main__":
  main()

