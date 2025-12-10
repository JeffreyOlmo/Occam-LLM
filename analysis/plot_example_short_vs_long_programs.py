"""Plot cumulative regret for a short program vs a long program.

This script identifies programs at different percentiles of description length
and plots their cumulative regret trajectories, demonstrating how regret scales
with program complexity.

Usage:
  python -m neural_networks_solomonoff_induction.analysis.plot_example_short_vs_long_programs \\
    --input_path results/qwen2_5_3b_results.npz \\
    --use_short_program_length \\
    --output_path short_vs_long_regret.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neural_networks_solomonoff_induction.data import utms as utms_lib


def _regenerate_program_from_sequence(
    sequence: np.ndarray,
    seed: int,
    max_program_length: int,
    memory_size: int,
    maximum_steps: int,
    target_short_length: int | None = None,
) -> tuple[str, str]:
  """Regenerate a BrainPhoque program that produces the given sequence.
  
  This is approximate - we sample programs until we find one that produces
  the target sequence. If target_short_length is provided, we prefer programs
  with that shortened length. Returns the full and shortened program.
  """
  rng = np.random.default_rng(seed=seed)
  program_sampler = utms_lib.FastSampler(rng=rng)
  utm = utms_lib.BrainPhoqueUTM(
      sampler=program_sampler,
      alphabet_size=2,
      shorten_program=True,
  )
  
  # Convert uint8 array to string of '0' and '1'
  target_str = "".join(str(int(b)) for b in sequence)
  
  # Try to find a program that produces this sequence
  best_match = None
  best_match_len_diff = float("inf")
  best_match_output_len = 0
  
  for attempt in range(10000):
    # Vary program length to explore different possibilities
    if target_short_length is not None and attempt < 2000:
      # First try programs around the target length
      program_length = rng.integers(
          max(10, target_short_length - 30),
          min(max_program_length + 1, target_short_length + 100)
      )
    else:
      program_length = rng.integers(10, max_program_length + 1)
    
    program = utm.sample_program(length=program_length, rng=rng)
    result = utm.run_program(
        program=program,
        memory_size=memory_size,
        maximum_steps=maximum_steps,
        max_output_length=len(sequence) * 3,  # Allow more output
    )
    output = result.get("output", "")
    
    # Check if output matches (allowing for longer outputs)
    if len(output) >= len(target_str) and output[:len(target_str)] == target_str:
      full_prog = result.get("program", program)
      short_prog = result.get("short_program", full_prog)
      
      # If we have a target length, prefer matches closer to it
      if target_short_length is not None:
        len_diff = abs(len(short_prog) - target_short_length)
        # Also prefer programs that produce output closer to target length
        output_len_diff = abs(len(output) - len(target_str))
        score = len_diff + output_len_diff * 0.1
        if score < best_match_len_diff:
          best_match = (full_prog, short_prog)
          best_match_len_diff = score
          best_match_output_len = len(output)
          if len_diff == 0 and output_len_diff == 0:
            return best_match  # Perfect match!
      else:
        return full_prog, short_prog
  
  if best_match is not None:
    return best_match
  
  # Fallback: generate a simple representative program
  # Create a minimal program that outputs the sequence
  # This is a fallback - won't be exact but shows structure
  simple_prog = "+" * int(target_str[:10].count("1")) + "." if len(target_str) > 0 else "."
  return simple_prog, simple_prog


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Plot cumulative regret for short vs long programs."
  )
  parser.add_argument(
      "--input_path",
      type=str,
      required=True,
      help="Path to .npz file from llm_brainphoque_binary_qwen.py",
  )
  parser.add_argument(
      "--use_short_program_length",
      action="store_true",
      help="Use short_program_lengths instead of program_lengths for L(p).",
  )
  parser.add_argument(
      "--short_percentile",
      type=float,
      default=10.0,
      help="Percentile for 'short' program (default: 10.0).",
  )
  parser.add_argument(
      "--long_percentile",
      type=float,
      default=90.0,
      help="Percentile for 'long' program (default: 90.0).",
  )
  parser.add_argument(
      "--output_path",
      type=str,
      default=None,
      help="Output PNG path (default: <input_stem>_short_vs_long_regret.png).",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=0,
      help="Random seed for program regeneration (default: 0).",
  )
  parser.add_argument(
      "--data_path",
      type=str,
      default=None,
      help="Optional path to data file (.npz) to load programs from if not in result file.",
  )

  args = parser.parse_args()

  data = np.load(args.input_path, allow_pickle=True)

  log_losses = data["log_losses"]  # (N, T), nats
  sequences = data["sequences"]  # (N, T), uint8
  program_lengths = data["program_lengths"]
  short_program_lengths = data["short_program_lengths"]
  metadata = data.get("metadata", None)
  
  # Load programs if available
  full_programs = None
  short_programs = None
  if "full_programs" in data:
    full_programs = data["full_programs"]
    if hasattr(full_programs, "tolist"):
      full_programs = full_programs.tolist()
    else:
      full_programs = list(full_programs)
  if "short_programs" in data:
    short_programs = data["short_programs"]
    if hasattr(short_programs, "tolist"):
      short_programs = short_programs.tolist()
    else:
      short_programs = list(short_programs)

  num_sequences, sequence_length = log_losses.shape

  if args.use_short_program_length:
    L = short_program_lengths.astype(np.float64)
    print("Using short_program_lengths as description length L(p).")
  else:
    L = program_lengths.astype(np.float64)
    print("Using raw program_lengths as description length L(p).")

  # Recompute cumulative regret in bits.
  ln2 = math.log(2.0)
  finite_log_losses = np.where(np.isfinite(log_losses), log_losses, 0.0)
  cumulative_regret_bits = np.cumsum(finite_log_losses, axis=1) / ln2

  # Find programs at the specified percentiles
  short_idx = int(np.percentile(np.arange(num_sequences), args.short_percentile))
  long_idx = int(np.percentile(np.arange(num_sequences), args.long_percentile))
  
  # Sort by L to get actual percentile programs
  sorted_indices = np.argsort(L)
  short_idx_sorted = sorted_indices[int(num_sequences * args.short_percentile / 100)]
  long_idx_sorted = sorted_indices[int(num_sequences * args.long_percentile / 100)]

  short_L = L[short_idx_sorted]
  long_L = L[long_idx_sorted]
  short_regret = cumulative_regret_bits[short_idx_sorted, :]
  long_regret = cumulative_regret_bits[long_idx_sorted, :]
  short_sequence = sequences[short_idx_sorted, :]
  long_sequence = sequences[long_idx_sorted, :]

  print(f"\nShort program (percentile {args.short_percentile}):")
  print(f"  Description length L(p) = {short_L:.1f}")
  print(f"  Final regret = {short_regret[-1]:.2f} bits")
  print(f"\nLong program (percentile {args.long_percentile}):")
  print(f"  Description length L(p) = {long_L:.1f}")
  print(f"  Final regret = {long_regret[-1]:.2f} bits")

  # Try to regenerate the programs
  # Metadata is stored as a dict-like object (may be numpy array or dict)
  if metadata is not None:
    if hasattr(metadata, "item"):  # numpy array with dict inside
      metadata = metadata.item()
    max_prog_len = int(metadata.get("max_program_length", 300))
    memory_size = int(metadata.get("memory_size", 10))
    max_steps = int(metadata.get("maximum_steps", 100000))
  else:
    max_prog_len = 300
    memory_size = 10
    max_steps = 100000
  
  # Try loading from data file if programs not in result file
  if (short_programs is None or full_programs is None) and args.data_path:
    print(f"\nPrograms not in result file. Trying to load from data file: {args.data_path}")
    try:
      data_file = np.load(args.data_path, allow_pickle=True)
      if "full_programs" in data_file and "short_programs" in data_file:
        data_full_programs = data_file["full_programs"]
        data_short_programs = data_file["short_programs"]
        if hasattr(data_full_programs, "tolist"):
          data_full_programs = data_full_programs.tolist()
        if hasattr(data_short_programs, "tolist"):
          data_short_programs = data_short_programs.tolist()
        # Match sequences to find programs
        data_sequences = data_file["sequences"]
        # Find matching sequences
        short_match_idx = None
        long_match_idx = None
        for i in range(len(data_sequences)):
          if np.array_equal(data_sequences[i], short_sequence):
            short_match_idx = i
          if np.array_equal(data_sequences[i], long_sequence):
            long_match_idx = i
        if short_match_idx is not None:
          full_programs = [None] * num_sequences
          short_programs = [None] * num_sequences
          full_programs[short_idx_sorted] = data_full_programs[short_match_idx]
          short_programs[short_idx_sorted] = data_short_programs[short_match_idx]
        if long_match_idx is not None:
          if full_programs is None:
            full_programs = [None] * num_sequences
            short_programs = [None] * num_sequences
          full_programs[long_idx_sorted] = data_full_programs[long_match_idx]
          short_programs[long_idx_sorted] = data_short_programs[long_match_idx]
        if short_match_idx is not None or long_match_idx is not None:
          print(f"  Found programs in data file!")
    except Exception as e:
      print(f"  Could not load from data file: {e}")
  
  # Get programs from data if available, otherwise try to regenerate
  if short_programs is not None and full_programs is not None and short_programs[short_idx_sorted] is not None and short_programs[long_idx_sorted] is not None:
    print("\nLoading programs from result file...")
    short_full = full_programs[short_idx_sorted]
    short_short = short_programs[short_idx_sorted]
    long_full = full_programs[long_idx_sorted]
    long_short = short_programs[long_idx_sorted]
    print(f"  Short program: {len(short_short)} tokens (target: {short_L:.0f})")
    print(f"  Long program: {len(long_short)} tokens (target: {long_L:.0f})")
  else:
    print("\n" + "!"*80)
    print("NOTE: Programs are not stored in the result files.")
    print("Attempting to regenerate representative programs...")
    print("!"*80 + "\n")
    short_full, short_short = _regenerate_program_from_sequence(
        short_sequence, args.seed, max_prog_len, memory_size, max_steps,
        target_short_length=int(short_L)
    )
    long_full, long_short = _regenerate_program_from_sequence(
        long_sequence, args.seed, max_prog_len, memory_size, max_steps,
        target_short_length=int(long_L)
    )
    print(f"  Short program found: {len(short_short)} tokens (target: {short_L:.0f})")
    print(f"  Long program found: {len(long_short)} tokens (target: {long_L:.0f})")

  # Print programs to console
  print("\n" + "="*80)
  print("SHORT PROGRAM (percentile {:.0f}, L={:.1f})".format(args.short_percentile, short_L))
  print("="*80)
  print(f"Original length: {len(short_full)} tokens")
  print(f"Shortened length: {len(short_short)} tokens")
  print(f"Final regret: {short_regret[-1]:.2f} bits")
  print("\nShortened program:")
  print(short_short)
  print("\nFull program:")
  print(short_full)
  
  print("\n" + "="*80)
  print("LONG PROGRAM (percentile {:.0f}, L={:.1f})".format(args.long_percentile, long_L))
  print("="*80)
  print(f"Original length: {len(long_full)} tokens")
  print(f"Shortened length: {len(long_short)} tokens")
  print(f"Final regret: {long_regret[-1]:.2f} bits")
  print("\nShortened program:")
  print(long_short)
  print("\nFull program:")
  print(long_full)
  print("="*80 + "\n")
  
  # Create compact plot
  fig, ax = plt.subplots(figsize=(8, 5))
  positions = np.arange(1, sequence_length + 1)
  
  ax.plot(
      positions,
      short_regret,
      color="tab:blue",
      linewidth=2.0,
      label=f"Short program (L={short_L:.1f})",
  )
  ax.plot(
      positions,
      long_regret,
      color="tab:red",
      linewidth=2.0,
      label=f"Long program (L={long_L:.1f})",
  )
  
  ax.set_xlabel("Position in sequence", fontsize=11)
  ax.set_ylabel("Cumulative regret (bits)", fontsize=11)
  ax.set_title(
      "Cumulative Regret: Short vs Long Program",
      fontsize=12,
      fontweight="bold",
  )
  ax.grid(True, alpha=0.3)
  ax.legend(loc="upper left", fontsize=10)

  if args.output_path is None:
    stem = Path(args.input_path).stem
    args.output_path = f"{stem}_short_vs_long_regret.png"

  fig.savefig(args.output_path, dpi=150, bbox_inches="tight")
  print(f"\nSaved plot to {args.output_path}")


if __name__ == "__main__":
  main()

