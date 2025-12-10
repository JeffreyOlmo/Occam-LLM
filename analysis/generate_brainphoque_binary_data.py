"""Generate and save BrainPhoque binary sequences for evaluation.

This script samples BrainPhoque programs, executes them to produce binary sequences,
and saves the sequences along with program metadata. This allows multiple models
to be evaluated on the same dataset without regenerating it each time.

Usage:
  python -m neural_networks_solomonoff_induction.analysis.generate_brainphoque_binary_data \\
    --num_sequences 32 \\
    --sequence_length 250 \\
    --max_program_length 120 \\
    --output_path brainphoque_binary_data_32x250.npz
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np

from neural_networks_solomonoff_induction.data import utms as utms_lib


def _sample_binary_sequences_from_brainphoque(
    num_sequences: int,
    sequence_length: int,
    max_program_length: int,
    memory_size: int,
    maximum_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
  """Samples binary sequences by executing random BrainPhoque programs.

  The BrainPhoque UTM is instantiated with alphabet_size=2 so that each output
  character is a single bit in {0, 1}. We repeatedly:
    * Sample a random program of length max_program_length (uniform over tokens).
    * Run it on the UTM up to `maximum_steps`, with max_output_length equal to
      `sequence_length`.
    * Keep the resulting output if it has length exactly `sequence_length`.
    * Record both the original program length and the `short_program` length
      (if shortening is enabled) so we can study regret as a function of
      description length L(p).

  Returns:
    sequences:      Array of shape (num_sequences, sequence_length) with entries
                    in {0, 1}.
    prog_lengths:   Array of shape (num_sequences,) with original program
                    lengths (in BrainPhoque tokens).
    short_lengths:  Array of shape (num_sequences,) with lengths of
                    `short_program` when available, otherwise equal to
                    `prog_lengths`.
    full_programs:  List of full program strings.
    short_programs: List of shortened program strings.
  """
  rng = np.random.default_rng(seed=seed)
  program_sampler = utms_lib.FastSampler(rng=rng)
  utm = utms_lib.BrainPhoqueUTM(
      sampler=program_sampler,
      alphabet_size=2,
      shorten_program=True,
  )

  sequences: List[np.ndarray] = []
  prog_lengths: List[int] = []
  short_lengths: List[int] = []
  full_programs: List[str] = []
  short_programs: List[str] = []
  attempts = 0
  print(f"Sampling {num_sequences} binary sequences of length {sequence_length} from BrainPhoque UTM...")
  print(f"  Using length-biased sampling: programs sampled with weights favoring longer lengths")
  
  # Create a distribution that heavily favors longer programs
  # Use exponential weighting: P(length) ∝ length^power
  # Higher power = stronger bias toward longer programs
  length_power = 2.0  # Quadratic weighting: longer programs are much more likely
  min_program_length = 10  # Minimum program length to sample
  
  # Pre-compute weights for each possible length
  possible_lengths = np.arange(min_program_length, max_program_length + 1)
  weights = possible_lengths ** length_power
  weights = weights / weights.sum()  # Normalize to probabilities
  
  while len(sequences) < num_sequences:
    attempts += 1
    if attempts % 100 == 0:
      print(f"  Attempts: {attempts}, Collected: {len(sequences)}/{num_sequences}")
    
    # Sample program length from weighted distribution favoring longer programs
    program_length = rng.choice(possible_lengths, p=weights)
    program = utm.sample_program(length=program_length, rng=rng)
    # Set max_output_length much higher to allow programs to produce more output,
    # then truncate afterward. This helps capture longer programs that would
    # otherwise be cut off early.
    result = utm.run_program(
        program=program,
        memory_size=memory_size,
        maximum_steps=maximum_steps,
        max_output_length=sequence_length * 10,  # Allow up to 10x the target length
    )
    output = result["output"]
    # Allow programs that produce *at least* `sequence_length` bits; truncate.
    # Still skip programs that halt or time out before producing enough output.
    if len(output) < sequence_length:
      continue
    output = output[:sequence_length]

    buffer = bytes(output, "utf-8")
    bits = np.frombuffer(buffer, dtype=np.uint8)

    if bits.shape[0] != sequence_length:
      continue
    if not np.all((bits == 0) | (bits == 1)):
      # With alphabet_size=2 this *should* not happen, but be defensive.
      continue

    sequences.append(bits)
    full_prog = result.get("program", program)
    short_prog = result.get("short_program", None)
    if short_prog is None:
      short_prog = full_prog
    prog_len = len(full_prog)
    short_len = len(short_prog)
    prog_lengths.append(prog_len)
    short_lengths.append(short_len)
    full_programs.append(full_prog)
    short_programs.append(short_prog)

  print(f"Collected sequences array of shape ({len(sequences)}, {sequence_length}).")
  return (
      np.stack(sequences, axis=0),
      np.asarray(prog_lengths, dtype=np.int32),
      np.asarray(short_lengths, dtype=np.int32),
      full_programs,
      short_programs,
  )


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Generate and save BrainPhoque binary sequences for evaluation."
  )
  parser.add_argument(
      "--num_sequences",
      type=int,
      default=32,
      help="Number of binary sequences to generate.",
  )
  parser.add_argument(
      "--sequence_length",
      type=int,
      default=250,
      help="Length of each binary sequence (in bits).",
  )
  parser.add_argument(
      "--max_program_length",
      type=int,
      default=120,
      help="Maximum program length used when sampling BrainPhoque programs.",
  )
  parser.add_argument(
      "--memory_size",
      type=int,
      default=10,
      help="Memory size for the BrainPhoque UTM.",
  )
  parser.add_argument(
      "--maximum_steps",
      type=int,
      default=1500,
      help="Maximum number of UTM steps per program execution.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=0,
      help="Random seed for reproducibility.",
  )
  parser.add_argument(
      "--output_path",
      type=str,
      default="brainphoque_binary_data.npz",
      help="Path to save the generated sequences and metadata.",
  )
  args = parser.parse_args()

  sequences, prog_lengths, short_lengths, full_programs, short_programs = _sample_binary_sequences_from_brainphoque(
      num_sequences=args.num_sequences,
      sequence_length=args.sequence_length,
      max_program_length=args.max_program_length,
      memory_size=args.memory_size,
      maximum_steps=args.maximum_steps,
      seed=args.seed,
  )

  metadata = {
      "num_sequences": args.num_sequences,
      "sequence_length": args.sequence_length,
      "max_program_length": args.max_program_length,
      "memory_size": args.memory_size,
      "maximum_steps": args.maximum_steps,
      "seed": args.seed,
  }

  np.savez(
      args.output_path,
      sequences=sequences,
      program_lengths=prog_lengths,
      short_program_lengths=short_lengths,
      full_programs=full_programs,
      short_programs=short_programs,
      metadata=metadata,
  )
  print(f"Saved generated data to {args.output_path}")


if __name__ == "__main__":
  main()

