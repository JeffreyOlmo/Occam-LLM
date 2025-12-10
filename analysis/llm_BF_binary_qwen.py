"""Evaluate Qwen2.5-3B in-context on BF-generated binary sequences.

This script:
  * Uses the BF UTM implementation to *execute* randomly sampled
    BF programs.
  * Configures the UTM with a binary output alphabet (alphabet_size = 2),
    so each output character is a bit in {0, 1}.
  * Collects fixed-length binary sequences from these program executions.
  * Feeds growing binary prefixes to a base Qwen2.5-3B model and measures the
    predictive log-loss for the next bit at each position (pure in-context
    learning; no training).
  * Restricts the model’s prediction to {0, 1} by projecting its vocabulary
    distribution onto tokens whose decoded text is exactly "0" or "1"
    (up to surrounding whitespace), then renormalizing, following the spirit
    of the experiment described in:
      https://www.lesswrong.com/posts/xyYss3oCzovibHxAF/llm-in-context-learning-as-approximating-solomonoff

Usage (example):

  python -m neural_networks_solomonoff_induction.analysis.llm_BF_binary_qwen \\
    --num_sequences 256 \\
    --sequence_length 128 \\
    --output_path qwen2_5_3b_BF_binary_results.npz

Notes:
  * This script requires a GPU; there is intentionally no CPU fallback.
  * It downloads the base Qwen2.5-3B model from Hugging Face by default:
      model_name = "Qwen/Qwen2.5-3B"
"""

from __future__ import annotations

import argparse
import math
from typing import List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]
from tqdm import trange

from neural_networks_solomonoff_induction.data import utms as utms_lib


def _build_qwen_model_and_tokenizer(
    model_name: str,
    use_multi_gpu: bool = False,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, torch.device]:
  """Loads the base Qwen model + tokenizer on GPU only.
  
  Args:
    model_name: Hugging Face model identifier.
    use_multi_gpu: If True, use device_map="auto" to split across multiple GPUs.
                   Otherwise, load on a single GPU (default).
  """
  if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this script (no CPU fallback). "
        "Please run on a machine with a GPU and CUDA-enabled PyTorch."
    )
  device = torch.device("cuda")

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  # Use bfloat16 if available; fall back to float16/float32 as needed.
  dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  
  if use_multi_gpu:
    # Let transformers automatically split the model across available GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
    )
  else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
  model.eval()
  return model, tokenizer, device


def _digit_token_id_sets(
    tokenizer: AutoTokenizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Returns token-id sets for digits '0' and '1'.

  We scan the tokenizer vocabulary and collect ids where
  tokenizer.decode([id]).strip() == "0" (or "1"). This mirrors the approach in
  the LessWrong post, where the model's distribution is restricted to the
  target alphabet {0, 1} and renormalized [1]_.

  References:
    .. [1] Cole Wyeth, "LLM in-context learning as (approximating) Solomonoff
           induction", LessWrong (2025).
           https://www.lesswrong.com/posts/xyYss3oCzovibHxAF/llm-in-context-learning-as-approximating-solomonoff
  """
  zero_ids = []
  one_ids = []

  vocab_size = tokenizer.vocab_size
  for tok_id in range(vocab_size):
    text = tokenizer.decode([tok_id])
    stripped = text.strip()
    if stripped == "0":
      zero_ids.append(tok_id)
    elif stripped == "1":
      one_ids.append(tok_id)

  if not zero_ids or not one_ids:
    raise RuntimeError(
        "Could not find dedicated '0' and '1' tokens in the tokenizer "
        "vocabulary. This script assumes such tokens exist."
    )

  return (
      torch.tensor(zero_ids, dtype=torch.long),
      torch.tensor(one_ids, dtype=torch.long),
  )


def _bits_to_text(bits: Sequence[int]) -> str:
  """Encodes a bit sequence as comma-separated binary, e.g. '0,1,0,1'."""
  return ",".join(str(int(b)) for b in bits)


def _sample_binary_sequences_from_BF(
    num_sequences: int,
    sequence_length: int,
    max_program_length: int,
    memory_size: int,
    maximum_steps: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
  """Samples binary sequences by executing random BF programs.

  The BF UTM is instantiated with alphabet_size=2 so that each output
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
                    lengths (in BF tokens).
    short_lengths:  Array of shape (num_sequences,) with lengths of
                    `short_program` when available, otherwise equal to
                    `prog_lengths`.
    full_programs:  List of full program strings.
    short_programs: List of shortened program strings.
  """
  rng = np.random.default_rng(seed=seed)
  program_sampler = utms_lib.FastSampler(rng=rng)
  utm = utms_lib.BFUTM(
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
  while len(sequences) < num_sequences:
    attempts += 1
    program = utm.sample_program(length=max_program_length, rng=rng)
    result = utm.run_program(
        program=program,
        memory_size=memory_size,
        maximum_steps=maximum_steps,
        max_output_length=sequence_length,
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

  return (
      np.stack(sequences, axis=0),
      np.asarray(prog_lengths, dtype=np.int32),
      np.asarray(short_lengths, dtype=np.int32),
      full_programs,
      short_programs,
  )


def _evaluate_qwen_on_binary_sequences(
    sequences: np.ndarray,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: torch.device,
    zero_token_ids: torch.Tensor,
    one_token_ids: torch.Tensor,
    eval_batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
  """Computes per-position log-loss for next-bit prediction with Qwen.

  This implementation is **token-aligned and single-pass-per-sequence**:
    * We map each bit in {0, 1} to a canonical token id for '0' or '1'.
    * For each batch of sequences, we build a token id matrix of shape (B, T).
    * We run the model once to obtain logits for all positions.
    * For each position t >= 1, we use logits at position t-1 to obtain a
      distribution over the next bit, restricted and renormalized over {0, 1}
      (matching the earlier semantics where we projected onto the binary DSL).

  Args:
    sequences: Array of shape (N, T) with bits in {0, 1}.
    model: Loaded Qwen model.
    tokenizer: Matching tokenizer (unused here but kept for API compatibility).
    device: Torch device (expects CUDA).
    zero_token_ids: 1D tensor with vocab ids for tokens decoding to '0'.
    one_token_ids: 1D tensor with vocab ids for tokens decoding to '1'.
    eval_batch_size: Number of sequences to evaluate per forward pass.

  Returns:
    A tuple (log_losses, avg_log_loss_by_position) where:
      * log_losses has shape (N, T) and contains -ln p(x_t | x_<t) in nats
        for t >= 1; position 0 is filled with NaNs.
      * avg_log_loss_by_position has shape (T,) and is the mean over sequences
        at each position (ignoring NaNs at position 0).
  """
  num_sequences, sequence_length = sequences.shape
  log_losses = np.full((num_sequences, sequence_length), np.nan, dtype=np.float64)

  # Choose a canonical token id for '0' and '1' (first entry in each set).
  zero_id = int(zero_token_ids[0].item())
  one_id = int(one_token_ids[0].item())

  for start in trange(
      0,
      num_sequences,
      eval_batch_size,
      desc="Evaluating sequences",
      leave=False,
  ):
    end = min(start + eval_batch_size, num_sequences)
    batch_bits = sequences[start:end]  # (B, T), uint8 in {0,1}
    batch_size = batch_bits.shape[0]

    # Map bits to token ids directly.
    token_ids_np = np.where(
        batch_bits == 0,
        zero_id,
        one_id,
    ).astype(np.int64)  # (B, T)

    input_ids = torch.from_numpy(token_ids_np).to(device=device, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    with torch.no_grad():
      outputs = model(input_ids=input_ids, attention_mask=attention_mask)
      logits = outputs.logits  # (B, T, V)

    # Convert to log-probabilities.
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)  # (B, T, V)

    # For predicting bit at position t (1..T-1), we use logits at position t-1.
    log_probs_prev = log_probs[:, :-1, :]  # (B, T-1, V)

    # Compute log-probabilities for '0' and '1' by log-summing over their id sets.
    zero_ids_dev = zero_token_ids.to(device)
    one_ids_dev = one_token_ids.to(device)
    log_p0 = torch.logsumexp(
        log_probs_prev.index_select(dim=-1, index=zero_ids_dev), dim=-1
    )  # (B, T-1)
    log_p1 = torch.logsumexp(
        log_probs_prev.index_select(dim=-1, index=one_ids_dev), dim=-1
    )  # (B, T-1)

    # Renormalize over {0,1} as in the original implementation.
    log_Z = torch.logsumexp(
        torch.stack([log_p0, log_p1], dim=-1), dim=-1
    )  # (B, T-1)

    # True bits for positions t=1..T-1.
    bits_next = torch.from_numpy(
        batch_bits[:, 1:].astype(np.int64)
    ).to(device)  # (B, T-1)

    log_p_target = torch.where(bits_next == 0, log_p0, log_p1) - log_Z
    batch_log_losses = (-log_p_target).cpu().numpy()  # (B, T-1)

    # Insert into log_losses with NaN at position 0.
    log_losses[start:end, 1:] = batch_log_losses

  avg_log_loss_by_position = np.nanmean(log_losses, axis=0)
  return log_losses, avg_log_loss_by_position


def main() -> None:
  parser = argparse.ArgumentParser(
      description=(
          "Evaluate base Qwen2.5-3B on binary sequences generated by "
          "BF programs (in-context, no training)."
      )
  )
  parser.add_argument(
      "--num_sequences",
      type=int,
      default=256,
      help="Number of BF-generated binary sequences to evaluate.",
  )
  parser.add_argument(
      "--sequence_length",
      type=int,
      default=128,
      help="Number of bits per sequence.",
  )
  parser.add_argument(
      "--max_program_length",
      type=int,
      default=100,
      help="Maximum program length used when sampling BF programs.",
  )
  parser.add_argument(
      "--memory_size",
      type=int,
      default=10,
      help="Memory size for the BF UTM.",
  )
  parser.add_argument(
      "--maximum_steps",
      type=int,
      default=1000,
      help="Maximum number of UTM steps per program execution.",
  )
  parser.add_argument(
      "--model_name",
      type=str,
      default="Qwen/Qwen2.5-3B",
      help="Hugging Face model id for the base Qwen2.5-3B model.",
  )
  parser.add_argument(
      "--eval_batch_size",
      type=int,
      default=8,
      help="Batch size for Qwen forward passes.",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=0,
      help="Random seed for BF program and sequence sampling.",
  )
  parser.add_argument(
      "--data_path",
      type=str,
      default=None,
      help=(
          "Path to pre-generated .npz file containing sequences and program metadata. "
          "If provided, skips data generation and loads from this file instead. "
          "The file should contain 'sequences', 'program_lengths', and 'short_program_lengths'."
      ),
  )
  parser.add_argument(
      "--output_path",
      type=str,
      default="qwen2_5_3b_BF_binary_results.npz",
      help="Where to save the sequences and log-loss results (NumPy .npz).",
  )
  args = parser.parse_args()

  # 1) Either load pre-generated sequences or generate new ones.
  full_programs = None
  short_programs = None
  if args.data_path:
    print(f"Loading pre-generated sequences from {args.data_path}...")
    data = np.load(args.data_path, allow_pickle=True)
    sequences = data["sequences"]
    prog_lengths = data["program_lengths"]
    short_lengths = data["short_program_lengths"]
    num_sequences, sequence_length = sequences.shape
    print(f"Loaded sequences array of shape {sequences.shape}.")
    # Load programs if available
    if "full_programs" in data:
      full_programs = data["full_programs"].tolist() if hasattr(data["full_programs"], "tolist") else list(data["full_programs"])
    if "short_programs" in data:
      short_programs = data["short_programs"].tolist() if hasattr(data["short_programs"], "tolist") else list(data["short_programs"])
    # Override metadata from loaded data if available
    if "metadata" in data:
      loaded_metadata = data["metadata"].item() if hasattr(data["metadata"], "item") else data["metadata"]
      args.sequence_length = loaded_metadata.get("sequence_length", sequence_length)
      args.max_program_length = loaded_metadata.get("max_program_length", args.max_program_length)
      args.memory_size = loaded_metadata.get("memory_size", args.memory_size)
      args.maximum_steps = loaded_metadata.get("maximum_steps", args.maximum_steps)
      args.seed = loaded_metadata.get("seed", args.seed)
  else:
    print(
        f"Sampling {args.num_sequences} binary sequences of length "
        f"{args.sequence_length} from BF UTM..."
    )
    sequences, prog_lengths, short_lengths, full_programs, short_programs = _sample_binary_sequences_from_BF(
        num_sequences=args.num_sequences,
        sequence_length=args.sequence_length,
        max_program_length=args.max_program_length,
        memory_size=args.memory_size,
        maximum_steps=args.maximum_steps,
        seed=args.seed,
    )
    num_sequences, sequence_length = sequences.shape
    print(f"Collected sequences array of shape {sequences.shape}.")

  # 2) Load Qwen model and tokenizer on GPU (multi-GPU for large models).
  print(f"Loading model and tokenizer: {args.model_name}...")
  # Use multi-GPU for 32B model
  use_multi_gpu = "32B" in args.model_name or "32b" in args.model_name
  model, tokenizer, device = _build_qwen_model_and_tokenizer(
      args.model_name, use_multi_gpu=use_multi_gpu
  )

  # 3) Pre-compute token-id sets for digits '0' and '1'.
  print("Computing token-id sets for digits '0' and '1'...")
  zero_ids, one_ids = _digit_token_id_sets(tokenizer)
  print(
      f"Found {zero_ids.numel()} token ids for '0' and "
      f"{one_ids.numel()} token ids for '1'."
  )

  # 4) Evaluate Qwen's next-bit predictive log-loss across positions.
  print(
      "Evaluating Qwen2.5-3B log-loss on binary sequences "
      "(in-context; no training)..."
  )
  log_losses, avg_log_loss_by_position = _evaluate_qwen_on_binary_sequences(
      sequences=sequences,
      model=model,
      tokenizer=tokenizer,
      device=device,
      zero_token_ids=zero_ids,
      one_token_ids=one_ids,
      eval_batch_size=args.eval_batch_size,
  )

  # 5) Compute cumulative regret in bits as a function of program length.
  # log_losses are in nats; convert to bits via ln(2).
  ln2 = math.log(2.0)
  # Shape (N, T); first column is NaN by construction.
  # For cumulative sums, treat NaNs as zeros so that we effectively start
  # accumulating from t=1.
  finite_log_losses = np.where(np.isfinite(log_losses), log_losses, 0.0)
  cumulative_regret_bits = np.cumsum(finite_log_losses, axis=1) / ln2
  total_regret_bits = cumulative_regret_bits[:, -1]

  # 6) Save results for downstream analysis / plotting.
  print(f"Saving results to {args.output_path} ...")
  save_dict = {
      "sequences": sequences.astype(np.uint8),
      "log_losses": log_losses.astype(np.float32),
      "avg_log_loss_by_position": avg_log_loss_by_position.astype(np.float32),
      "program_lengths": prog_lengths.astype(np.int32),
      "short_program_lengths": short_lengths.astype(np.int32),
      "cumulative_regret_bits": cumulative_regret_bits.astype(np.float32),
      "total_regret_bits": total_regret_bits.astype(np.float32),
      "metadata": dict(
          num_sequences=int(num_sequences),
          sequence_length=int(sequence_length),
          max_program_length=int(args.max_program_length),
          memory_size=int(args.memory_size),
          maximum_steps=int(args.maximum_steps),
          model_name=str(args.model_name),
          eval_batch_size=int(args.eval_batch_size),
          seed=int(args.seed),
      ),
  }
  # Add programs if available
  if full_programs is not None:
    save_dict["full_programs"] = full_programs
  if short_programs is not None:
    save_dict["short_programs"] = short_programs
  np.savez(args.output_path, **save_dict)
  print("Done.")


if __name__ == "__main__":
  main()


