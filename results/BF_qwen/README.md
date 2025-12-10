# BF Qwen Evaluation Results

This directory contains results from evaluating Qwen models on BF-generated binary sequences to investigate LLM in-context learning as an approximation of Solomonoff induction.

## Directory Structure

```
BF_qwen/
├── data/                          # Pre-generated BF sequences (optional)
├── individual_results/            # Model evaluation results (.npz files)
│   ├── qwen2_5_0_5b_BF_binary_results_1000x250_len300.npz
│   ├── qwen2_5_3b_BF_binary_results_1000x250_len300.npz
│   ├── qwen2_5_7b_BF_binary_results_1000x250_len300.npz
│   └── qwen2_5_32b_BF_binary_results_1000x250_len300.npz
└── plots/
    └── comparison/
        ├── qwen_token_loss_comparison_all_4_models_1000x250_bits.png
        └── regret_vs_complexity_all_4_models_1000x250_no_outliers.png
```

## Key Results

### Main Plots

1. **Token Loss Over Position** (`qwen_token_loss_comparison_all_4_models_1000x250_bits.png`):
   - Shows how average predictive log-loss (in bits) decreases as more bits are revealed
   - Compares all four Qwen model sizes (0.5B, 3B, 7B, 32B)
   - Demonstrates in-context learning behavior

2. **Regret vs Complexity** (`regret_vs_complexity_all_4_models_1000x250_no_outliers.png`):
   - Shows cumulative regret (total log-loss) as a function of program description length
   - Tests whether models exhibit Solomonoff-style simplicity bias
   - Linear regression reveals implicit prior exponent α

### Result Files (.npz)

Each result file contains:
- `sequences`: The binary sequences evaluated (shape: N×T)
- `log_losses`: Per-position log-loss in nats (shape: N×T)
- `program_lengths`: Original program lengths
- `short_program_lengths`: MDL-style shortened program lengths (used as L(p))
- `metadata`: Model name, evaluation parameters, etc.

## Scripts

All analysis scripts are in `neural_networks_solomonoff_induction/analysis/`:

1. **`generate_BF_binary_data.py`**: Generate BF sequences
2. **`llm_BF_binary_qwen.py`**: Evaluate Qwen models on sequences
3. **`plot_multiple_models_token_loss.py`**: Create loss-over-position comparison plot
4. **`plot_multiple_models_regret.py`**: Create regret-vs-complexity comparison plot

## Usage Example

```bash
# 1. Generate data (optional - can also generate on-the-fly)
python -m neural_networks_solomonoff_induction.analysis.generate_BF_binary_data \
  --num_sequences 1000 --sequence_length 250 --max_program_length 300 \
  --output_path results/BF_qwen/data/BF_binary_data_1000x250.npz

# 2. Evaluate a model
python -m neural_networks_solomonoff_induction.analysis.llm_BF_binary_qwen \
  --data_path results/BF_qwen/data/BF_binary_data_1000x250.npz \
  --model_name Qwen/Qwen2.5-3B \
  --output_path results/BF_qwen/individual_results/qwen2_5_3b_results.npz

# 3. Create comparison plots
python -m neural_networks_solomonoff_induction.analysis.plot_multiple_models_token_loss \
  --input_files results/BF_qwen/individual_results/qwen2_5_0_5b_results.npz \
                 results/BF_qwen/individual_results/qwen2_5_3b_results.npz \
  --labels "Qwen2.5-0.5B" "Qwen2.5-3B" \
  --output_path results/BF_qwen/plots/comparison/token_loss_comparison.png

python -m neural_networks_solomonoff_induction.analysis.plot_multiple_models_regret \
  --input_files results/BF_qwen/individual_results/qwen2_5_0_5b_results.npz \
                 results/BF_qwen/individual_results/qwen2_5_3b_results.npz \
  --labels "Qwen2.5-0.5B" "Qwen2.5-3B" \
  --use_short_program_length --remove_outliers \
  --output_path results/BF_qwen/plots/comparison/regret_vs_complexity.png
```
