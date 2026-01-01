# Figure Reproduction Guide

## Dialogue Turn Distribution
python plots/plot_turn_distribution.py --rounds_dir data/rounds --split train --out_dir artifacts/figures
python plots/plot_turn_distribution.py --rounds_dir data/rounds --split dev   --out_dir artifacts/figures
python plots/plot_turn_distribution.py --rounds_dir data/rounds --split test  --out_dir artifacts/figures

## Attention Heatmap (Guided Attention)
python plots/plot_attention_heatmap.py --attn_path artifacts/attention_samples.npz --out_dir artifacts/figures

## Nonverbal Marker Ablation
python plots/plot_nonverbal_ablation.py --metrics_path artifacts/nonverbal_metrics.json --out_dir artifacts/figures
