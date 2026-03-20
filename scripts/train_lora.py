#!/usr/bin/env python
"""
Train LoRA adapter for battery domain.
Usage: python scripts/train_lora.py
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.fine_tuning.loratuner import train_with_lora


def main():
    parser = argparse.ArgumentParser(description="Train LoRA adapter")
    parser.add_argument(
        "--dataset",
        default="data/finetune/battery_dataset.jsonl",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--output",
        default="models/lora_adapter",
        help="Output directory for adapter",
    )
    parser.add_argument(
        "--base-model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=8)
    args = parser.parse_args()

    print(f"Training LoRA adapter...")
    print(f"  Dataset: {args.dataset}")
    print(f"  Output: {args.output}")
    print(f"  Base model: {args.base_model}")

    tuner = train_with_lora(
        base_model=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        lora_r=args.rank,
    )

    print(f"Training complete! Adapter saved to {args.output}")


if __name__ == "__main__":
    main()
