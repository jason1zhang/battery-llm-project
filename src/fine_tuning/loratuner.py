"""
LoRA fine-tuning using PEFT library.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)
from datasets import Dataset


class LoRATuner:
    """LoRA fine-tuning for language models."""

    def __init__(
        self,
        base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_dir: str = "models/lora_adapter",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        learning_rate: float = 3e-4,
        num_epochs: int = 3,
        per_device_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
    ):
        """
        Initialize LoRA tuner.

        Args:
            base_model_name: Hugging Face model ID
            output_dir: Directory to save adapter
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: Dropout probability
            target_modules: Modules to apply LoRA
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            per_device_batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Warmup steps
            logging_steps: Logging frequency
            save_steps: Checkpoint save frequency
            eval_steps: Evaluation frequency
        """
        self.base_model_name = base_model_name
        self.output_dir = output_dir

        # LoRA configuration
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]

        # Training configuration
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.per_device_batch_size = per_device_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps

        self.model = None
        self.tokenizer = None
        self.peft_config = None

    def load_model(self):
        """Load base model and tokenizer."""
        print(f"Loading base model: {self.base_model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )

        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model - use CPU to avoid MPS issues
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        # Explicitly move to CPU
        self.model = self.model.to("cpu")

        print("Model loaded successfully")

    def setup_lora(self):
        """Set up LoRA configuration."""
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
            inference_mode=False,
        )

        # Apply LoRA to model
        self.model = get_peft_model(self.model, self.peft_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(
        self,
        dataset_path: str,
        max_length: int = 512,
    ) -> Dataset:
        """
        Prepare dataset for training.

        Args:
            dataset_path: Path to JSONL dataset
            max_length: Maximum sequence length

        Returns:
            HuggingFace Dataset
        """
        # Load dataset
        with open(dataset_path, "r") as f:
            data = [json.loads(line) for line in f]

        # Format as instruction dataset
        def format_instruction(example):
            # Format: [INST] instruction [/INST] input [/INST] output
            text = f"[INST] {example['instruction']} [/INST]"
            if example.get("input"):
                text += f" {example['input']} [/INST]"
            else:
                text += " [/INST]"
            text += f" {example['output']}"
            return {"text": text}

        # Create dataset
        dataset = Dataset.from_list(data)
        dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        return dataset

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ):
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        if self.model is None:
            self.load_model()
            self.setup_lora()

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.per_device_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            eval_strategy="steps" if eval_dataset else "no",
            fp16=False,  # CPU training doesn't support fp16
            logging_dir=f"{self.output_dir}/logs",
            report_to="none",
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        print("Starting training...")
        trainer.train()

        # Save adapter
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"Model saved to {self.output_dir}")

    def merge_and_save(self, output_path: str):
        """
        Merge LoRA weights with base model and save.

        Args:
            output_path: Path to save merged model
        """
        if self.model is None:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                trust_remote_code=True,
            )
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(
                base_model,
                self.output_dir,
            )

        # Merge
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        print(f"Merged model saved to {output_path}")

    @staticmethod
    def load_trained_model(
        model_path: str,
        device: str = "cpu",
    ) -> tuple:
        """
        Load a trained model.

        Args:
            model_path: Path to saved model
            device: Device to load on

        Returns:
            Tuple of (model, tokenizer)
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer


def train_with_lora(
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dataset_path: str = "data/finetune/battery_dataset.jsonl",
    output_dir: str = "models/lora_adapter",
    **kwargs,
) -> LoRATuner:
    """
    Convenience function to train with LoRA.

    Args:
        base_model: Base model name
        dataset_path: Path to training dataset
        output_dir: Output directory
        **kwargs: Additional training arguments

    Returns:
        Trained LoRATuner
    """
    tuner = LoRATuner(
        base_model_name=base_model,
        output_dir=output_dir,
        **kwargs,
    )

    tuner.load_model()
    tuner.setup_lora()

    # Prepare dataset
    dataset = tuner.prepare_dataset(dataset_path)

    # Train (use 90% for train, 10% for eval)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size

    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))

    tuner.train(train_dataset, eval_dataset)

    return tuner
