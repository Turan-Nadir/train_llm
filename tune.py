import os
import logging
import argparse
import math
import numpy as np
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    BartConfig,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)
from datasets import load_dataset, DatasetDict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Custom progress callback
class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and logs:
            if 'loss' in logs:
                logger.info(f"Step {state.global_step}: Loss: {logs['loss']:.4f}")

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            logger.info(f"Epoch {state.epoch} completed")
            
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.is_local_process_zero and metrics:
            logger.info(f"Evaluation results at step {state.global_step}:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")

def setup_environment():
    """Setup the distributed environment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="DeepSpeed configuration file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--evaluation_steps", type=int, default=500, help="Evaluate every X steps")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--split", type=str, default="lat", choices=["lat", "original"], help="Dataset split to use (lat or original)")
    args = parser.parse_args()
    
    # Initialize the distributed environment
    deepspeed.init_distributed()
    
    # Create output directory
    if dist.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
    
    return args

def load_and_prepare_dataset(split="lat"):
    """Load and prepare the Uzbek dataset from Hugging Face, using only the specified split."""
    logger.info(f"Loading Uzbek 'tahrirchi' dataset from HuggingFace using only the '{split}' split...")
    
    # Load the dataset
    try:
        # Directly load just the 'lat' split
        dataset = load_dataset("tahrirchi/uz-books", split=split)
        logger.info(f"Dataset loaded successfully with {len(dataset)} examples in the {split} split")
        
        # Create train/validation splits since the dataset doesn't come with them
        train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
        train_val_split = train_test_split['train'].train_test_split(test_size=0.1, seed=42)
        
        # Create a DatasetDict with the splits
        dataset_dict = DatasetDict({
            'train': train_val_split['train'],
            'validation': train_val_split['test'],
            'test': train_test_split['test']
        })
        
        logger.info(f"Created dataset splits:")
        logger.info(f"  Train: {len(dataset_dict['train'])} examples")
        logger.info(f"  Validation: {len(dataset_dict['validation'])} examples")
        logger.info(f"  Test: {len(dataset_dict['test'])} examples")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    # Display dataset structure
    logger.info(f"Dataset structure: {dataset_dict}")
    logger.info(f"Sample data: {dataset_dict['train'][0]}")
    
    return dataset_dict

def preprocess_dataset(dataset, tokenizer, max_length=512):
    """Preprocess and tokenize the dataset for language modeling tasks."""
    logger.info("Preprocessing and tokenizing dataset...")
    
    # Check for the presence of 'text' field in the dataset
    if "text" not in dataset["train"].features:
        raise ValueError(f"Expected 'text' field in dataset but found: {dataset['train'].features}")
    
    def preprocess_function(examples):
        # For language modeling, we can use the same text as both input and target
        # This is suitable for BART training where we can learn text reconstruction
        text = examples["text"]
        
        # For QA fine-tuning later, we'll need to format text appropriately
        # Here we're preparing the model for general Uzbek language understanding
        
        # Tokenize inputs
        model_inputs = tokenizer(
            text, 
            max_length=max_length, 
            truncation=True, 
            padding="max_length"
        )
        
        # Set up decoder inputs which will be shifted inside the model
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        # Randomly mask some tokens for the Mask Language Modeling (MLM) objective
        # This helps with BART's denoising objective
        for i, input_ids in enumerate(model_inputs["input_ids"]):
            # Don't mask special tokens
            special_tokens_mask = [
                tokenizer.get_special_tokens_mask([id], already_has_special_tokens=True)[0]
                for id in input_ids
            ]
            
            probability_matrix = torch.full((len(input_ids),), 0.15)
            probability_matrix = torch.tensor(
                [p * (1 - s) for p, s in zip(probability_matrix, special_tokens_mask)]
            )
            
            # Sample masked indices from the probability distribution
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
            # If no tokens are masked, ensure at least one is (excluding special tokens)
            if not masked_indices.any() and not all(special_tokens_mask):
                # Find a non-special token to mask
                available_indices = [i for i, s in enumerate(special_tokens_mask) if not s]
                if available_indices:
                    masked_indices[np.random.choice(available_indices)] = True
            
            # For BART, we replace masked tokens with mask token
            input_ids_sentinel = input_ids.copy()
            indices_replaced = masked_indices.nonzero().tolist()
            for idx in indices_replaced:
                input_ids_sentinel[idx[0]] = tokenizer.mask_token_id
                
            # Store the modified input
            model_inputs["input_ids"][i] = input_ids_sentinel
        
        return model_inputs
    
    # Apply preprocessing to the dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,  # Use multiple processes for faster preprocessing
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )
    
    logger.info(f"Dataset tokenized successfully. Sample features: {list(tokenized_dataset['train'][0].keys())}")
    
    return tokenized_dataset

def create_bart_model_and_tokenizer(args):
    """Create a BART model and tokenizer for Uzbek."""
    logger.info("Loading BART model and tokenizer...")
    
    try:
        # Try loading facebook/bart-large
        model_name = "facebook/bart-large"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        logger.info(f"Loaded {model_name} successfully")
    except Exception as e:
        logger.warning(f"Error loading {model_name}: {e}")
        # Fallback to bart-base
        model_name = "facebook/bart-base"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        logger.info(f"Loaded {model_name} successfully as fallback")
    
    # Add Uzbek-specific tokens for the Latin alphabet
    # These are specific characters used in the Latinized Uzbek alphabet
    special_tokens = {
        "additional_special_tokens": [
            "oʻ", "gʻ", "sh", "ch", "Oʻ", "Gʻ", "Sh", "Ch"
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info(f"Model has {model.num_parameters():,} parameters")
    logger.info(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    return model, tokenizer

def train_model(model, tokenizer, tokenized_dataset, args):
    """Train the model using DeepSpeed."""
    logger.info("Starting training process...")
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.evaluation_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        deepspeed=args.deepspeed_config if args.deepspeed else None,
        local_rank=args.local_rank,
        fp16=True,
        report_to="none",  # Disable wandb, tensorboard, etc.
        predict_with_generate=True  # Enable generation during evaluation
    )
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding="max_length",
        max_length=args.max_length,
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[ProgressCallback()]
    )
    
    # Log some info before training
    logger.info(f"Training on {len(tokenized_dataset['train'])} examples")
    logger.info(f"Evaluating on {len(tokenized_dataset['validation'])} examples")
    
    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size per GPU: {args.batch_size}")
    logger.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  DeepSpeed enabled: {args.deepspeed}")
    logger.info(f"  Total GPUs: {dist.get_world_size()}")
    logger.info(f"  Effective batch size: {args.batch_size * dist.get_world_size() * args.gradient_accumulation_steps}")
    
    # Train the model
    logger.info("Starting training...")
    train_result = trainer.train()
    
    # Log training results
    if dist.get_rank() == 0:
        logger.info("Training completed!")
        logger.info(f"Training metrics: {train_result.metrics}")
        
        # Save the final model
        logger.info(f"Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Save training metrics
        with open(os.path.join(args.output_dir, "training_metrics.txt"), "w") as f:
            for key, value in train_result.metrics.items():
                f.write(f"{key}: {value}\n")
        
    return trainer

def main():
    """Main function to orchestrate the fine-tuning process."""
    start_time = datetime.now()
    logger.info(f"=== Starting BART fine-tuning process for Uzbek (Latin alphabet) at {start_time} ===")
    
    # Setup distributed environment
    args = setup_environment()
    
    # Log GPU information
    if dist.get_rank() == 0:
        logger.info(f"Using {torch.cuda.device_count()} GPUs on this node")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        logger.info(f"Total world size: {dist.get_world_size()}")
    
    # Create DeepSpeed config file if it doesn't exist
    if dist.get_rank() == 0 and args.deepspeed and not os.path.exists(args.deepspeed_config):
        logger.info(f"Creating DeepSpeed config file: {args.deepspeed_config}")
        with open(args.deepspeed_config, "w") as f:
            f.write('''{
                "train_batch_size": "auto",
                "gradient_accumulation_steps": "auto",
                "gradient_clipping": 1.0,
                "fp16": {
                    "enabled": true,
                    "loss_scale": 0,
                    "loss_scale_window": 1000,
                    "initial_scale_power": 16,
                    "hysteresis": 2,
                    "min_loss_scale": 1
                },
                "zero_optimization": {
                    "stage": 2,
                    "allgather_partitions": true,
                    "allgather_bucket_size": 2e8,
                    "overlap_comm": true,
                    "reduce_scatter": true,
                    "reduce_bucket_size": 2e8,
                    "contiguous_gradients": true
                },
                "optimizer": {
                    "type": "AdamW",
                    "params": {
                        "lr": "auto",
                        "weight_decay": "auto",
                        "betas": [0.9, 0.999],
                        "eps": 1e-8
                    }
                },
                "scheduler": {
                    "type": "WarmupLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": "auto",
                        "warmup_num_steps": "auto"
                    }
                },
                "steps_per_print": 100,
                "wall_clock_breakdown": false
            }''')
    
    # Ensure all processes have the DeepSpeed config file
    dist.barrier()
    
    # Load dataset - only the 'lat' (Latin alphabet) split as specified
    dataset = load_and_prepare_dataset(args.split)
    
    # Create model and tokenizer with consideration for Uzbek Latin alphabet
    model, tokenizer = create_bart_model_and_tokenizer(args)
    
    # Preprocess dataset
    tokenized_dataset = preprocess_dataset(dataset, tokenizer, args.max_length)
    
    # Train model
    trainer = train_model(model, tokenizer, tokenized_dataset, args)
    
    # Evaluate model
    if dist.get_rank() == 0 and "test" in tokenized_dataset:
        logger.info("Performing final evaluation on test set...")
        eval_results = trainer.evaluate(tokenized_dataset["test"])
        logger.info(f"Final test evaluation results: {eval_results}")
        
        # Save evaluation results
        with open(os.path.join(args.output_dir, "test_results.txt"), "w") as f:
            for key, value in eval_results.items():
                f.write(f"{key}: {value}\n")
    
    # Log completion time
    end_time = datetime.now()
    training_duration = end_time - start_time
    if dist.get_rank() == 0:
        logger.info(f"=== Fine-tuning completed at {end_time} ===")
        logger.info(f"Total training time: {training_duration}")
        logger.info(f"Fine-tuned model saved to: {args.output_dir}")
        logger.info(f"This model was trained on the '{args.split}' split (Latin alphabet)")

if __name__ == "__main__":
    main()