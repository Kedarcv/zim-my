"""
Clair v5 - Training Script
Trains with improved dataset that fixes behavioral issues:
- Only mentions identity when explicitly asked
- Proper greeting/goodbye handling
- Says "I don't understand" when confused
- Normal conversations without constant identity mentions
"""

# Set GPU 1 BEFORE any imports that might initialize CUDA
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Configuration
BASE_MODEL_PATH = "/mnt/workspace/models/Qwen/Qwen2.5-3B-Instruct"
DATASET_PATH = "/mnt/workspace/zim-my/data/clair_v5_training.jsonl"
OUTPUT_PATH = "/mnt/workspace/models/clair-lora-v5"

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Configuration
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 2048

def load_dataset():
    """Load the v5 training dataset."""
    print(f"Loading dataset from: {DATASET_PATH}")
    
    examples = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"✓ Loaded {len(examples)} training examples")
    return Dataset.from_list(examples)

def format_example(example):
    """Format example for training using proper chat template."""
    messages = example['messages']
    
    # Use the tokenizer's chat template for proper formatting
    # This ensures the model learns the correct special tokens
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    # Format using chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

def train():
    """Train Clair v5."""
    print("=" * 60)
    print("Clair v5 - Training with Improved Behavior")
    print("=" * 60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("✗ No GPU available!")
        return False
    
    device = torch.device("cuda:0")  # Now refers to GPU 1 due to CUDA_VISIBLE_DEVICES
    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load dataset
    dataset = load_dataset()
    dataset = dataset.map(format_example)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    print("✓ Tokenizer loaded")
    
    # Load model with quantization
    print(f"\nLoading base model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        quantization_config=bnb_config,
    )
    print("✓ Model loaded")
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print(f"\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
    )
    
    # Initialize trainer
    print(f"\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
    )
    
    # Train
    print(f"\n{'=' * 60}")
    print("Starting training...")
    print(f"{'=' * 60}")
    
    trainer.train()
    
    # Save
    print(f"\n{'=' * 60}")
    print("Saving LoRA adapters...")
    print(f"{'=' * 60}")
    
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    print(f"\n✓ Training complete!")
    print(f"✓ LoRA adapters saved to: {OUTPUT_PATH}")
    
    return True

if __name__ == "__main__":
    success = train()
    
    if success:
        print("\n" + "=" * 60)
        print("Next steps:")
        print("=" * 60)
        print("1. Merge LoRA with base model:")
        print("   python notebooks/merge_clair_v5.py")
        print("\n2. Test the merged model")
        print("\n3. Convert to GGUF and quantize")
        print("\n4. Push to Ollama:")
        print("   ollama create r245142r/Clair-3B -f deploy/Modelfile")
        print("   ollama push r245142r/Clair-3B")
    else:
        print("\n✗ Training failed!")
