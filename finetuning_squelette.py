import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# -----------------------
# 1. Configuration
# -----------------------

model_name = "mistralai/Ministral-8B-v0.1"  # adapter si besoin
output_dir = "./mistral_finetuned_model"

# -----------------------
# 2. Chargement tokenizer
# -----------------------

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------
# 3. Chargement modèle (4-bit)
# -----------------------

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# -----------------------
# 4. Configuration LoRA
# -----------------------

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# -----------------------
# 5. Dataset
# -----------------------

dataset = load_dataset("json", data_files="train.json") #à adapter 

def format_example(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(format_example)

# -----------------------
# 6. Training
# -----------------------

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
