import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

#Chargement du modèle de base et l'adpatateur 
base_model_name = "mistralai/Ministral-8B-v0.1"
lora_path = "./mistral_finetuned_model"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(model, lora_path)

#Génération 
def generate_answer(question, max_new_tokens=200):
    prompt = f"""### Instruction:
{question}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test
question = "Comment obtenir une attestation de scolarité ?"
print(generate_answer(question))
