from vllm import LLM, SamplingParams

vllm_param = {
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "tensor_parallel_size": 2,
    "max_model_len": 32768,
    "gpu_memory_utilization": 0.2,
    "dtype": "float16",
}

sampling_params = SamplingParams(
    temperature=0.3,
    top_p=0.9,
    max_tokens=300,
    stop=["Étudiant:", "User:"]
)

llm = LLM(**vllm_param)

SYSTEM_PROMPT = """
Tu es un assistant étudiant de l’INALCO.
Tu aides les étudiants avec :
- inscriptions
- cours
- emplois du temps
- examens
- services étudiants
- bibliothèque
- campus

Réponds clairement et simplement.
Si tu ne sais pas, dis de contacter la scolarité.
"""


def ask_chatbot(question):
    prompt = f"""
[INST] {SYSTEM_PROMPT}

Question étudiant:
{question}
[/INST]
"""

    output = llm.generate(prompt, sampling_params)[0].outputs[0].text.strip()
    return output


print("Chatbot prêt (quit pour sortir)\n")

while True:
    q = input("Vous: ")
    if q == "quit":
        break
    print("Assistant:", ask_chatbot(q))

