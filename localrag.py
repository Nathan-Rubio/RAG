import torch
import ollama
import os
from openai import OpenAI
import argparse
import json

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

def carregar_chunks_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    
def gerar_embeddings(chunks, embedding_model="mxbai-embed-large"):
    embeddings = []
    textos = []

    print(NEON_GREEN + "Gerando embeddings dos chunks..." + RESET_COLOR)

    for chunk in chunks:
        texto = chunk["texto"]
        response = ollama.embeddings(model=embedding_model, prompt=texto)
        embeddings.append(response["embedding"])
        textos.append(texto)

    return torch.tensor(embeddings), textos


# Function to get relevant context from the vault based on user input
def get_relevant_context(user_input, chunk_embeddings, chunks, top_k=3):
    if chunk_embeddings.nelement() == 0:
        return []
    
    input_embedding = ollama.embeddings(
        model='mxbai-embed-large',
        prompt=user_input
    )["embedding"]

    cos_scores = torch.cosine_similarity(
        torch.tensor(input_embedding).unsqueeze(0),
        chunk_embeddings
    )

    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()

    relevant_chunks = [chunks[idx] for idx in top_indices]
    return relevant_chunks
   
def ollama_chat(user_input, system_message, chunk_embeddings, chunks, ollama_model, conversation_history):
    conversation_history.append({"role": "user", "content": user_input})

    relevant_chunks = get_relevant_context(user_input, chunk_embeddings, chunks, top_k=3)

    if relevant_chunks:
        context_str = "\n\n".join([
            f"[Chunk ID: {chunk['id']} | {chunk.get('capitulo', 'Sem capítulo')}]\n{chunk['texto']}"
            for chunk in relevant_chunks
        ])
        print(CYAN + "\nContexto recuperado:\n" + RESET_COLOR)
        print(CYAN + context_str[:3000] + RESET_COLOR)
    else:
        context_str = ""
        print(CYAN + "Nenhum contexto relevante encontrado." + RESET_COLOR)

    user_input_with_context = f"""
        Pergunta do usuário:
        {user_input}

        Trechos relevantes do regulamento:
        {context_str}
        """

    messages = [
        {"role": "system", "content": system_message},
        *conversation_history[:-1],  # histórico anterior
        {"role": "user", "content": user_input_with_context}
    ]

    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=1200,
        temperature=0.2,
    )

    resposta = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": resposta})

    return resposta



print(NEON_GREEN + "Parsing command-line arguments..." + RESET_COLOR)
parser = argparse.ArgumentParser(description="Local RAG com Ollama")
parser.add_argument("--model", default="llama3", help="Modelo Ollama (default: llama3)")
parser.add_argument("--chunks", default="chunks.json", help="Arquivo JSON com os chunks")
args = parser.parse_args()

print(NEON_GREEN + "Inicializando cliente Ollama..." + RESET_COLOR)
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'
)

print(NEON_GREEN + "Carregando chunks..." + RESET_COLOR)
chunks = carregar_chunks_json(args.chunks)

print(NEON_GREEN + f"{len(chunks)} chunks carregados." + RESET_COLOR)

chunk_embeddings, _ = gerar_embeddings(chunks)

conversation_history = []

system_message = """
Você é um assistente especializado nas regras de matrícula da universidade.

Suas instruções:
- Responda com base prioritariamente nos trechos recuperados do regulamento.
- Se a resposta estiver claramente no contexto fornecido, responda de forma objetiva e fiel ao texto.
- Se a informação não estiver clara ou não estiver presente, diga explicitamente que não encontrou essa informação no regulamento fornecido.
- Não invente regras, prazos ou exceções.
- Sempre que possível, cite o artigo ou capítulo mencionado no contexto.
- Seja claro, organizado e útil para estudantes.
"""

print(NEON_GREEN + "Sistema pronto! Digite sua pergunta ou 'quit' para sair.\n" + RESET_COLOR)

while True:
    user_input = input(YELLOW + "Pergunta: " + RESET_COLOR)

    if user_input.lower() == 'quit':
        break

    resposta = ollama_chat(
        user_input,
        system_message,
        chunk_embeddings,
        chunks,
        args.model,
        conversation_history
    )

    print(NEON_GREEN + "\nResposta:\n" + RESET_COLOR)
    print(resposta)
    print("\n" + "=" * 100 + "\n")