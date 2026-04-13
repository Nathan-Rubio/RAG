import json
import argparse
import re
from statistics import mean

import torch
import ollama
from openai import OpenAI

# Modelo usado para gerar embeddings dos chunks e das perguntas.
# É importante usar o MESMO modelo de embedding para os dois lados,
# senão a comparação vetorial perde sentido
EMBED_MODEL = "mxbai-embed-large:latest"

# Cliente com a API da OpenAI, apontando para o servidor local do Ollama para uso local
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# Lê e retorna o conteúdo de um arquivo JSON.
def carregar_json(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# Gera as embeddings a partir dos chunks
# Transforma o texto em vetor numérico permitindo comparar os chunks armazenados
def gerar_embeddings_chunks(chunks, embedding_model=EMBED_MODEL):
    embeddings = []

    print("Gerando embeddings dos chunks...")

    for chunk in chunks:
        texto = chunk["texto"]

        response = ollama.embeddings(model=embedding_model, prompt=texto)
        embeddings.append(response["embedding"])

    # Converte a lista de embeddings em tensor para facilitar os cálculos vetoriais
    return torch.tensor(embeddings)


# Recupera os top_k chunks mais relevantes para uma pergunta
def get_relevant_chunks(user_input, chunk_embeddings, chunks, top_k=3):

    if chunk_embeddings.nelement() == 0:
        return [], []

    # Gera embedding da pergunta do usuário
    input_embedding = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=user_input
    )["embedding"]

    # Converte para tensor e adiciona dimensão para comparação com os chunks
    input_tensor = torch.tensor(input_embedding).unsqueeze(0)

    # Calcula a similaridade cosseno entre a pergunta e cada chunk
    cos_scores = torch.cosine_similarity(input_tensor, chunk_embeddings)

    # Garante que top_k não seja maior do que a quantidade de chunks
    top_k = min(top_k, len(cos_scores))

    # Seleciona os maiores values e seus respectivos índices
    top_values, top_indices = torch.topk(cos_scores, k=top_k)

    # Recupera os chunks correspondentes aos melhores índices
    relevant_chunks = [chunks[idx] for idx in top_indices.tolist()]
    relevant_scores = top_values.tolist()

    return relevant_chunks, relevant_scores


# Gera a resposta final do modelo usando a pergunta e os chunks recuperados
def gerar_resposta(user_input, retrieved_chunks, model_name, system_message):


    # Monta uma string com os chunks recuperados
    context_str = "\n\n".join([
        f"[ID {chunk.get('id', '-')}] "
        f"{chunk.get('capitulo', 'Sem capítulo')} | "
        f"{chunk.get('artigo', 'Sem artigo')}\n"
        f"{chunk['texto']}"
        for chunk in retrieved_chunks
    ])

    # Prompt final enviado como mensagem do usuário
    user_message = f"""
Pergunta do usuário:
{user_input}

Trechos relevantes do regulamento:
{context_str}
"""

    # Chamada ao modelo local via Ollama
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        max_tokens=1000,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


# Métrica de recall
def calcular_recall_at_k(artigos_esperados, retrieved_chunks):
    """
    Calcula Recall@k binário.

    Retorna 1 se pelo menos um dos artigos esperados aparecer entre os chunks recuperados.
    Retorna 0 caso contrário.
    """
    artigos_recuperados = [chunk.get("artigo", "") for chunk in retrieved_chunks]

    for artigo_esperado in artigos_esperados:
        if artigo_esperado in artigos_recuperados:
            return 1

    return 0

# Métrica MRR
def calcular_mrr(artigos_esperados, retrieved_chunks):
    """
    Se o primeiro artigo correto aparecer:
    - em 1º lugar -> retorna 1.0
    - em 2º lugar -> retorna 0.5
    - em 3º lugar -> retorna 0.333...
    - etc.

    Se nenhum artigo esperado for recuperado, retorna 0.
    """
    artigos_recuperados = [chunk.get("artigo", "") for chunk in retrieved_chunks]

    for rank, artigo in enumerate(artigos_recuperados, start=1):
        if artigo in artigos_esperados:
            return 1 / rank

    return 0


# Normaliza texto para facilitar comparação
def normalizar_texto(texto):

    # Remove pontuação, espaços duplicados e deixa tudo minusculo
    texto = texto.lower().strip()
    texto = re.sub(r'[^\w\sà-úÀ-Ú]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto

# Extrai apenas palavras relevantes
def extrair_palavras_relevantes(texto):

    # Remove palavras insignificantes no contexto
    stopwords_basicas = {
        "a", "o", "e", "de", "da", "do", "das", "dos", "em", "no", "na", "nas", "nos",
        "para", "por", "com", "sem", "um", "uma", "os", "as", "ao", "à", "é", "ser",
        "que", "se", "não", "sim", "ou", "mais", "menos", "até"
    }

    palavras = normalizar_texto(texto).split()
    return [p for p in palavras if p not in stopwords_basicas and len(p) > 2]


# Métrica simples de avaliar a quantidade de palavras relevantes na resposta
def calcular_keyword_recall(resposta_esperada, resposta_modelo):

    palavras_esperadas = extrair_palavras_relevantes(resposta_esperada)
    palavras_modelo = set(extrair_palavras_relevantes(resposta_modelo))

    if not palavras_esperadas:
        return 0.0

    acertos = sum(1 for p in palavras_esperadas if p in palavras_modelo)
    return acertos / len(palavras_esperadas)


# Executa a avaliação da RAG
def avaliar_rag(chunks, dataset, chunk_embeddings, model_name, top_k=3):
    """
    1. recupera os chunks mais relevantes
    2. gera a resposta do modelo
    3. calcula Recall@k, mrr e palavras-chaves
    4. salva em uma estrutura de resultados
    """

    # Prompt de sistema que restringe o comportamento do modelo
    system_message = """
Você é um assistente especializado nas regras de matrícula da universidade.

Regras:
- Responda apenas com base nos trechos recuperados do regulamento.
- Se a resposta não estiver claramente presente, diga:
  "Não encontrei essa informação de forma explícita no regulamento fornecido."
- Não invente regras, prazos, exceções ou interpretações não presentes no texto.
- Sempre que possível, mencione o artigo relevante.
- Responda em português claro e objetivo.
"""

    resultados = []

    # Listas para acumular métricas individuais e depois calcular médias
    recall_scores = []
    mrr_scores = []
    keyword_scores = []

    for i, item in enumerate(dataset, start=1):
        pergunta = item["pergunta"]
        resposta_esperada = item["resposta_esperada"]
        artigos_esperados = item["artigos_esperados"]

        print(f"\n[{i}/{len(dataset)}] Avaliando: {pergunta}")

        # Etapa 1: retrieval
        retrieved_chunks, retrieved_scores = get_relevant_chunks(
            pergunta,
            chunk_embeddings,
            chunks,
            top_k=top_k
        )

        # Etapa 2: geração da resposta
        resposta_modelo = gerar_resposta(
            pergunta,
            retrieved_chunks,
            model_name,
            system_message
        )

        # Etapa 3: métricas
        recall_at_k = calcular_recall_at_k(artigos_esperados, retrieved_chunks)
        mrr = calcular_mrr(artigos_esperados, retrieved_chunks)
        keyword_recall = calcular_keyword_recall(resposta_esperada, resposta_modelo)

        recall_scores.append(recall_at_k)
        mrr_scores.append(mrr)
        keyword_scores.append(keyword_recall)

        # Salva resultados detalhados da pergunta atual
        resultados.append({
            "pergunta": pergunta,
            "resposta_esperada": resposta_esperada,
            "resposta_modelo": resposta_modelo,
            "artigos_esperados": artigos_esperados,
            "chunks_recuperados": [
                {
                    "id": chunk.get("id"),
                    "capitulo": chunk.get("capitulo"),
                    "artigo": chunk.get("artigo"),
                    "score": score,
                    "texto": chunk.get("texto")
                }
                for chunk, score in zip(retrieved_chunks, retrieved_scores)
            ],
            "metricas": {
                "recall_at_k": recall_at_k,
                "mrr": mrr,
                "keyword_recall": keyword_recall
            }
        })

    # Cálculo das métricas médias do experimento inteiro
    metricas_finais = {
        "total_perguntas": len(dataset),
        "recall_at_k_medio": mean(recall_scores) if recall_scores else 0,
        "mrr_medio": mean(mrr_scores) if mrr_scores else 0,
        "keyword_recall_medio": mean(keyword_scores) if keyword_scores else 0
    }

    return resultados, metricas_finais


# Salva resultados em JSON para observação específica de cada pergunta
def salvar_resultados(resultados, metricas_finais, output_path):

    output = {
        "metricas_finais": metricas_finais,
        "resultados": resultados
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    print(f"\nResultados salvos em: {output_path}")


###################################################################################


def main():

    #Adiciona as especificações do modelo, tipo, chunks, dataset de avaliação, saída, e o top_k desejado
    parser = argparse.ArgumentParser(description="Avaliação de RAG local com Ollama")
    parser.add_argument("--model", default="llama3:latest", help="Modelo de chat do Ollama")
    parser.add_argument("--chunks", default="chunks.json", help="Arquivo com os chunks")
    parser.add_argument("--dataset", default="evaluation_dataset.json", help="Arquivo com perguntas de avaliação")
    parser.add_argument("--output", default="evaluation_results.json", help="Arquivo de saída")
    parser.add_argument("--top_k", type=int, default=5, help="Quantidade de chunks recuperados")
    args = parser.parse_args()

    print("Carregando chunks...")
    chunks = carregar_json(args.chunks)

    print("Carregando dataset de avaliação...")
    dataset = carregar_json(args.dataset)

    # Gera embeddings de todos os chunks antes de iniciar o loop de avaliação
    chunk_embeddings = gerar_embeddings_chunks(chunks)

    # Executa o pipeline de avaliação
    resultados, metricas_finais = avaliar_rag(
        chunks=chunks,
        dataset=dataset,
        chunk_embeddings=chunk_embeddings,
        model_name=args.model,
        top_k=args.top_k
    )

    salvar_resultados(resultados, metricas_finais, args.output)

    print("\nMÉTRICAS FINAIS")
    print(json.dumps(metricas_finais, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()