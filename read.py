import re
import json
from pypdf import PdfReader


def extrair_texto(pdf_path):
    try:
        leitor = PdfReader(pdf_path)
        texto_inteiro = ""

        for pagina in leitor.pages:
            pagina_texto = pagina.extract_text()
            if pagina_texto:
                texto_inteiro += pagina_texto + "\n"
            else:
                texto_inteiro += "Texto nao encontrado\n"

        return texto_inteiro

    except FileNotFoundError:
        return f"Erro: Arquivo nao encontrado em {pdf_path}"
    except Exception as e:
        return f"Ocorreu um erro: {e}"


def limpar_texto(texto):
    # Remove espaços duplicados
    texto = re.sub(r'[ \t]+', ' ', texto)

    # Corrige muitas quebras
    texto = re.sub(r'\n{3,}', '\n\n', texto)

    # Remove espaços em excesso antes de quebra
    texto = re.sub(r' +\n', '\n', texto)

    return texto.strip()


def dividir_por_artigos(texto):
    """
    Divide o texto em chunks estruturados por:
    - CAPÍTULO
    - Artigo
    Mantém os § dentro do mesmo artigo.
    """
    linhas = texto.split("\n")

    chunks = []
    chunk_atual = []
    capitulo_atual = ""

    padrao_capitulo = re.compile(r'^(CAP[IÍ]TULO\b.*)', re.IGNORECASE)
    padrao_artigo = re.compile(r'^(Art\.?\s*\d+º?.*)', re.IGNORECASE)

    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue

        # Detecta capítulo
        if padrao_capitulo.match(linha):
            capitulo_atual = linha
            continue

        # Detecta novo artigo
        if padrao_artigo.match(linha):
            # Fecha chunk anterior
            if chunk_atual:
                texto_chunk = "\n".join(chunk_atual).strip()
                chunks.append({
                    "capitulo": capitulo_atual,
                    "texto": texto_chunk
                })

            # Começa novo chunk
            chunk_atual = []
            if capitulo_atual:
                chunk_atual.append(capitulo_atual)
            chunk_atual.append(linha)

        else:
            # Continua no mesmo artigo
            chunk_atual.append(linha)

    # Salva o último chunk
    if chunk_atual:
        texto_chunk = "\n".join(chunk_atual).strip()
        chunks.append({
            "capitulo": capitulo_atual,
            "texto": texto_chunk
        })

    return chunks


def subdividir_chunk_grande(chunk_texto, tamanho_max=1800):
    """
    Se um artigo ficar muito grande, divide em subchunks por parágrafos (§ ou frases).
    """
    if len(chunk_texto) <= tamanho_max:
        return [chunk_texto]

    # Tenta dividir por § primeiro
    partes = re.split(r'(?=§\s*\d+º?)', chunk_texto)

    if len(partes) == 1:
        # Se não houver §, divide por frases
        frases = re.split(r'(?<=[.!?])\s+', chunk_texto)
        partes = frases

    subchunks = []
    atual = ""

    for parte in partes:
        parte = parte.strip()
        if not parte:
            continue

        if len(atual) + len(parte) + 2 <= tamanho_max:
            atual += parte + "\n\n"
        else:
            if atual.strip():
                subchunks.append(atual.strip())
            atual = parte + "\n\n"

    if atual.strip():
        subchunks.append(atual.strip())

    return subchunks


def processar_chunks_estruturados(chunks_artigos, tamanho_max=1800):
    """
    Garante que cada chunk final tenha tamanho adequado.
    """
    chunks_finais = []
    contador = 1

    for item in chunks_artigos:
        capitulo = item["capitulo"]
        texto = item["texto"]

        subchunks = subdividir_chunk_grande(texto, tamanho_max=tamanho_max)

        for sub in subchunks:
            chunks_finais.append({
                "id": contador,
                "capitulo": capitulo,
                "texto": sub
            })
            contador += 1

    return chunks_finais


def salvar_chunks_json(chunks, arquivo_saida="chunks.json"):
    with open(arquivo_saida, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

    print(f"Chunks salvos em {arquivo_saida}")


# ==========================
# EXECUÇÃO
# ==========================

pdf_arquivo_path = "resolucao.pdf"

texto = extrair_texto(pdf_arquivo_path)
texto_limpo = limpar_texto(texto)

chunks_artigos = dividir_por_artigos(texto_limpo)
chunks_finais = processar_chunks_estruturados(chunks_artigos, tamanho_max=1800)

print(f"Total de artigos/chunks detectados: {len(chunks_finais)}\n")

for chunk in chunks_finais[:5]:
    print(f"\n--- CHUNK {chunk['id']} ---")
    print(f"Capítulo: {chunk['capitulo']}")
    print(chunk["texto"][:1200])
    print("\n" + "=" * 80)

salvar_chunks_json(chunks_finais, "chunks.json")