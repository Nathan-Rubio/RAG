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
    texto = re.sub(r"[ \t]+", " ", texto)

    # Corrige quebras excessivas
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    # Remove espaços antes de quebra
    texto = re.sub(r" +\n", "\n", texto)

    # Corrige alguns ruídos comuns
    texto = texto.replace("Procedimento s", "Procedimentos")
    texto = texto.replace("C .asos", "Casos")

    return texto.strip()


def remover_preambulo(texto):
    """
    Remove tudo antes do primeiro CAPÍTULO I ou Art. 1º.
    Isso evita criar chunk com cabeçalho e menções soltas a artigos.
    """
    match = re.search(r"(CAP[IÍ]TULO\s+I\b|Art\.?\s*1º)", texto, re.IGNORECASE)
    if match:
        return texto[match.start():]
    return texto


def normalizar_capitulo(linha):
    """
    Limpa e normaliza a linha do capítulo.
    """
    linha = linha.strip()
    linha = re.sub(r"\s+", " ", linha)
    return linha


def extrair_artigo_do_inicio(texto):
    """
    Extrai o artigo apenas se ele aparecer no início estrutural do chunk,
    e não em qualquer menção solta dentro do texto.
    """
    match = re.search(r"^\s*(?:CAP[IÍ]TULO\b.*\n)?\s*(Art\.?\s*\d+º?)", texto, re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def dividir_por_artigos(texto):
    """
    Divide o texto em chunks por artigo, herdando corretamente o capítulo atual.
    Cada chunk final representa 1 artigo completo.
    """
    linhas = texto.split("\n")

    chunks = []
    chunk_atual = []
    capitulo_atual = None

    padrao_capitulo = re.compile(r"^\s*(CAP[IÍ]TULO\b.*)", re.IGNORECASE)
    padrao_artigo = re.compile(r"^\s*(Art\.?\s*\d+º?.*)", re.IGNORECASE)

    for linha in linhas:
        linha = linha.strip()

        if not linha:
            continue

        match_cap = padrao_capitulo.match(linha)
        if match_cap:
            capitulo_atual = normalizar_capitulo(match_cap.group(1))
            continue

        match_art = padrao_artigo.match(linha)
        if match_art:
            # Fecha o chunk anterior
            if chunk_atual:
                texto_chunk = "\n".join(chunk_atual).strip()
                artigo = extrair_artigo_do_inicio(texto_chunk)

                chunks.append({
                    "capitulo": capitulo_atual,
                    "artigo": artigo,
                    "texto": texto_chunk
                })

            # Começa novo chunk
            chunk_atual = []
            if capitulo_atual:
                chunk_atual.append(capitulo_atual)
            chunk_atual.append(linha)
        else:
            # Só adiciona linhas se já estivermos dentro de um artigo
            if chunk_atual:
                chunk_atual.append(linha)

    # Salva o último chunk
    if chunk_atual:
        texto_chunk = "\n".join(chunk_atual).strip()
        artigo = extrair_artigo_do_inicio(texto_chunk)

        chunks.append({
            "capitulo": capitulo_atual,
            "artigo": artigo,
            "texto": texto_chunk
        })

    return chunks


def subdividir_chunk_grande(item, tamanho_max=1800):
    """
    Se um artigo ficar muito grande, divide em subchunks,
    preservando capitulo e artigo.
    """
    capitulo = item["capitulo"]
    artigo = item["artigo"]
    texto = item["texto"]

    if len(texto) <= tamanho_max:
        return [{
            "capitulo": capitulo,
            "artigo": artigo,
            "texto": texto
        }]

    # Tenta dividir por § primeiro
    partes = re.split(r"(?=§\s*\d+º?)", texto)

    # Se não houver §, divide por frases
    if len(partes) == 1:
        partes = re.split(r"(?<=[.!?])\s+", texto)

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
                subchunks.append({
                    "capitulo": capitulo,
                    "artigo": artigo,
                    "texto": atual.strip()
                })
            atual = parte + "\n\n"

    if atual.strip():
        subchunks.append({
            "capitulo": capitulo,
            "artigo": artigo,
            "texto": atual.strip()
        })

    return subchunks


def processar_chunks_estruturados(chunks_artigos, tamanho_max=1800):
    """
    Garante tamanho adequado e adiciona ID final.
    """
    chunks_finais = []
    contador = 1

    for item in chunks_artigos:
        subchunks = subdividir_chunk_grande(item, tamanho_max=tamanho_max)

        for sub in subchunks:
            chunks_finais.append({
                "id": contador,
                "capitulo": sub["capitulo"],
                "artigo": sub["artigo"],
                "texto": sub["texto"]
            })
            contador += 1

    return chunks_finais


def filtrar_chunks_invalidos(chunks):
    """
    Remove chunks sem artigo identificado.
    """
    return [chunk for chunk in chunks if chunk.get("artigo")]


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
texto_limpo = remover_preambulo(texto_limpo)

chunks_artigos = dividir_por_artigos(texto_limpo)
chunks_finais = processar_chunks_estruturados(chunks_artigos, tamanho_max=1800)
chunks_finais = filtrar_chunks_invalidos(chunks_finais)

print(f"Total de artigos/chunks detectados: {len(chunks_finais)}\n")

for chunk in chunks_finais[:5]:
    print(f"\n--- CHUNK {chunk['id']} ---")
    print(f"Capítulo: {chunk['capitulo']}")
    print(f"Artigo: {chunk['artigo']}")
    print(chunk["texto"][:1200])
    print("\n" + "=" * 80)

salvar_chunks_json(chunks_finais, "chunks.json")