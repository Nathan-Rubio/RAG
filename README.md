# Introduction

Implementação de uma RAG rodando localmente com Ollama para responder perguntas com base em um regulamento academico

## Funcionamento

### 1. Preparação dos dados

O regulamento em PDF é convertido em texto e dividido em chunks estruturados por artigo, feito automaticamente com read.py, mas modificado manualmente para consertar erros específicos do PDF

Também foi criado um dataset em json de perguntas relacionadas ao artigo específico para funcionar como base da avaliação da eficiência da RAG. Este dataset é de uso específico para o conteúdo do pdf original, para rodar a avaliação com outro tipo de conteúdo é necessário criar outro dataset de perguntas de avaliação.

A qualidade do sistema depende fortemente da estrutura dos chunks, por isso é necessário uma revisão específica para cada tipo de arquivo, com o objetivo de gerar a melhor estrutura possível para a RAG

### Geração de Embeddings

Cada chunk é transformado em um vetor numérico (embedding) usando o modelo
```
mxbai-embed-large
```
Esses vetores representam semanticamente o conteúdo do texto

### Retrieval

Quando uma pergunta é feita ao modelo a pergunta também passa pelo processo de embedding, depois é calculado a similaridade da pergunta em relação aos chunks e os top_k chunks mais relevantes são selecionados

### Gerar resposta

Os chunks recuperados são enviados junto com a pergunta para um modelo local que gera a resposta

O modelo é instruído a responder apenas com base no contexto e citar os artigos se for possível

### Avaliação

O sistema avalia o conjunto de perguntas usando as métricas:

Recall@k - verifica se o artigo correto foi encontrado
MRR (Mean Reciprocal Rank) - avalia a posição do artigo correto no ranking
Keyword Recall: Mede a relação de palavras entre a resposta esperada e a gerada

Os resultados são salvos em ```evaluation_results.json```



# Steps

## 1. Download Ollama

Linux: 
```
curl -fsSL https://ollama.com/install.sh | sh
```
Windows:
```
irm https://ollama.com/install.ps1 | iex
```

## 2. Go to the local file

run

```
ollama pull mxbai-embed-large
ollama pull llama3
```

## 3. Create venv

Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

Windows:
```
python -m venv .venv
venv\Scripts\activate.bat
```

## 4. Install requirements

```
pip install -r requirements.txt
```

## 5. Run Files

Para rodar o read e automaticamente criar o json do pdf (Foi necessário manualmente modificar o arquivo por conta de falhas na leitura do pdf, o json disponível já está em um bom formato)
```
python read.py
```

Para rodar a RAG e fazer perguntas
```
python localrag.py
```

Para rodar a avaliação do modelo
```
python evaluate_rag.py
```
