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

## 4. Run Files

```
python read.py
python localrag.py
```
