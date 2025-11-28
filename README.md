# MAGI-02: Multi-Agent LLM Council

A multi-agent LLM council using Strands with Search and RAG capability, inspired by Neon Genesis Evangelion and Pewdiepie.

## Features

- **Multi-Agent Council**: Three specialized AI agents (MELCHIOR, BALTHASAR, CASPER) that deliberate and provide consensus-based responses
- **Local RAG**: Vector-based document retrieval using ChromaDB and sentence-transformers
- **Web Search**: DuckDuckGo-based web search capability
- **LM Studio Integration**: Use local LLMs via LM Studio's OpenAI-compatible API
- **Streamlit Frontend**: Interactive web interface for chatting with the council

## Installation

### Prerequisites

1. Python 3.10+
2. [LM Studio](https://lmstudio.ai/) installed and running with a model loaded
3. Start LM Studio's local server (default: `http://localhost:1234/v1`)

### Setup

```bash
# Clone the repository
git clone https://github.com/calvpang/MAGI-02.git
cd MAGI-02

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

## Configuration

Create a `.env` file in the project root:

```env
# LM Studio Configuration
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=local-model

# RAG Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Usage

### Adding Documents for RAG

Place your documents (PDF, TXT, MD files) in the `data/documents` folder, then run:

```bash
python -m magi.tools.rag --ingest
```

### Running the Streamlit Frontend

```bash
streamlit run src/magi/app.py
```

### Using the Council Programmatically

```python
from magi.agents.council import MAGICouncil

# Initialize the council
council = MAGICouncil()

# Ask a question
response = council.deliberate("What is the meaning of life?")
print(response)
```

## Architecture

### Agents

The MAGI system consists of three specialized agents:

1. **MELCHIOR-1** (Scientist): Focuses on logical analysis and scientific reasoning
2. **BALTHASAR-2** (Mother): Provides empathetic and nurturing perspectives
3. **CASPER-3** (Woman): Offers balanced and intuitive insights

### Tools

- **RAG Tool**: Local document search using ChromaDB vector store
- **Search Tool**: Web search using DuckDuckGo

## Development

```bash
# Run tests
pytest

# Run linter
ruff check src/

# Format code
ruff format src/
```

## License

MIT License
