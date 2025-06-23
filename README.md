# RAG-K: Local RAG Implementation with AWS Bedrock

A simple yet powerful implementation of Retrieval-Augmented Generation (RAG) using Python, Langchain, and AWS Bedrock. This demo project showcases how to create a local RAG system using AWS Titan models and PostgreSQL as a vector database.

## 📚 Definitions

### 1. RAG (Retrieval-Augmented Generation)
RAG is a machine learning architecture that combines information retrieval with language generation. It retrieves relevant documents from a knowledge base to generate more accurate and context-aware responses.

### 2. Embeddings
Embeddings are numerical representations of text data in a high-dimensional space. They capture semantic meaning and relationships between words and documents, enabling efficient similarity searches.

### 3. Vector Database
A specialized database designed to store and query vector embeddings. It enables fast similarity searches and is crucial for RAG systems to retrieve relevant documents efficiently.

### 4. Retriever
A component in RAG systems that uses embeddings to find the most relevant documents from a knowledge base based on user queries.

## 🎮 Project Overview

This demo project implements a RAG system specifically for the UNO board game. It allows users to ask questions about UNO game rules and receive accurate answers based on the game's documentation.

## 🛠️ Requirements

- Python 3.11 or higher
- AWS CLI
- AWS Credentials configured in `~/.aws/credentials`
- AWS Bedrock with access to Titan models
- PostgreSQL database

## 🔧 Setup

1. Install UV package manager:
```bash
pip install uv
```

2. Install pre-commit hooks:
```bash
uv run pre-commit install
```

3. Set up your environment variables:
Create a `.env` file based on `.env.template` with the following variables:
- Database configuration
- AWS Bedrock model configurations

4. Initialize the PostgreSQL database:
We use docker-compose to run the database.
```bash
docker compose up -d
```

## 📦 Dependencies

The project uses the following key dependencies:
- `langchain`: For building the RAG pipeline
- `langchain-aws`: For AWS Bedrock integration
- `pgvector`: For PostgreSQL vector storage
- `pypdf`: For PDF document processing

## 🚀 Usage

1. Load your UNO game rules PDF into the system
2. Ask questions about UNO game rules
3. The system will retrieve relevant sections from the PDF and generate accurate answers

### Run the application

1. Using `uv` (local development):
```bash
uv run src/uno_game/main.py
```

## 📊 Models Used

- **Embedding Model**: `amazon.titan-embed-text-v1`
- **Language Model**: `amazon.titan-text-express-v1`

## 📝 Code Style

The project uses:
- `black` for code formatting
- `pre-commit` for code quality checks

## 📄 License

MIT License
