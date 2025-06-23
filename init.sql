-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table for storing document vectors
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index for faster similarity search
CREATE INDEX IF NOT EXISTS document_embeddings_embedding_idx
ON document_embeddings USING ivfflat (embedding vector_cosine_ops);
