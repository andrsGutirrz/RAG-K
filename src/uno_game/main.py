#!/usr/bin/env python3
"""
UNO Game Rules RAG System
Uses AWS Bedrock, LangChain, FAISS, and PostgreSQL for local RAG implementation
"""

import os
import psycopg2
from dotenv import load_dotenv
import json

from chunker import Chunker
import boto3
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import numpy as np

# Load environment variables
load_dotenv()


class UNOGameRAG:
    def __init__(self):
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "database": os.getenv("DB_NAME", "unodb"),
            "user": os.getenv("DB_USER", "uno"),
            "password": os.getenv("DB_PASSWORD", "admin123"),
        }

        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "amazon.titan-embed-text-v1"
        )
        self.llm_model = os.getenv("LLM_MODEL", "amazon.titan-text-express-v1")
        self.pdf_path = "src/data/UNO.pdf"

        # Initialize AWS clients
        session = boto3.Session(profile_name="personal")
        self.bedrock = session.client("bedrock-runtime", region_name=self.aws_region)
        self.embeddings = BedrockEmbeddings(
            client=self.bedrock, model_id=self.embedding_model
        )
        self.llm = BedrockLLM(client=self.bedrock, model_id=self.llm_model)
        self.vectorstore = None
        self.qa_chain = None
        self.chunker = Chunker(self.pdf_path)

    def check_vectors_in_db(self) -> bool:
        """Check if vectors already exist in database"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM document_embeddings")
            count = cur.fetchone()[0]
            conn.close()
            return count > 0
        except Exception as e:
            print(f"Error checking database: {e}")
            return False

    def save_vectors_to_db(self, documents: list[Document]):
        """Save vectors to PostgreSQL"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()

            print("Creating embeddings and saving to database...")
            for i, doc in enumerate(documents):
                print(f"Processing document {i+1}/{len(documents)}")

                # Create embedding
                embedding = self.embeddings.embed_query(doc.page_content)

                # Insert into database
                cur.execute(
                    """INSERT INTO document_embeddings (content, embedding, metadata)
                       VALUES (%s, %s, %s)""",
                    (doc.page_content, embedding, json.dumps({})),
                )

            conn.commit()
            conn.close()
            print("Vectors saved to database successfully!")

        except Exception as e:
            print(f"Error saving to database: {e}")
            raise

    def load_vectors_from_db(self) -> list[tuple[str, list[float]]]:
        """Load vectors from PostgreSQL and convert embeddings to float lists"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("SELECT content, embedding FROM document_embeddings")
            results = cur.fetchall()
            conn.close()

            # Convert vector strings to float lists
            processed_results = []
            for content, embedding_str in results:
                # Remove brackets and split by commas
                embedding_list = [
                    float(x.strip()) for x in embedding_str.strip("[]").split(",")
                ]
                processed_results.append((content, embedding_list))

            return processed_results
        except Exception as e:
            print(f"Error loading from database: {e}")
            return []

    def create_vectorstore(self):
        """Create or load FAISS vectorstore"""
        if self.check_vectors_in_db():
            print("Loading vectors from database...")
            db_vectors = self.load_vectors_from_db()

            if db_vectors:
                texts = [item[0] for item in db_vectors]
                embeddings_matrix = np.array([item[1] for item in db_vectors])

                # Use pre-computed embeddings from database
                text_embeddings = [
                    (texts[i], embeddings_matrix[i]) for i in range(len(texts))
                ]
                self.vectorstore = FAISS.from_embeddings(
                    text_embeddings, self.embeddings
                )
                print("Vectorstore loaded from database!")
            else:
                print("No vectors found in database, creating new ones...")
                self._create_new_vectorstore()
        else:
            print("No vectors in database, creating new ones...")
            self._create_new_vectorstore()

    def _create_new_vectorstore(self):
        """Create new vectorstore from PDF"""
        documents = self.chunker.split_documents()

        # Save to database
        self.save_vectors_to_db(documents)

        # Create FAISS vectorstore
        texts = [doc.page_content for doc in documents]
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        print("New vectorstore created!")

    def setup_qa_chain(self):
        """Setup QA chain using new langchain approach"""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 5,
                "threshold": 0.5,
            }
        )

        system_prompt = (
            "You are an expert UNO game rules assistant. Your task is to provide accurate and concise answers "
            "about UNO game rules based on the provided context. Follow these instructions carefully:\n\n"
            "1. Always use the provided context to answer questions. If the context doesn't contain the answer, "
            "   explicitly state that the information is not available in the provided context.\n"
            "2. Provide clear, specific, and actionable answers based on official UNO rules.\n"
            "3. If multiple answers are possible, explain the different scenarios and their implications.\n"
            "4. Format your answers in a clear, structured way, using bullet points or numbered lists when appropriate.\n"
            "5. If the question is about strategy, base your answer on official rules and common tournament practices.\n"
            '6. Never make assumptions or guess when the context is unclear. and simply say "I don\'t know".\n\n'
            "Context: {context}"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        self.qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    def ask_question(self, question: str) -> str:
        """Ask question about UNO rules"""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup() first.")

        response = self.qa_chain.invoke({"input": question})
        return response["answer"]

    def setup(self):
        """Initialize the RAG system"""
        print("Setting up UNO Game RAG system...")
        self.create_vectorstore()
        self.setup_qa_chain()
        print("RAG system ready!")


def main():
    """Main function"""
    rag = UNOGameRAG()
    rag.setup()

    print("\n🎮 UNO Game Rules Assistant")
    print("Ask questions about UNO rules (type 'quit' to exit)")
    print("-" * 50)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye! 👋")
            break

        if not question:
            continue

        try:
            answer = rag.ask_question(question)
            print(f"\n🤖 Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
