from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    """Chunker class for splitting PDF into chunks"""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def load_pdf(self) -> list[Document]:
        """Load PDF into documents"""
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        return documents

    def split_documents(self) -> list[Document]:
        """Split documents into chunks"""
        documents = self.load_pdf()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Number of tokens per chunk
            chunk_overlap=200,  # Number of tokens to overlap between chunks
            separators=[
                "\n\n",
                "\n",
                " ",
                "",
            ],  # List of separators to use for splitting
        )
        documents = splitter.split_documents(documents)
        return documents
