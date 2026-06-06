from utils.pdf_loader import load_pdf
from utils.chunker import chunk_documents
from utils.embeddings import get_embeddings
from utils.vector_store import create_vector_store

documents = load_pdf("data/sample.pdf")

chunks = chunk_documents(documents)

embeddings = get_embeddings()

create_vector_store(
    chunks,
    embeddings
)

print("Vector Database Created")