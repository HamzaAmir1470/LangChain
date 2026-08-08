import os

# Prevent Hugging Face rate-limit requests
os.environ["HF_HUB_OFFLINE"] = "1"

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"local_files_only": True},
)

# 1. Initialize DB
vector_store = Chroma(
    embedding_function=embedding,
    persist_directory="my_chroma_db",
    collection_name="sample",
)

# 2. Add Documents with explicit IDs
docs = [
    Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"},
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"},
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"},
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"},
    ),
]
doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]

vector_store.add_documents(docs, ids=doc_ids)

# 3. Perform Similarity Search
print("--- Similarity Search Results ---")
results = vector_store.similarity_search(query="Who among these are a bowler?", k=2)
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")

# 4. View Stored Database Records
print("--- Stored DB Records ---")
print(vector_store.get(include=["documents", "metadatas"]))