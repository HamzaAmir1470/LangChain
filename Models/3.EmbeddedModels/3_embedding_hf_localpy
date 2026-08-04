from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Japan is Tokyo.",
    "The capital of the United States is Washington, D.C.",
]

vector = embedding.embed_documents(documents)

print(str(vector))
