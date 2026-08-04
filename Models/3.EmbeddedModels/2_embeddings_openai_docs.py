from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimension=32)

documents = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
    "The capital of Japan is Tokyo.",
    "The capital of the United States is Washington, D.C.",
]

result = embeddings.embed_documents(documents)

print(str(result))
