import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Initialize local Hugging Face embedding model
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast, dark-colored fox leaps over a sleepy canine.",
    "An agile, russet fox vaults over a lethargic hound.",
    "Hamza is a software engineer who loves to code and solve complex problems.",
    "Usman is a data scientist who enjoys analyzing data and building predictive models.",
    "Javascript is a versatile programming language used for web development, server-side scripting, and building interactive applications.",
]

query = "tell me about javascript"

document_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], document_embeddings)[0]

# Added reverse=True to get highest similarity score, not lowest
index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]

print("Query:", query)
print("Most similar document:", documents[index])
print("Similarity score:", score)

# from langchain_openai import OpenAIEmbeddings

# from dotenv import load_dotenv

# from sklearn.metrics.pairwise import cosine_similarity

# import numpy as np

# load_dotenv()

# embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

# documents = [
#     "The quick brown fox jumps over the lazy dog.",
#     "A fast, dark-colored fox leaps over a sleepy canine.",
#     "An agile, russet fox vaults over a lethargic hound.",
#     "Hamza is a software engineer who loves to code and solve complex problems.",
#     "Usman is a data scientist who enjoys analyzing data and building predictive models.",
#     "Javascript is a versatile programming language used for web development, server-side scripting, and building interactive applications.",
# ]

# query = "tell me about javascript"

# document_embeddings = embedding.embed_documents(documents)

# query_embedding = embedding.embed_query(query)

# scores = cosine_similarity([query_embedding], document_embeddings)[0]

# index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])

# print("Query:", query)
# print("Most similar document:", documents[index])
# print("Similarity score:", score)
