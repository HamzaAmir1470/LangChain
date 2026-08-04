import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Write a short story about a robot learning to love.")
print("Chat Model Answer:\n", result.content)