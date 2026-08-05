from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

# Set up Hugging Face Endpoint LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
)

# Wrap with ChatHuggingFace for proper chat template handling
model = ChatHuggingFace(llm=llm)

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    result = model.invoke(user_input)
    print("Chat Model Answer:\n", result.content)
