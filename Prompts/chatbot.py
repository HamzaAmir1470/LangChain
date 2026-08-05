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

chat_history = []

while True:
    user_input = input("You: ")

    # Check for exit before appending to history
    if user_input.strip().lower() == "exit":
        break

    chat_history.append({"role": "user", "content": user_input})

    result = model.invoke(chat_history)

    # LangChain requires lowercase 'assistant' or 'ai'
    chat_history.append({"role": "assistant", "content": result.content})

    print("Chat Model Answer:\n", result.content)

print("\nChat session ended.")
print("Chat History:")
for message in chat_history:
    print(f"{message['role'].capitalize()}: {message['content']}")
