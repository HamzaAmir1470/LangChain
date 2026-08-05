from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

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

chat_history = [
    SystemMessage(
        content="You are a helpful assistant. You will answer questions about research papers in a clear and concise manner."
    ),
]

while True:
    user_input = input("You: ")

    # Check for exit before appending to history
    if user_input.strip().lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    result = model.invoke(chat_history)

    # LangChain requires lowercase 'assistant' or 'ai'
    chat_history.append(AIMessage(content=result.content))

    print("Chat Model Answer:\n", result.content)

print("\nChat session ended.")
print("Chat History:", chat_history)
