from langchain_anthropic import ChatAnthropic

from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model="claude-3.5-sonnet-20241022")

model_result = model.invoke("Write a short story about a robot learning to love.")

print("ChatModel Answer:\n", model_result.content)
