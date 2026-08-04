from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()

# Switch to flash-lite or 2.5-pro to bypass the flash quota limit
llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm.invoke("Write a short story about a robot learning to love.")

print("LLM Answer:\n", result.content)
