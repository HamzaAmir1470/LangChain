from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
)
model = ChatHuggingFace(llm=llm)


# 1st prompt => detailed report
template1 = PromptTemplate(
    template="Write a detailed report on the following topic: {topic}",
    input_variables=["topic"],
)

template2 = PromptTemplate(
    template="Write a 5 lined summary of the following report: {text}",
    input_variables=["text"],
)

prompt1 = template1.format(topic="Black Hole")

result = model.invoke(prompt1)

prompt2 = template2.format(text=result.content)

result2 = model.invoke(prompt2)

print("Detailed Report:\n", result.content)
print("\nSummary:\n", result2.content)
