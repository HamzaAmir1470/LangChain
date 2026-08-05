from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

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

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Black Hole"})

print(result)
