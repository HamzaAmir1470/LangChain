from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableBranch,
    RunnablePassthrough,
)

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}.",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Summarize the following {text}.",
    input_variables=["text"],
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

report_chain = prompt1 | model | parser
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 200, prompt2 | model | parser),
    RunnablePassthrough(),
)

final_chain = RunnableSequence(report_chain, branch_chain)

result = final_chain.invoke({"topic": "Muslim VS Non-Muslim"})

print(result)
