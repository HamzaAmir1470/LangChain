from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}.",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Generate a Linked in post about {topic}.",
    input_variables=["topic"],
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1, model, parser),
        "linkedin_post": RunnableSequence(prompt2, model, parser),
    }
)

parallel_result = parallel_chain.invoke({"topic": "Artificial Intelligence"})

print(parallel_result["tweet"])
print(parallel_result["linkedin_post"])