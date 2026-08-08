from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}.",
    input_variables=["topic"],
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
)

prompt2 = PromptTemplate(
    template="Now, explain the following joke - {text}",
    input_variables=["text"],
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, prompt2, model, parser)

result = chain.invoke({"topic": "Black Holes"})

print("Chat Model Answer:\n", result)
