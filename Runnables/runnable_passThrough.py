from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableSequence,
    RunnableParallel,
    RunnablePassthrough,
)

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}.",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Now, explain the following joke - {text}",
    input_variables=["text"],
)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel(
    {
        "joke": joke_gen_chain,
        "explanation": RunnableSequence(prompt2, model, parser),
    }
)
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic": "Artificial Intelligence"})

print(result["joke"])
print(result["explanation"])
