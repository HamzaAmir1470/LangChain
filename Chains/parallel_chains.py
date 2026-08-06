from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
)

llm2 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template="Generate a short and simple notes of the following text: {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text: {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and {quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model | parser,
        "quiz": prompt2 | model2 | parser,
    }
)

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

result = chain.invoke(
    {
        "text": "The Earth is the third planet from the Sun and the only astronomical object known to harbor life. About 71% of Earth's surface is covered with water, mostly by its oceans. The remaining 29% consists of continents and islands. Earth's atmosphere consists mostly of nitrogen and oxygen."
    }
)

print("Result:\n", result)

chain.get_graph().print_ascii()
