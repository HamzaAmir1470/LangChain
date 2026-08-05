from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import ResponseSchema, StructuredOutputParser
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
)
model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1", description="A fact about the world."),
    ResponseSchema(name="fact_2", description="A fact about the EFTP."),
    ResponseSchema(name="fact_3", description="A fact about the Sea."),
    ResponseSchema(name="fact_4", description="A fact about the Sky."),
    ResponseSchema(name="fact_5", description="A fact about the Space."),
]

parser = StructuredOutputParser.from_response_schemas(schemas=schema)

template = PromptTemplate(
    template="Give me 5 facts about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = template | model | parser

result = chain.invoke({"topic": "World"})

print("Result:\n", result)
