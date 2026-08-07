from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)

prompt = PromptTemplate(
    input_variables=["topic"], template="Suggest a catchy blog title about {topic}"
)

chain = LLMChain(llm=llm, prompt=prompt)

topic = input("Enter a topic for the blog title: ")

output = chain.run(topic)
print("Suggested blog title:", output)