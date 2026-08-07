from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Suggest a catchy blog title about {topic}.",
)

topic = input("Enter a topic for the blog title: ")

formatted_prompt = prompt.format(topic=topic)

blog_title = llm.predict(formatted_prompt)

print("Suggested Blog Title:", blog_title)
