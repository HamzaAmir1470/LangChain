from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
    max_new_tokens=256,
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=["question", "text"],
)

parser = StrOutputParser()

url = "https://www.daraz.pk/products/samsung-galaxy-s26-ultra-256-gb-i1947937840-s14012836155.html?scm=1007.51610.379274.0&pvid=7b6d5666-e5f8-45b8-b727-640067ae8f17&search=flashsale&spm=a2a0e.tm80335142.FlashSale.d_1947937840"
loader = WebBaseLoader(url)

docs = loader.load()


chain = prompt | model | parser

print(
    chain.invoke(
        {
            "question": "What is the product that we are talking about?",
            "text": docs[0].page_content,
        }
    )
)
