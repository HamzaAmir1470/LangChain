import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables (.env file)
load_dotenv()

# Set up Hugging Face Endpoint LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-7B-Instruct",
    task="text-generation",
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
)

# Wrap with ChatHuggingFace for proper chat template handling
model = ChatHuggingFace(llm=llm)

st.header("Research Tool")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
    ],
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"],
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (detailed explanation)",
    ],
)

template = load_prompt("template.json")

# Pipe prompt into model using LangChain Expression Language (LCEL)
chain = template | model

if st.button("Summarize"):
    with st.spinner("Generating summary..."):
        result = chain.invoke(
            {
                "paper_input": paper_input,
                "style_input": style_input,
                "length_input": length_input,
            }
        )

        # Print to terminal
        print("Chat Model Answer:\n", result.content)

        # Display on Streamlit UI
        st.write(result.content)
