import os
import streamlit as st                          # pip install streamlit

from dotenv import load_dotenv, find_dotenv     # pip install python-dotenv
from langchain import HuggingFaceHub            # pip install langchain && pip install huggingface_hub


# Load the HuggingFaceHub API token from the .env file
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]


# Setup the app
# Start the app with: streamlit run streamlit_app.py
st.title('LLM App')
prompt = st.text_input('Was willst du wissen?')


# Load the LLM model from the HuggingFaceHub
repo_id = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500}
)

# show stuff to the screen
if prompt:
    response = falcon_llm(prompt)
    st.write(response)
    

