import streamlit as st
from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

st.title('Langchain Demo With NER Of Token Classification')
input_text=st.text_input("Input text")

model_path = "Chessmen/token_classifier_scratch"
ner_task  = pipeline(
    "token-classification",
    model=model_path,
    aggregation_strategy="simple",
)

if input_text:
	st.write(ner_task(input_text))
    