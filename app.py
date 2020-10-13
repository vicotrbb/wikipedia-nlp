import streamlit as st
import numpy as np
import pandas as pd
import json

generations_info = None
generations = []
topics = []
model_version = []
temperature = 0.1
predict_len = 100
model_gen = None
model_version = None

def setup():
    load_data()
    st.title('Wikipedia NLP')
    st.text(
        """
        """
    )

def main():
    temperature = st.sidebar.slider(
        label='Predict temperature',
        min_value=0.1, 
        max_value=2.,
        step=0.1)

    predict_len = st.sidebar.slider(
        label='Predicted text length',
        min_value=100, 
        max_value=500,
        step=10)
    
    model_gen = st.sidebar.selectbox(
        label='Model generation',
        options=generations,
        index=-1
    )
    
    model_version = st.sidebar.selectbox(
        label='Model version',
        options=model_version,
        index=-1
    )
    
@st.cache
def load_data():
    with open('Files/generations.json') as json_file:
        generations_info = json.load(json_file)
    generations = generations_info.keys()
    
    model_version = generations_info[model_gen].models.keys()


if __name__ == "__main__":
    setup()
    main()
    
if __name__ != "__main__":
    setup()
    main()
