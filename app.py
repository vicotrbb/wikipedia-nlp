import streamlit as st
import numpy as np
import pandas as pd

generations = None

def setup()
    st.title('Wikipedia NLP')
    with open('Files/generations.json') as json_file:
        generations = json.load(json_file)


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
        options=[],
        index=-1
    )
    
@st.cache
def load_data():


if __name__ == "__main__":
    setup()
    main()
    
if __name__ != "__main__":
    setup()
    main()
