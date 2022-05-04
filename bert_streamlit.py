import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text

st.set_page_config(
    page_title ="Movie Comment Analyzer",
    page_icon=":shark:"
)

st.title("Movie Comment Analyzer (v1.0)")


st.header("Introduction")
st.caption(
    """
    You can check the sentimental of your input setences. 
    The other way is to upload a text file of comments (each comment in one line) and to analyze the sentiment.
    """
)

# load model
model = tf.keras.models.load_model("./imdb_bert")

# user input text
usr_input = st.text_area("Type your sentence here")
st.caption(usr_input)
if usr_input:
    results = tf.sigmoid(model(tf.constant(usr_input)))
    st.caption(results)


# user upload files
usr_data = st.file_uploader("Upload a text file")




