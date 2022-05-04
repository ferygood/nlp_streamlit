from email import header
from io import StringIO
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import altair as alt


st.set_page_config(
    page_title ="Movie Comment Analyzer",
    page_icon=":shark:"
)

st.title("Movie Comment Analyzer (v1.0)")


st.header("Introduction")
st.caption(
    """
    You can check the sentimental of your input sentences. The other way is to upload a text file of 
    comments (each comment in one line) and to analyze the sentiment. \n
    This is a BERT model which trained using imdb comments data.
    """
)

# load model
model = tf.saved_model.load("./imdb_bert")

# user input text
st.subheader("(1) Directly type your comment in the text area below.")
usr_input = st.text_area("Type comment.")

sentence = []
if usr_input:
    sentence.append(usr_input)
    results = tf.sigmoid(model(tf.constant(sentence))).numpy()
    results = results.astype(float)

    score = round(results[0][0]*100, 2) 

    st.caption(f"This comment gets {score}/100 points.")    


# user upload files
st.subheader("(2) Upload multiple comments in a text file.")
st.caption("The format of your text file should follow each comment in one line.")
usr_data = st.file_uploader(".txt file", type=".txt")

if usr_data:
    df = pd.read_csv(usr_data, header=None)
    num_comment = len(df)
    st.write(f"There are {num_comment} comments in the file.")

    comment_list = []
    for i in range(len(df)):
        comment_list.append(df.iloc[i, 0])
    
    results = tf.sigmoid(model(tf.constant(comment_list))).numpy()
    results = results*100
    df_results = pd.DataFrame(results, columns=["Score/100"])
    avg_score = np.average(results).astype(float)
    avg_score = round(avg_score, 2)
    
    st.write(f"Average score of all comments is {avg_score}/100.")
    
    c = alt.Chart(df_results).mark_bar().encode(
        alt.X("Score/100", bin=True),
        y="count()"
    )
    
    st.altair_chart(c, use_container_width=True)


st.subheader("To do")
st.caption(
    """
    (1) Use web crawler to get movie data and show result in dashboard.\n
    """
)
st.subheader("Contact")
st.caption(
    """
    If you have any issue or suggestion, please contact the author.\n
    Yao-Chung Chen \n
    email: yaochung41@gmail.com
    """
)


