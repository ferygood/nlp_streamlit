FROM python:3.9
WORKDIR /bert_streamlit
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 8501
COPY . /bert_streamlit
ENTRYPOINT [ "streamlit", "run" ]
CMD ["bert_streamlit.py"]
