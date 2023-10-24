import streamlit as st
from streamlit_chat import message
import requests
import openai 
import pandas as pd
from configs import *
import json
import numpy as np
from PIL import Image, ImageDraw
import chromadb
from build_db import final_engine, embeddings
from llama_index.indices.query.schema import QueryBundle
from llama_index.llms import ChatMessage, MessageRole
from llama_index.prompts import ChatPromptTemplate

                                               
col1,_ = st.columns([1,2])
with col1:
    logo_image = Image.open(f"{PATH_LOGO}/logo.jpg")
    st.image(logo_image)

st.title("QNU Chatbot")

with st.chat_message("assistant"):
    st.write("Chào bạn! Tôi là QNU - Chatbot cung cấp các thông tin về trường Đại học Quy Nhơn. Tương tác với tôi để đặt câu hỏi hoặc mô tả công việc bạn cần tôi giúp đỡ. Tôi sẵn sàng hỗ trợ bạn một cách tốt nhất có thể.")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

input_user = st.chat_input("Enter your question")
if input_user:
    with st.chat_message("user"):
        st.markdown(input_user)
    print('input_user',input_user)

    st.session_state.messages.append({"role": "user","content": input_user})
        

    with st.chat_message("assistant"):
        response = final_engine.query(input_user)
        st.markdown(response.response)
    st.session_state.messages.append({"role": "assistant","content": response.response})
    print(response.response)
   