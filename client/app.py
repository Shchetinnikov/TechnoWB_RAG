import json
import requests
import streamlit as st
import os, sys

sys.path.append(os.path.dirname("config.py"))
from config import SERVER_HOST, SERVER_PORT

server_url = "http://" + SERVER_HOST + ":" + SERVER_PORT

# Создать хранилище сообщений
if "messages" not in st.session_state:
    st.session_state.messages = []

# Отображение истории сообщений
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

prompt = st.chat_input("Задать вопрос")

if prompt:
    # Сообщение пользователя
    user_data = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_data)
    with st.chat_message("user"):
        st.write(user_data['content'])

    # Сообщение бота
    response = requests.post(server_url + "/user_message", data=json.dumps(user_data))
    bot_data = response.json()['data']
    with st.chat_message("assistant"):
        st.session_state.messages.append(bot_data)
        st.write(bot_data['content'])
