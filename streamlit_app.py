import os
import torch

from uuid import uuid4

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

from app.llm import get_ai_response, get_ai_response_stream

import streamlit as st

SESSION_ID = str(uuid4())

st.title("OpenSpecimen AI")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What brings you here today?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        resp_container = st.empty()
        response_text = ""
        response = get_ai_response_stream(prompt, SESSION_ID)
        with st.spinner("🤔 Thinking..."):
            for i in response:
                response_text += i.content
                resp_container.markdown(response_text)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_text})
