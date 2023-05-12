#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from streamlit_chat import message
from utils import LLMHelper

def clear_text_input():
    st.session_state['question'] = st.session_state['input']
    st.session_state['input'] = ""

def clear_chat_data():
    st.session_state['input'] = ""
    st.session_state['chat_history'] = []
    st.session_state['source_documents'] = []

# Initialize chat history
st.session_state.setdefault('question', None)
st.session_state.setdefault('chat_history', [])
st.session_state.setdefault('source_documents', [])

llm_helper = LLMHelper()

# Chat 
st.text_input("You: ", placeholder="type your question", key="input", on_change=clear_text_input)
clear_chat = st.button("Clear chat", key="clear_chat", on_click=clear_chat_data)

for key in st.session_state.keys():
    print(key)
    
# if 'question' not in st.session_state:
#     st.session_state['question'] = None
#     print(st.session_state['question'])

if st.session_state['question']:
    question, result, _, sources = llm_helper.get_semantic_answer_lang_chain(st.session_state['question'], st.session_state['chat_history'])
    st.session_state['chat_history'].append((question, result))
    st.session_state['source_documents'].append(sources)

if st.session_state['chat_history']:
    for i in range(len(st.session_state['chat_history'])-1, -1, -1):
        message(st.session_state['chat_history'][i][1], key=str(i))
        st.markdown(f'\n\nSources: {st.session_state["source_documents"][i]}')
        message(st.session_state['chat_history'][i][0], is_user=True, key=str(i) + '_user')

