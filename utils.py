#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import openai
from dotenv import load_dotenv
import logging
import re
import hashlib

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.vectorstores.base import VectorStore
from langchain.chains import ChatVectorDBChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.llm import LLMChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.prompts import PromptTemplate
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter, TextSplitter
from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders import TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from my_redis import RedisExtended
import streamlit as st
from streamlit_chat import message

import pandas as pd
import urllib
from fake_useragent import UserAgent

# For Adding Document
import streamlit as st
import os, json, re, io
from os import path
import requests
import mimetypes
import traceback
import chardet
import uuid
from redis.exceptions import ResponseError 
from urllib import parse
from customprompt import PROMPT


class LLMHelper:
    def __init__(self,
        document_loaders : BaseLoader = None, 
        text_splitter: TextSplitter = None,
        embeddings: OpenAIEmbeddings = None,
        llm: AzureOpenAI = None,
        temperature: float = None,
        max_tokens: int = None,
        custom_prompt: str = "",
        vector_store: VectorStore = None,
        k: int = None):

        env_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path=env_path) 

        openai.api_key = os.getenv("OPENAI_API_KEY")
        #openai.api_key = os.environ["OPENAI_API_KEY"] = "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"
        
        self.api_base = openai.api_base
        self.pdf_path = os.getenv("PDF_PATH_SINGLE")
        self.api_version = openai.api_version
        self.index_name: str = "embeddings"
        self.model: str = os.getenv('OPENAI_EMBEDDINGS_ENGINE_DOC', "text-embedding-ada-002")
        self.deployment_name: str = os.getenv("OPENAI_ENGINE", os.getenv("OPENAI_ENGINES", "text-davinci-003"))
        self.deployment_type: str = os.getenv("OPENAI_DEPLOYMENT_TYPE", "Text")
        self.temperature: float = float(os.getenv("OPENAI_TEMPERATURE", 0.7)) if temperature is None else temperature
        self.max_tokens: int = int(os.getenv("OPENAI_MAX_TOKENS", -1)) if max_tokens is None else max_tokens
        self.prompt = PROMPT if custom_prompt == '' else PromptTemplate(template=custom_prompt, input_variables=["summaries", "question"])


        # Vector store settings
        self.vector_store_address: str = os.getenv('REDIS_ADDRESS', "localhost")
        self.vector_store_port: int= int(os.getenv('REDIS_PORT', 6379))
        self.vector_store_protocol: str = os.getenv("REDIS_PROTOCOL", "redis://")
        self.vector_store_password: str = os.getenv("REDIS_PASSWORD", None)

        if self.vector_store_password:
            self.vector_store_full_address = f"{self.vector_store_protocol}:{self.vector_store_password}@{self.vector_store_address}:{self.vector_store_port}"
        else:
            self.vector_store_full_address = f"{self.vector_store_protocol}{self.vector_store_address}:{self.vector_store_port}"
        
        self.chunk_size = int(os.getenv('CHUNK_SIZE', 500))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', 100))
        self.document_loaders: BaseLoader = WebBaseLoader if document_loaders is None else document_loaders
        self.text_splitter: TextSplitter = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap) if text_splitter is None else text_splitter
        self.embeddings: OpenAIEmbeddings = OpenAIEmbeddings(model=self.model, chunk_size=1) if embeddings is None else embeddings
        if self.deployment_type == "Chat":
            self.llm: ChatOpenAI = ChatOpenAI(model_name=self.deployment_name, engine=self.deployment_name, temperature=self.temperature, max_tokens=self.max_tokens if self.max_tokens != -1 else None) if llm is None else llm
        else:
            self.llm: AzureOpenAI = AzureOpenAI(deployment_name=self.deployment_name, temperature=self.temperature, max_tokens=self.max_tokens) if llm is None else llm
        self.vector_store: RedisExtended = RedisExtended(redis_url=self.vector_store_full_address, index_name=self.index_name, embedding_function=self.embeddings.embed_query) if vector_store is None else vector_store   
        self.k : int = 3 if k is None else k

#         self.pdf_parser : AzureFormRecognizerClient = AzureFormRecognizerClient() if pdf_parser is None else pdf_parser
#         self.blob_client: AzureBlobStorageClient = AzureBlobStorageClient() if blob_client is None else blob_client
#         self.enable_translation : bool = False if enable_translation is None else enable_translation
#         self.translator : AzureTranslatorClient = AzureTranslatorClient() if translator is None else translator

        self.user_agent: UserAgent() = UserAgent()
        self.user_agent.random
        

        
    def get_semantic_answer_lang_chain(self, question, chat_history):
        question_generator = LLMChain(llm=self.llm, prompt=CONDENSE_QUESTION_PROMPT, verbose=False)
        doc_chain = load_qa_with_sources_chain(self.llm, chain_type="stuff", verbose=True, prompt=self.prompt)
        chain = ConversationalRetrievalChain(
            retriever=self.vector_store.as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            return_source_documents=True,
            # top_k_docs_for_context= self.k
        )
        result = chain({"question": question, "chat_history": chat_history})
        context = "\n".join(list(map(lambda x: x.page_content, result['source_documents'])))
        sources = "\n".join(set(map(lambda x: x.metadata["source"], result['source_documents'])))

        #container_sas = self.blob_client.get_container_sas()
        
        result['answer'] = result['answer'].split('SOURCES:')[0].split('Sources:')[0].split('SOURCE:')[0].split('Source:')[0]
        #sources = sources.replace('_SAS_TOKEN_PLACEHOLDER_', container_sas)

        return question, result['answer'], context, sources

    def get_embeddings_model(self):
        OPENAI_EMBEDDINGS_ENGINE_DOC = os.getenv('OPENAI_EMEBDDINGS_ENGINE', os.getenv('OPENAI_EMBEDDINGS_ENGINE_DOC', 'text-embedding-ada-002'))  
        OPENAI_EMBEDDINGS_ENGINE_QUERY = os.getenv('OPENAI_EMEBDDINGS_ENGINE', os.getenv('OPENAI_EMBEDDINGS_ENGINE_QUERY', 'text-embedding-ada-002'))
        return {
            "doc": OPENAI_EMBEDDINGS_ENGINE_DOC,
            "query": OPENAI_EMBEDDINGS_ENGINE_QUERY
        }

    def get_completion(self, prompt, **kwargs):
        if self.deployment_type == 'Chat':
            return self.llm([HumanMessage(content=prompt)]).content
        else:
            return self.llm(prompt)

        



# In[ ]:





# In[ ]:




# In[ ]:





# In[ ]:





# In[ ]:




