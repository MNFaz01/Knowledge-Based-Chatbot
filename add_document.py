#!/usr/bin/env python
# coding: utf-8

# In[7]:


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
from utils import LLMHelper
from langchain.document_loaders import PyPDFLoader
import hashlib

# Load environment variables

helper = LLMHelper()

# Split the file path into its components
dir_path, file_name = os.path.split(helper.pdf_path)

# Extract the keyword from the file name
filename = file_name.split('.')[0]

documents = PyPDFLoader(pdf_path).load()

for(document) in documents:
    try:
        if document.page_content.encode("iso-8859-1") == document.page_content.encode("latin-1"):
            document.page_content = document.page_content.encode("iso-8859-1").decode("utf-8", errors="ignore")
    except:
        pass

    
docs = helper.text_splitter.split_documents(documents)

pattern = re.compile(r'[\x00-\x1f\x7f\u0080-\u00a0\u2000-\u3000\ufff0-\uffff]')
for(doc) in docs:
    doc.page_content = re.sub(pattern, '', doc.page_content)
    if doc.page_content == '':
        docs.remove(doc)
        
        
keys = []
for i, doc in enumerate(docs):
    # Create a unique key for the document
    hash_key = hashlib.sha1(f"{filename}_{i}".encode('utf-8')).hexdigest()
    hash_key = f"doc:{helper.index_name}:{hash_key}"
    keys.append(hash_key)
    doc.metadata = {"source": f"[{filename}]({filename}_SAS_TOKEN_PLACEHOLDER_)", "chunk": i, "key": hash_key, "filename": filename}
helper.vector_store.add_documents(documents=docs, redis_url=helper.vector_store_full_address,  index_name=helper.index_name, keys=keys)


