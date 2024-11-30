#!/usr/bin/env python
# coding: utf-8

# ### This notebook is focused on embedding generation and document storage for a Retrieval-Augmented Generation (RAG) system. It processes previously saved documentation from data science libraries (Pandas, NumPy, Scikit-learn) by loading them, generating embeddings using transformer models, and storing these embeddings in a vector database. The notebook utilizes libraries such as langchain, sentence-transformers, and torch to manage document processing and embedding generation. It also includes steps for managing GPU memory and organizing documents into parent-child relationships for efficient retrieval.

# In[1]:


# Import necessary libraries

import os
import glob
from tqdm import tqdm
from langchain_community.document_loaders import UnstructuredHTMLLoader
import pickle
import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
import time
from langchain_community.vectorstores import FAISS
import os
import shutil
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore


# ## Reading saved documents

# In[2]:


# Load pre-processed Pandas documentation
path = '../data/documents/processed_docs/pandas_docs.pkl'

# Load data from a file
with open(path, 'rb') as file:
    pandas_docs = pickle.load(file)


# In[3]:


# Example of accessing a specific document
#print(pandas_docs[134].page_content)


# In[4]:


pandas_docs[134]


# ## Embedding generation

# In[5]:


#### https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/


# In[6]:


# need to update this
# !pip install --upgrade pyOpenSSL


# In[7]:


# Function to check GPU memory usage

def memory_check():
    """
    Checks and prints the memory usage of the GPU.
    """
    # Get the memory allocated on the GPU
    allocated_memory = torch.cuda.memory_allocated()
    # Get the total memory on the GPU
    total_memory = torch.cuda.get_device_properties(0).total_memory
    # Get the free memory
    free_memory = total_memory - allocated_memory

    print(f"Allocated Memory: {allocated_memory / 1e6:.2f} MB")
    print(f"Free Memory: {free_memory / 1e6:.2f} MB")
    print(f"Total Memory: {total_memory / 1e6:.2f} MB")
    
memory_check()


# ## embedding models

# In[8]:


# model_name = "thenlper/gte-large"
# embeddings = HuggingFaceEmbeddings(model_name=model_name) 


# In[9]:


# Initialize embeddings model

model_name = "Alibaba-NLP/gte-large-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"trust_remote_code": True}
)


# In[10]:


memory_check()


# In[11]:


# #https://huggingface.co/dunzhang/stella_en_400M_v5


# model_name = "dunzhang/stella_en_400M_v5"
# embeddings = HuggingFaceEmbeddings(
#     model_name=model_name,
#     model_kwargs={"trust_remote_code": True}
# )


# **We tried using 3 different embedding models and we got the best document retrival using gte-large-en-v1.5.**

# ### Using ChromaDB

# In[12]:


# Initialize vector database and document retriever

parent_doc_path = "../data/vector_db/parent_docs"
child_doc_path = "../data/vector_db/child_docs"


# In[13]:


# Paths to empty
parent_docs_path = parent_doc_path
child_docs_path = os.path.abspath(child_doc_path)

# Function to empty a folder
def empty_folder(folder):
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f'Failed to delete {item_path}. Reason: {e}')

# Empty the folders
empty_folder(parent_docs_path)
empty_folder(child_docs_path)

print("Folders emptied successfully.")


# In[14]:


# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# The vectorstore to use to index the child chunks


# In[15]:


# Create and configure vectorstore and retriever

fs = LocalFileStore(parent_doc_path)
store = create_kv_docstore(fs)

vectorstore = Chroma(collection_name= "split_parents", 
                     embedding_function= embeddings, 
                     persist_directory= child_doc_path)
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)


# ### adding pandas docs

# In[16]:


# Add documents to the retriever
for doc in tqdm(pandas_docs, desc="Adding documents", unit="doc"):
    retriever.add_documents([doc], ids=None)


# In[17]:


# Load Chroma vectorstore
# Perform similarity search to test the retriever

loaded_vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=embeddings,
    persist_directory= child_doc_path
)
 
# Load parent document store
loaded_file_store = LocalFileStore(parent_doc_path)
 
# Recreate ParentDocumentRetriever
loaded_retriever = ParentDocumentRetriever(
    vectorstore=loaded_vectorstore,
    docstore=loaded_file_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)


# In[18]:


len(list(loaded_file_store.yield_keys()))


# In[24]:


# Example query

loaded_vectorstore.similarity_search("how can I handle irregularly spaced time series data, including filling missing values?")


# In[25]:


loaded_retriever.invoke("how can I handle irregularly spaced time series data, including filling missing values?")


# ### adding scikit-learn docs

# In[28]:


# Define the path
path = '../data/documents/processed_docs/scikit_learn_docs.pkl'

# Load data from a file
with open(path, 'rb') as file:
    scikit_learn_docs = pickle.load(file)


# In[30]:


len(scikit_learn_docs)


# In[41]:


# Adding docs to local storage
for doc in tqdm(scikit_learn_docs, desc="Adding documents", unit="doc"):
    retriever.add_documents([doc], ids=None)


# In[44]:


len(list(loaded_file_store.yield_keys()))


# In[51]:


loaded_vectorstore.similarity_search("hyperparameters for linear regression model")


# In[48]:


loaded_retriever.invoke("I am getting an error while try to normalize the data")


# ### adding numpy docs

# In[53]:


# Define the path
path = '../data/documents/processed_docs/numpy_docs.pkl'

# Load data from a file
with open(path, 'rb') as file:
    numpy_docs = pickle.load(file)


# In[54]:


len(numpy_docs)


# In[55]:


# Adding docs to local storage
for doc in tqdm(numpy_docs, desc="Adding documents", unit="doc"):
    retriever.add_documents([doc], ids=None)


# In[56]:


len(list(loaded_file_store.yield_keys()))


# In[59]:


loaded_vectorstore.similarity_search("how can i create a 2d array?")


# In[60]:


loaded_retriever.invoke("how can i create a 2d array?")


# In[61]:


loaded_retriever


# In[ ]:




