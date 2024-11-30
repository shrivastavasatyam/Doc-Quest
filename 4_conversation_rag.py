#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install -qU langchain-groq


# In[2]:


import getpass
import os

# os.environ["GROQ_API_KEY"] = getpass.getpass()
os.environ["GROQ_API_KEY"]= "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"


# In[3]:


from langchain_groq import ChatGroq

llm = ChatGroq(model="llama-3.1-8b-instant")


# In[4]:


import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil
from langchain.storage._lc_store import create_kv_docstore
from langchain.storage import LocalFileStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
import pickle
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from langchain_core.documents import Document
import json


# ## loading stored embedding vectors

# In[5]:


model_name = "Alibaba-NLP/gte-large-en-v1.5"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"trust_remote_code": True}
)


# In[6]:


import psutil

def check_ram():
    # Get memory information
    memory = psutil.virtual_memory()
    
    # Convert to GB for easier reading
    total = memory.total / (1024.0 ** 3)
    available = memory.available / (1024.0 ** 3)
    used = memory.used / (1024.0 ** 3)
    
    print(f"Total RAM: {total:.2f} GB")
    print(f"Available RAM: {available:.2f} GB")
    print(f"Used RAM: {used:.2f} GB")
    print(f"RAM Usage: {memory.percent}%")

check_ram()


# In[7]:


import torch

def check_mps_memory():
    if torch.backends.mps.is_available():
        print("MPS device is available.")
        
        # Get the total memory allocated by the Metal driver
        total_memory = torch.mps.current_allocated_memory()
        
        # Convert bytes to megabytes for easier reading
        total_memory_mb = total_memory / (1024 * 1024)
        
        print(f"Total GPU memory allocated: {total_memory_mb:.2f} MB")
    else:
        print("MPS device is not available.")

check_mps_memory()


# In[8]:


parent_doc_path = "/Users/mehuljain/Documents/course_related/Capstone/rag_ds5500/vector_db/parent_docs"
child_doc_path = "../vector_db/child_docs"


# In[9]:


# This text splitter is used to create the parent documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
# This text splitter is used to create the child documents
# It should create documents smaller than the parent
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# The vectorstore to use to index the child chunks


# In[10]:


# Load Chroma vectorstore
loaded_vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=embeddings,
    persist_directory= child_doc_path
)
 
# Load parent document store
loaded_file_store = LocalFileStore(parent_doc_path)


# In[11]:


class CustomParentDocumentRetriever(ParentDocumentRetriever):
    def invoke(self, query, config=None, **kwargs):
        results = super().invoke(query, config=config, **kwargs)
        documents = []
        for result in results:
            try:
                if isinstance(result, bytes):
                    json_string = result.decode('utf-8')
                elif isinstance(result, str):
                    json_string = result
                else:
                    print(f"Unexpected result type: {type(result)}")
                    continue

                deserialized_result = json.loads(json_string)
                
                # Extract metadata and page_content from the deserialized result
                metadata = deserialized_result['kwargs']['metadata']
                page_content = deserialized_result['kwargs']['page_content']

                doc = Document(metadata=metadata, page_content=page_content)
                documents.append(doc)
            except Exception as e:
                print(f"Error processing result: {e}")
                print(f"Problematic result: {result}")

        return documents


# In[12]:


# Use the custom retriever
loaded_retriever = CustomParentDocumentRetriever(
    vectorstore=loaded_vectorstore,
    docstore=loaded_file_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)


# In[13]:


len(list(loaded_file_store.yield_keys()))


# In[14]:


check_mps_memory()


# In[15]:


loaded_vectorstore.similarity_search("how can i create a 2d array?")


# In[16]:


rt=loaded_retriever.invoke("how can i create a 2d array?")
rt


# In[17]:


rt[0].metadata['source']


# In[18]:


for i in rt:
    print(i.metadata['source'])


# In[19]:


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(loaded_retriever, question_answer_chain)


# In[20]:


response = rag_chain.invoke({"input": "how can i create a 2d array? Give detailed instructions"})


# In[21]:


print(f"Answer: {response['answer']}")


# ## Adding chat history

# In[22]:


from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# In[23]:


### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, loaded_retriever, contextualize_q_prompt
)


# In[24]:


### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# In[25]:


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


# In[26]:


conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)["answer"]


# In[27]:


store


# In[28]:


conversational_rag_chain.invoke(
    {"input": "how can i create a 2d array?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]


# In[29]:


store


# In[30]:


conversational_rag_chain.invoke(
    {"input": "can you elaborate more on a = np.arange(4).reshape(2,2)?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]


# In[31]:


store


# In[32]:


conversational_rag_chain.invoke(
    {"input": "can you give some more examples and use cases?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]


# In[45]:


store['abc123'].messages[-1].content


# In[33]:


print(conversational_rag_chain.invoke(
    {"input": "can you give some more examples and use cases?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"])


# In[ ]:


store


# In[ ]:


print(conversational_rag_chain.invoke(
    {"input": "who are you?"},
    config={"configurable": {"session_id": "xyz456"}},
)["answer"])


# In[ ]:


store

