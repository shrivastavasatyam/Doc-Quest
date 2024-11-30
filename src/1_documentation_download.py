#!/usr/bin/env python
# coding: utf-8

# ### The notebook includes steps to download, extract, and prepare the documentation for use in the RAG system.

# In[1]:


# Import necessary libraries

import requests
import zipfile
import os


# ## pandas

# In[1]:


"""
Downloads a zip file from the specified URL and extracts its contents to the given directory.

Parameters:
- url (str): The URL of the zip file to download.
- zip_file_path (str): The path where the zip file will be saved.
- extract_dir (str): The directory where the contents will be extracted.
"""

# URL of the pandas documentation zip file
url = 'https://pandas.pydata.org/pandas-docs/stable/pandas.zip'

# Path to save the downloaded file
zip_file_path = '../data/documents/pandas_docs.zip'

# Download the file
response = requests.get(url)
with open(zip_file_path, 'wb') as file:
    file.write(response.content)

# Create a directory to extract the contents
extract_dir = '../data/documents/pandas_docs'
os.makedirs(extract_dir, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f'Documentation downloaded and extracted to {extract_dir}')


# ## scikit-learn

# In[2]:


# URL of the scikit-learn documentation zip file
url = 'https://scikit-learn.org/stable//_downloads/scikit-learn-docs.zip'

# Path to save the downloaded file
zip_file_path = '../data/documents/scikit_learn_docs.zip'

# Download the file
response = requests.get(url)
with open(zip_file_path, 'wb') as file:
    file.write(response.content)

# Create a directory to extract the contents
extract_dir = '../data/documents/scikit_learn_docs'
os.makedirs(extract_dir, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f'Documentation downloaded and extracted to {extract_dir}')


# ## numpy

# In[2]:


# URL of the numpy documentation zip file
url = 'https://numpy.org/doc/2.0/numpy-html.zip'

# Path to save the downloaded file
zip_file_path = '../data/documents/numpy_docs.zip'

# Download the file
response = requests.get(url)
with open(zip_file_path, 'wb') as file:
    file.write(response.content)

# Create a directory to extract the contents
extract_dir = '../data/documents/numpy_docs'
os.makedirs(extract_dir, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f'Documentation downloaded and extracted to {extract_dir}')


# In[ ]:




