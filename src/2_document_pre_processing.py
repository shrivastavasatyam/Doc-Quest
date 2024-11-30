#!/usr/bin/env python
# coding: utf-8

# ### This notebook is designed to pre-process HTML documentation from data science libraries such as Pandas, NumPy, and Scikit-learn. The goal is to prepare these documents for use in a Retrieval-Augmented Generation (RAG) system by extracting relevant content and metadata. The notebook includes functions for loading HTML files, processing them to extract text content, and updating metadata. It also handles parallel processing to efficiently manage large volumes of documentation.

# In[1]:


# Import necessary libraries

import os
import glob
from tqdm import tqdm
from langchain_community.document_loaders import UnstructuredHTMLLoader, BSHTMLLoader
import multiprocessing
from tqdm import tqdm
import pickle
import os
from bs4 import BeautifulSoup


# In[2]:


def load_html_files(base_path, include_folders):
    """
    Loads HTML files from specified directories, excluding files and directories that are not needed.

    Parameters:
    - base_path (str): Base directory path containing the documentation.
    - include_folders (list): List of folders to include in the search for HTML files.

    Returns:
    - list: List of paths to HTML files.
    """
    all_files = []
    for folder in include_folders:
        folder_path = os.path.join(base_path, folder)
        for root, dirs, files in os.walk(folder_path):
            # Exclude directories starting with an underscore
            dirs[:] = [d for d in dirs if not d.startswith('_')]
            for file in files:
                if (file.endswith('.html')) and file != 'index.html':
                    all_files.append(os.path.join(root, file))
    return all_files


# ## Pandas

# In[3]:


# extracting relevant file paths


# In[4]:


# Example usage for Pandas documentation

pandas_file_path = "../data/documents/pandas_docs/"
pandas_include_folders = ['development', 'getting_started', 'reference/api', 'user_guide', 'whatsnew']
pandas_html_files = load_html_files(pandas_file_path, pandas_include_folders)


# In[5]:


len(pandas_html_files)


# In[104]:


def process_html_file(html_file):
    """
    Processes an HTML file to extract and format its main content.

    Parameters:
    - html_file (str): Path to the HTML file.

    Returns:
    - list: List of processed document data.
    """
    loader = BSHTMLLoader(html_file, open_encoding="utf-8")
    data = loader.load()
    
    for doc in data:
        if 'soup' in doc.metadata:
            soup = doc.metadata['soup']
        else:
            with open(html_file, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
        
        target_main = soup.find('div', class_='bd-article-container')
        if target_main:
            # Remove 'admonition seealso' div, 'prev-next-footer' footer, and 'header-article-items header-article__inner' div
            for element in target_main.find_all(['div', 'footer'], class_=['admonition seealso', 'prev-next-footer', 'header-article-items header-article__inner']):
                element.decompose()

            # Extract and format the text content
            formatted_text = []
            seen_content = set()  # To keep track of unique content
            for element in target_main.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'pre', 'code', 'dt']):
                text = element.get_text(strip=True)
                if text and text not in seen_content:
                    if element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'dt']:
                        formatted_text.append(f'\n{text}\n')
                    elif element.name == 'pre':
                        formatted_text.append(f'\n{element.get_text()}\n')
                    elif element.name == 'code':
                        formatted_text.append(text)
                    seen_content.add(text)
            
            doc.page_content = ''.join(formatted_text).strip()
        else:
            doc.page_content = "No content found in the specified main tag."
    
    return data

def parallel_load(html_files):
    """
    Processes HTML files in parallel to improve efficiency.

    Parameters:
    - html_files (list): List of HTML file paths.

    Returns:
    - list: Flattened list of processed document data.
    """
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(process_html_file, html_files), total=len(html_files), desc="Loading files"))
        return [item for sublist in results for item in sublist]  # Flatten the results


# In[105]:


pandas_data = parallel_load(pandas_html_files)


# In[106]:


len(pandas_data)


# In[107]:


# modifying URL source


# In[108]:


# Define the old and new base URLs
pandas_old_base_url = "../data/documents/pandas_docs"
pandas_new_base_url = "https://pandas.pydata.org/docs"


# In[109]:


# Iterate over each document and update the source in the metadata
for doc in pandas_data:
    if 'source' in doc.metadata:
        doc.metadata['source'] = doc.metadata['source'].replace(pandas_old_base_url, pandas_new_base_url)

# Now, pandas_docs contains updated sources


# In[135]:


print(pandas_data[265])


# In[136]:


print(pandas_data[1659])


# In[129]:


# removing documents that has no content as they were redirected to some other page

docs_with_no_content = []
for i in range(len(pandas_data)):
    if pandas_data[i].page_content == 'No content found in the specified main tag.':
        docs_with_no_content.append(i)
        
print(docs_with_no_content)
len(docs_with_no_content)


# In[130]:


print(pandas_data[1361])


# In[131]:


# Remove the documents with no content
# If pandas_data is a list:
pandas_data = [doc for i, doc in enumerate(pandas_data) if i not in docs_with_no_content]


# In[132]:


len(pandas_data)


# In[137]:


# Save the processed data to a file
# Define the path
path = '../data/documents/processed_docs/pandas_docs.pkl'

# Ensure the directory exists
os.makedirs(os.path.dirname(path), exist_ok=True)

# Save data to a file
with open(path, 'wb') as file:
    pickle.dump(pandas_data, file)


# ## scikit-learn

# In[3]:


import re

def process_html_file(html_file):
    loader = BSHTMLLoader(html_file, open_encoding="utf-8")
    data = loader.load()
    
    for doc in data:
        if 'soup' in doc.metadata:
            soup = doc.metadata['soup']
        else:
            with open(html_file, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
        
        target_main = soup.find('div', class_='bd-article-container')
        if target_main:
            # Remove 'admonition seealso' div, 'prev-next-footer' footer, and 'header-article-items header-article__inner' div
            for element in target_main.find_all(['div', 'footer', 'a', 'p'], 
                                                class_=['admonition seealso','footer-article-item',
                                                        'prev-next-footer',
                                                        'header-article-items header-article__inner',
                                                       'sphx-glr-timing']):
                element.decompose()

            # Extract and format the text content
            formatted_text = []
            seen_content = set()  # To keep track of unique content
            for element in target_main.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'pre', 'code', 'dt']):
                text = element.get_text(strip=True)
                if text and text not in seen_content:
                    if element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'dt']:
                        formatted_text.append(f'\n{text}\n')
                    elif element.name == 'pre':
                        formatted_text.append(f'\n{element.get_text()}\n')
                    elif element.name == 'code':
                        formatted_text.append(text)
                    seen_content.add(text)
            
            doc.page_content = ''.join(formatted_text).strip()
        else:
            doc.page_content = "No content found in the specified main tag."
    
    return data

def parallel_load(html_files):
  with multiprocessing.Pool() as pool:
    results = list(tqdm(pool.imap(process_html_file, html_files), total=len(html_files), desc="Loading files"))
    return [item for sublist in results for item in sublist]  # Flatten the results


# In[4]:


scikit_learn_file_path = "../data/documents/scikit_learn_docs/"
scikit_learn_include_folders = ['auto_examples', 'computing', 
                               'datasets', 'modules', 'developers', 'whats_new', 'notebooks']
scikit_learn_html_files = load_html_files(scikit_learn_file_path, scikit_learn_include_folders)


# In[5]:


len(scikit_learn_html_files)


# In[7]:


scikit_learn_data = parallel_load(scikit_learn_html_files)


# In[8]:


len(scikit_learn_data)


# In[10]:


# Define the old and new base URLs
scikit_learn_old_base_url = "../data/documents/scikit_learn_docs"
scikit_learn_new_base_url = "https://scikit-learn.org/stable"


# In[11]:


# Iterate over each document and update the source in the metadata
for doc in scikit_learn_data:
    if 'source' in doc.metadata:
        doc.metadata['source'] = doc.metadata['source'].replace(scikit_learn_old_base_url, scikit_learn_new_base_url)

# Now, pandas_docs contains updated sources


# In[319]:


print(scikit_learn_data[350])


# In[320]:


print(scikit_learn_data[900])


# In[321]:


print(scikit_learn_data[900])


# In[19]:


docs_with_no_content = []
for i in range(len(scikit_learn_data)):
    if scikit_learn_data[i].page_content == 'No content found in the specified main tag.'  or \
       'This document has been moved' in scikit_learn_data[i].page_content or \
        len(scikit_learn_data[i].page_content) < 200:
        docs_with_no_content.append(i)
        
print(docs_with_no_content)
len(docs_with_no_content)


# In[25]:


docs_with_no_content = []
for i in range(len(scikit_learn_data)):
    if 'Normalizes' in scikit_learn_data[i].page_content:
        docs_with_no_content.append(i)
        
print(docs_with_no_content)
len(docs_with_no_content)


# In[26]:


scikit_learn_data[654]


# In[324]:


# Remove the documents with no content
# If pandas_data is a list:
scikit_learn_data = [doc for i, doc in enumerate(scikit_learn_data) if i not in docs_with_no_content]


# In[325]:


len(scikit_learn_data)


# In[326]:


# Define the path
path = '../data/documents/processed_docs/scikit_learn_docs.pkl'

# Ensure the directory exists
os.makedirs(os.path.dirname(path), exist_ok=True)

# Save data to a file
with open(path, 'wb') as file:
    pickle.dump(scikit_learn_data, file)


# ## numpy

# In[3]:


import re

def process_html_file(html_file):
    loader = BSHTMLLoader(html_file, open_encoding="utf-8")
    data = loader.load()
    
    for doc in data:
        if 'soup' in doc.metadata:
            soup = doc.metadata['soup']
        else:
            with open(html_file, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
        
        target_main = soup.find('div', class_='bd-article-container')
        if target_main:
            # Remove 'admonition seealso' div, 'prev-next-footer' footer, and 'header-article-items header-article__inner' div
            for element in target_main.find_all(['div', 'footer', 'a', 'p'], 
                                                class_=['admonition seealso','footer-article-item',
                                                        'prev-next-footer',
                                                        'header-article-items header-article__inner',
                                                       'sphx-glr-timing']):
                element.decompose()

            # Extract and format the text content
            formatted_text = []
            seen_content = set()  # To keep track of unique content
            for element in target_main.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'pre', 'code', 'dt']):
                text = element.get_text(strip=True)
                if text and text not in seen_content:
                    if element.name in ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'dt']:
                        formatted_text.append(f'\n{text}\n')
                    elif element.name == 'pre':
                        formatted_text.append(f'\n{element.get_text()}\n')
                    elif element.name == 'code':
                        formatted_text.append(text)
                    seen_content.add(text)
            
            doc.page_content = ''.join(formatted_text).strip()
        else:
            doc.page_content = "No content found in the specified main tag."
    
    return data

def parallel_load(html_files):
  with multiprocessing.Pool() as pool:
    results = list(tqdm(pool.imap(process_html_file, html_files), total=len(html_files), desc="Loading files"))
    return [item for sublist in results for item in sublist]  # Flatten the results


# In[8]:


numpy_file_path = "../data/documents/numpy_docs/"
numpy_include_folders = ['user', 'reference', 
                               'building', 'dev', 'f2py', 'release']b
numpy_html_files = load_html_files(numpy_file_path, numpy_include_folders)
len(numpy_html_files)


# In[11]:


numpy_data = parallel_load(numpy_html_files)


# In[12]:


len(numpy_html_files)


# In[13]:


# Define the old and new base URLs
numpy_old_base_url = "../data/documents/numpy_docs"
numpy_new_base_url = "https://numpy.org/doc/stable"


# In[14]:


# Iterate over each document and update the source in the metadata
for doc in numpy_data:
    if 'source' in doc.metadata:
        doc.metadata['source'] = doc.metadata['source'].replace(numpy_old_base_url, numpy_new_base_url)


# In[26]:


print(numpy_data[50])


# In[27]:


print(numpy_data[51])


# In[29]:


docs_with_no_content = []
for i in range(len(numpy_data)):
    if numpy_data[i].page_content == 'No content found in the specified main tag.':
        docs_with_no_content.append(i)
        
print(docs_with_no_content)
len(docs_with_no_content)


# In[48]:


# removing documents that has no content as they were redirected to some other page 
# removing documents with content less than 100 characters

docs_with_no_content = []
for i in range(len(numpy_data)):
    if numpy_data[i].page_content == 'No content found in the specified main tag.' or \
       'This document has been moved' in numpy_data[i].page_content or \
        len(numpy_data[i].page_content) < 100:
        docs_with_no_content.append(i)

print(docs_with_no_content)
print(len(docs_with_no_content))


# In[50]:


# Remove the documents with no content
# If pandas_data is a list:
numpy_data = [doc for i, doc in enumerate(numpy_data) if i not in docs_with_no_content]


# In[51]:


len(numpy_data)


# In[52]:


# Define the path
path = '../data/documents/processed_docs/numpy_docs.pkl'

# Ensure the directory exists
os.makedirs(os.path.dirname(path), exist_ok=True)

# Save data to a file
with open(path, 'wb') as file:
    pickle.dump(numpy_data, file)


# In[53]:


print(numpy_data[542])


# In[ ]:




