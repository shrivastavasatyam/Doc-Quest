# DocQuest 🔍: In-House Retrieval-Augmented Generation (RAG) System

__Specific Research Question:__
How can we design and optimize a RAG system to provide personalized, reference-rich outputs for data science teams, while managing in-house packages, in an offline environment?

To break it down further:
- __Customized Output with References:__ Investigate how the RAG system can leverage context-specific information to generate tailored outputs for data scientists. Explore methods to include accurate and relevant references in the generated content.
- __In-House Offline System:__ Examine the technical requirements and constraints of deploying a RAG system in an offline environment. Assess the challenges related to data storage, processing, and model updates without internet connectivity.
- __Managing In-House Packages:__ Analyze strategies for integrating and managing proprietary data science packages within the RAG system. Look into the compatibility issues, version control, and dependency management to ensure seamless operation and integration.

## Overview:

DocQuest is an offline Retrieval-Augmented Generation (RAG) system designed for data science teams. It provides personalized, reference-rich outputs while operating in environments without internet connectivity. This system is ideal for secure settings where data privacy is paramount.

## Features:

- **Offline Operation**: Fully functional RAG system that works without internet connectivity
- **Advanced Document Preprocessing**: Robust pipeline for efficient document handling and content extraction
- **State-of-the-Art Embedding & Vector Storage**: Incorporates multiple embedding models and vector storage solutions
- **Optimized RAG Pipeline**: Includes intelligent query processing and high-precision document retrieval
- **User-Friendly Interface**: Streamlit-based UI for easy interaction

## Folder Structure:

```
DOC-QUEST/
│
├── data/                           # Data storage
│   ├── documents/                  # Raw or processed document storage
│   └── vector_db/                  # Vector databases
│       ├── child_docs/             # Embeddings for child documents
│       └── parent_docs/            # Embeddings for parent documents
│
├── notebooks/                      # Jupyter notebooks for prototyping and experimentation
│   ├── 1_documentation_download.ipynb
│   ├── 2_document_pre_processing.ipynb
│   ├── 3_embedding_vector_save_gpu.ipynb
│   ├── 4_conversation_rag.ipynb
│   ├── data_wrangling.ipynb
│   └── rag_v1.ipynb
│
├── src/                            # Core source code for pipeline components
│   ├── 1_documentation_download.py
│   ├── 2_document_pre_processing.py
│   ├── 3_embedding_vector_save_gpu.py
│   ├── 4_conversation_rag.py
│
├── .gitignore                      # Specifies files/folders to ignore in version control
├── doc_quest_app.py                # Main application entry point
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## Installation:

1. Clone the repository:
   ```
   git clone https://github.com/shrivastavasatyam/Doc-Quest.git
   cd golden-retriever
   ```

2. Set up a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Configure API key:
   Set up your GROQ API key as an environment variable:
   ```
   export GROQ_API_KEY=your_groq_api_key
   ```
   Or add it directly in the `doc_quest_app.py` file:
   ```python
   os.environ["GROQ_API_KEY"] = "your_groq_api_key"
   ```

5. Prepare document paths:
   Ensure your document paths are correctly set in the `doc_quest_app.py` file:
   ```python
   parent_doc_path = "/path/to/your/parent_docs"
   child_doc_path = "./path/to/your/child_docs"
   ```

## Usage:

1. Launch the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

2. Access the web interface at the URL provided by Streamlit (usually `http://localhost:8501`).

3. Use the chat interface to ask questions and interact with the RAG system.

## System Components:

- **Document Preprocessing**: Uses BSHTMLLoader and BeautifulSoup for content extraction
- **Embedding Models**: gte-large-en-v1.5, gte-large, stella_en_400M_v5
- **Vector Storage**: Chroma, FAISS
- **Text Generation**: Utilizes the llama-3.1-70b model

## Data Sources:

The system integrates documentation from multiple sources, including:
- Pandas
- NumPy
- Scikit-learn

## Evaluation:

***WORK IN PROGRESS***

## Future Improvements:

- Implementing chat history storage.
- Further optimizing the retrieval component.
- Adding anchored links for direct navigation to relevant document sections.
- Introducing multiple generative model selection options in the UI.