# Multi-Agent Research-RAG Chatbot

![Agent Workflow](agent_workflow.png)

A modular, multi-agent chatbot framework that combines Retrieval-Augmented Generation (RAG) with live internet search to deliver accurate, up-to-date answers. Powered by a supervisor agent orchestrating dedicated sub-agents for RAG, internet retrieval, and result synthesis.

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Architecture](#architecture)
* [Directory Structure](#directory-structure)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [Agents](#agents)

  * [Supervisor Agent](#supervisor-agent)
  * [RAG Agent](#rag-agent)
  * [Internet Agent](#internet-agent)
  * [Synthesis Agent](#synthesis-agent)
* [Utilities](#utilities)
* [Vector Database](#vector-database)
* [Contributing](#contributing)
* [License](#license)

## Overview

This repository implements a **multi-agent conversational AI system** that leverages both a local vector store (RAG) and internet search to answer user queries. A central **Supervisor Agent** decides which sub-agents to invoke and in what order, aggregating their outputs into a final, coherent response.

## Features

* **Retrieval-Augmented Generation (RAG)**: Query a local vector database of documents to ground responses in your own data.
* **Live Internet Search**: Fetch up-to-the-minute information when the local data is insufficient.
* **Multi-Agent Orchestration**: Supervisor coordinates specialized agents for retrieval, search, and synthesis.
* **Streamlit Frontend**: Interactive chat interface with caching for efficient reruns.

## Architecture

The system follows a directed graph workflow:

![Agent Workflow](agent_workflow.png)

1. **Supervisor** makes high-level decisions.
2. **RAG Agent** fetches relevant passages from the vector store.
3. **Internet Agent** (conditional) searches the web for live data.
4. **Synthesis Agent** merges results into a final answer.
5. Loop back if more context or iterations are needed.

## Directory Structure

```
research_rag/
├── streamlit.py
├── LICENSE
├── README.md
├── utils/
│   ├── agents.py
│   ├── rag.py
│   ├── doc_processor.py
│   └── query.py
└── vector_db/
```

````

## Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/your-org/research_rag.git
   cd research_rag
````

2. **Create & activate a virtual environment**:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate   # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your vector database**:

   * Place your documents under `vector_db/documents/`.
   * Initialize or update embeddings via the RAG utility (see [Utilities](#utilities)).

## Configuration

Create a `.env` file in the project root with your API keys:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

## Usage

Launch the chat UI:

```bash
streamlit run streamlit.py
```

* Enter your question in the chat box.
* The Supervisor Agent will route your query through RAG and/or an internet search.
* The Synthesis Agent composes and returns the final answer.

## Agents

### Supervisor Agent

* Orchestrator that decides whether to use RAG, internet search, or both.
* Determines when to loop for follow-up retrievals.

### RAG Agent

* Uses FAISS (via `utils/rag.py`) to retrieve top-
  k relevant passages from the local vector store.
* Returns context snippets and their source metadata.

### Internet Agent

* Performs live web searches when local data is insufficient.
* Cleans and ranks search results before passing to synthesis.

### Synthesis Agent

* Merges RAG and internet outputs.
* Uses an LLM to generate a coherent, context-rich response.

## Utilities

* **`utils/agents.py`**: Implements the `MultiAgentSystem` orchestrating supervisor, RAG, internet, and synthesis agents via LangGraph.
* **`utils/rag.py`**: Manages RAG pipeline: document chunking, embedding, indexing (Chroma or FAISS), and similarity search.
* **`utils/doc_processor.py`**: `DocumentProcessor` for loading and processing various document types (.pdf, .docx, .html, .xml, .md, .txt), enriching metadata.
* **`utils/query.py`**: `QueryEngine` providing query preprocessing, expansion, reranking, similarity search, and LLM-based answer generation.

## Vector Database

The `vector_db/` folder contains:

* **`index.faiss`**: Serialized FAISS index.
* **`documents/`**: Source documents used for embeddings.
* **`metadata.json`**: Maps vector ids to document metadata.

You can regenerate the index by running the RAG utility:

```bash
python -m utils/rag.py
```

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes and push: `git push origin feature-name`.
4. Open a Pull Request.

Please adhere to the existing code style and include tests for new features.

## License

This project is licensed under the [MIT License](LICENSE).
