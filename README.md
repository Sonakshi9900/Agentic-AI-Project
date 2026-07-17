# 🧠 Agentic AI Research Paper Intelligence System

An **agentic AI system** that helps researchers search, summarize, analyze, and compare scientific papers using semantic search, transformer-based NLP, and an LLM-powered agent with custom tools.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sonakshi9900/Agentic-AI-Project/blob/main/Coding_Blocks_Research_Paper_Intelligence_System.ipynb)

---

## 📌 Overview

Reading and keeping up with research papers is time-consuming. This project builds an **agentic assistant** that can autonomously decide which tool to use — semantic search, summarization, keyword extraction, or paper comparison — to answer natural language questions about a large corpus of ML research papers.

Instead of a single static pipeline, the system uses a **LangChain agent backed by an LLM (Groq's Llama 3.1)** that reasons over the user's query, selects the right tool(s), executes them, and returns a grounded, tool-verified response.

---

## ✨ Key Features

- 🔍 **Semantic Paper Search** – Finds the most relevant papers for a query using dense vector embeddings and FAISS similarity search (not just keyword matching).
- 📝 **Automatic Summarization** – Condenses paper abstracts into concise summaries using a BART transformer model.
- 🏷️ **Keyword Extraction** – Surfaces the most important keyphrases from a paper or query using KeyBERT.
- 🧩 **Named Entity Recognition (NER)** – Identifies models, frameworks, and organizations mentioned in text (e.g., "ResNet50", "PyTorch", "Stanford University").
- 🤝 **Paper Comparison** – Compares two papers based on their abstracts using retrieval + LLM reasoning.
- 🤖 **Agentic Orchestration** – An LLM agent dynamically chooses which tool(s) to call based on the user's request, rather than following a fixed script.

---

## 🏗️ Architecture

```
                     ┌─────────────────────┐
                     │   User Query (NL)   │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   LLM Agent (Groq)  │
                     │  Llama-3.1-8B-Instant│
                     └──────────┬──────────┘
                                │ decides which tool to call
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌──────────────┐      ┌──────────────────┐     ┌──────────────────┐
│ Search &      │      │ Extract          │     │ Compare           │
│ Summarize     │      │ Keywords         │     │ Papers             │
│ (FAISS + BART)│      │ (KeyBERT)        │     │ (FAISS + LLM)      │
└──────┬────────┘      └──────────────────┘     └──────────────────┘
       │
       ▼
┌───────────────────────┐
│ Sentence Embeddings    │
│ (all-MiniLM-L6-v2)     │
└───────────┬────────────┘
            ▼
┌───────────────────────┐
│  FAISS Vector Index    │
│  (ML-ArXiv-Papers)     │
└───────────────────────┘
```

---

## 🛠️ Tech Stack

| Category            | Technology                                         |
|----------------------|-----------------------------------------------------|
| LLM / Agent          | LangChain, `create_agent`, Groq (`llama-3.1-8b-instant`) |
| Embeddings           | `sentence-transformers` (`all-MiniLM-L6-v2`)        |
| Vector Search        | FAISS (`IndexFlatIP`, cosine similarity)            |
| Summarization        | Hugging Face Transformers (`facebook/bart-large-cnn`) |
| Keyword Extraction   | KeyBERT                                             |
| Named Entity Recognition | Hugging Face Transformers `pipeline("ner")`     |
| Dataset              | [`CShorten/ML-ArXiv-Papers`](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) (Hugging Face) |
| Data Handling        | Pandas, NumPy                                       |
| Environment          | Google Colab / Jupyter Notebook                     |

---

## 📂 Dataset

The system uses the **ML-ArXiv-Papers** dataset from Hugging Face, which contains titles and abstracts of machine learning research papers from arXiv. A subset of ~15,000 papers is embedded and indexed for fast semantic retrieval.

---

## ⚙️ How It Works

1. **Data Preparation** – Load the dataset, merge title + abstract into a single text field, and clean it.
2. **Embedding Generation** – Encode each paper into a 384-dimensional vector using `all-MiniLM-L6-v2`.
3. **Vector Indexing** – Store normalized embeddings in a FAISS `IndexFlatIP` index for cosine-similarity search.
4. **Tool Creation** – Wrap core capabilities (search+summarize, keyword extraction, comparison) as LangChain `@tool` functions.
5. **Agent Assembly** – Bind the tools to a Groq-hosted LLM and create an agent capable of tool selection and reasoning.
6. **Query Execution** – The agent receives a natural language query, invokes the right tool(s), and returns a response grounded in real tool output.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- A [Groq API key](https://console.groq.com/) (used for the LLM)
- (Optional) Google Colab, for the easiest way to run this notebook

### Installation

```bash
git clone https://github.com/Sonakshi9900/Agentic-AI-Project.git
cd Agentic-AI-Project

pip install datasets faiss-cpu sentence-transformers keybert \
            transformers==4.46.3 huggingface_hub==0.26.2 tokenizers==0.20.3 \
            sentence-transformers==3.3.1 \
            langchain langchain-community langchain-core langchain-huggingface langchain-groq
```

### Set your API key

If running in Google Colab:
```python
from google.colab import userdata
api_key = userdata.get("GROQ_API_KEY")
```

If running locally, set it as an environment variable instead:
```bash
export GROQ_API_KEY="your-api-key-here"
```

### Run the notebook

Open `Coding_Blocks_Research_Paper_Intelligence_System.ipynb` in Jupyter or Google Colab and run the cells sequentially.

---

## 💡 Example Usage

```python
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Find the top 3 research papers on Vision Transformers."}
    ]
})
```

```python
response = agent.invoke({
    "messages": [
        {"role": "user", "content": "Extract the top 5 keywords from Deep Learning for Medical Image Reconstruction."}
    ]
})
```

---

## 📁 Project Structure

```
Agentic-AI-Project/
├── Coding_Blocks_Research_Paper_Intelligence_System.ipynb   # Main notebook (data pipeline + agent)
└── README.md                                                # Project documentation
```

---

## 🔮 Future Improvements

- [ ] Add a Streamlit/Gradio UI for interactive querying
- [ ] Support PDF upload and full-text (not just abstract) indexing
- [ ] Add citation graph analysis between papers
- [ ] Deploy the FAISS index as a persistent vector database (e.g., Pinecone, Chroma, Weaviate)
- [ ] Add automated evaluation of summarization/search quality
- [ ] Multi-agent workflow (e.g., separate research, summarization, and critique agents)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.

## 👩‍💻 Author

**Sonakshi9900** – [GitHub Profile](https://github.com/Sonakshi9900)
