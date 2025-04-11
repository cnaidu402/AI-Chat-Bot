# 🤖 Reddit FAQ Chatbot with LlamaIndex, ChromaDB, and Gemini AI

This project builds an intelligent, domain-specific chatbot that answers Reddit FAQ questions using AI-powered natural language understanding. It combines **LlamaIndex** for indexing, **ChromaDB** for vector storage, and **Gemini AI** for answering questions with safety filtering and response evaluation.

---

## 🚀 Features

- ✅ **Fetches and indexes Reddit FAQ content** from the official source
- 🧠 **Uses Gemini AI** for high-quality question answering
- 🗃️ **Embeds and stores knowledge** in ChromaDB for fast retrieval
- 🧪 **Evaluates chatbot responses** with relevance, completeness, and clarity scores
- 🔒 **Implements content safety filtering** for sensitive prompts
- ⚙️ **Test suite** with safe and unsafe prompts

---

## 📦 Installation

Install the required dependencies:

```bash
pip install llama-index llama-index-vector-stores-chroma llama-index-embeddings-gemini
pip install llama-index-llms-gemini llama-index-readers-web
pip install chromadb
pip install --upgrade pydantic
```

---
                # You are here :)
```

---

## 🛠️ Usage

### 1. Fetch and Index Reddit FAQ

```python
from llama_index.readers.web import SimpleWebPageReader
from bs4 import BeautifulSoup
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

url = "https://www.reddit.com/r/reddit.com/wiki/faq/"
reader = SimpleWebPageReader()
raw_data = reader.load_data([url])

# Extract clean text
from bs4 import BeautifulSoup
soup = BeautifulSoup(raw_data[0].text, 'html.parser')
text = "\n".join(p.text for p in soup.find_all('p'))
document = Document(text=text)

# Store in ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("reddit_faq")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Index it
index = VectorStoreIndex.from_documents([document], storage_context=storage_context)
```

### 2. Ask the Chatbot

```python
# chatbot.py
from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

api_key = "YOUR_GEMINI_API_KEY"

Settings.llm = Gemini(api_key=api_key, model_name="models/gemini-1.5-flash")
Settings.embed_model = GeminiEmbedding(api_key=api_key, model_name="models/embedding-001")

query_engine = index.as_query_engine()
response = query_engine.query("What is a Reddit downvote?")
print(response)
```

---

## 🔒 Content Safety

Implemented in `content_filter.py`. Blocks topics related to:
- Religion
- Politics
- Illegal activities
- Personal advice

Examples that will be blocked:
- "How do I hack a website?"
- "Can you tell me who to vote for?"
- "I'm feeling depressed, what should I do?"

---

## ✅ Evaluation System

Implemented in `llm_evaluator.py`. Evaluates responses with the following weights:
- Relevance: 40%
- Completeness: 30%
- Clarity: 20%
- Safety (if unsafe): 10%

It returns:
- A confidence score (0-100%)
- A pass/fail status
- Notes/comments

---

## 🧪 Testing

Run your full test suite using:

```python
# test_suite.py
from llm_evaluator import ResponseEvaluator
from chatbot import chatbot_fn
from prompts import test_prompts  # A file with 30 test prompts

def run_test_suite():
    evaluator = ResponseEvaluator()
    for prompt in test_prompts:
        response = chatbot_fn(prompt)
        result = evaluator.evaluate(prompt, response)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")
        print(f"Score: {result['score']}% | Status: {result['status']}")
        print("Notes:", *result['comments'])
```

---

## 🧾 Conclusion

This project demonstrates how to build a Reddit FAQ-focused chatbot with LLM-based intelligence and safety using:

- **LlamaIndex** for indexing and query handling
- **ChromaDB** for persistent vector storage
- **Gemini AI** for answering natural language queries
- **Content filtering** for ethical and safe interactions
- **Evaluator agent** for performance scoring and automated testing

---

## 📧 Contact

If you'd like help extending this project or integrating it into your app, feel free to reach out at cnaidu402@gmail.com. Please give credits too. 

---

```
