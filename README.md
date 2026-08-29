# Basic Search Engine with Chatbot

This project implements a full-featured text search engine from scratch, combining classical **Information Retrieval techniques** with modern **semantic search methods**, including Boolean, TF-IDF, BM25 ranking, and BERT-based query expansion.

Course: DSAI 201 - Data Mining and Information Retrieval  
Institution: Zewail City of Science and Technology  
Semester: Spring 2026  

---

## 🚀 Project Overview

The system is designed to index and retrieve relevant documents efficiently from a large corpus, supporting both lexical and semantic search.

The engine is implemented across multiple components:

- Core IR pipeline (preprocessing + indexing)
- Ranking models (Boolean, TF-IDF, BM25)
- Query enhancement (WordNet, BERT, Rocchio)
- Conversational chatbot (query translation + search orchestration)
- Interactive UI with evaluation dashboard

---

## 1️⃣ Core IR Pipeline - Data Processing and Indexing

- Data collection from BBC-News / custom datasets  
- Text preprocessing (tokenization, normalization)  
- Stopword removal and stemming/lemmatization  
- Inverted index construction with term frequencies  

---

## 2️⃣ Retrieval Models - Ranking and Scoring

- Boolean (AND) retrieval
- TF-IDF vector space model  
- BM25 probabilistic ranking
- Efficient query-document matching  

---

## 3️⃣ Query Processing - Expansion and Enhancement

- WordNet-based synonym expansion  
- BERT embedding-based semantic expansion
- Rocchio relevance feedback
- Query spell correction

---

## 5️⃣ User Interface and Evaluation

- Interactive Streamlit UI for search  
- Real-time ranked results and conversational responses  
- Evaluation metrics (MAP, nDCG, Precision@K)  
- Visualization of model and system performance  

---

## 🛠 Design Highlights

- Modular and scalable architecture  
- Hybrid retrieval (lexical + semantic)  
- Conversational interface on top of core IR  
- Real-time interactive experience  
- Integrated evaluation pipeline  
- Extensible with clustering and spell correction  
- Clean separation of pipeline stages  

---

## 🛠 Tools Used

- Python  
- Streamlit  
- NLTK
- Scikit-learn  
- Transformers (BERT)  

---

## 📚 Dataset

- BBC News dataset
- Supports uploading local data through the UI  

---

## 📚 References

- Manning et al., *Introduction to Information Retrieval*  
- *DSAI 201* Course Materials 

---

## 👨‍💻 Authors

- Mohammed Soliman  
- Ziad Shaker  

---

## 📌 Key Takeaway

This project demonstrates how a modern search engine combines classical information retrieval techniques with semantic understanding and conversational interaction to deliver accurate, meaningful, and user-friendly search experiences, covering the full pipeline from raw text processing to intelligent ranking, evaluation, and chatbot-assisted exploration.
