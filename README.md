# Text Search Engine

This project implements a full-featured text search engine from scratch, combining classical **Information Retrieval techniques** with modern **semantic search methods**, including TF-IDF, BM25 ranking, and BERT-based query expansion, all wrapped in an interactive Streamlit interface with evaluation metrics.

Course: DSAI 201 – Data Mining and Information Retrieval  
Institution: Zewail City of Science and Technology  
Semester: Spring 2026  

---

## 🚀 Project Overview

The system is designed to index and retrieve relevant documents efficiently from a large corpus, supporting both lexical and semantic search.

The engine was implemented across multiple components:

- Core IR pipeline (preprocessing + indexing)
- Ranking models (TF-IDF, BM25)
- Query enhancement (WordNet, BERT, Rocchio)
- Interactive UI with evaluation dashboard

---

## 1️⃣ Core IR Pipeline – Data Processing & Indexing

- Data collection from 20 Newsgroups / custom datasets  
- Text preprocessing (tokenization, normalization)  
- Stopword removal and stemming/lemmatization  
- Inverted index construction with term frequencies  

---

## 2️⃣ Retrieval Models – Ranking & Scoring

- TF-IDF vector space model  
- BM25 probabilistic ranking (Bonus)  
- Efficient query-document matching   

---

## 3️⃣ Query Processing – Expansion & Enhancement

- WordNet-based synonym expansion  
- BERT embedding-based semantic expansion (Bonus)  
- Rocchio relevance feedback (Bonus)  
- Query spell correction (Bonus)  

---

## 4️⃣ User Interface & Evaluation

- Interactive Streamlit UI  
- Real-time search results  
- Evaluation metrics (MAP, nDCG, Precision@K)  
- Visualization of performance  

---

## 🛠 Design Highlights

- Modular and scalable architecture  
- Hybrid retrieval (lexical + semantic)  
- Real-time interactive interface  
- Integrated evaluation pipeline  
- Extensible with clustering and spell correction  
- Clean separation of pipeline stages  

---

## 🛠 Tools Used

- Python  
- Streamlit  
- NLTK & spaCy  
- Scikit-learn  
- Transformers (BERT)  

---

## 📚 Dataset

- 20 Newsgroups dataset (~18,000 documents across 20 categories)  
- Supports extension to TREC or custom corpora  

---

## 📚 References
- Manning et al., Introduction to Information Retrieval
- Jurafsky & Martin, Speech and Language Processing
- Scikit-learn Documentation
- HuggingFace Transformers Documentation

---

## 👨‍💻 Authors
- Mohammed Soliman
- Ziad Shaker
---

## 📌 Key Takeaway

This project demonstrates how a modern search engine combines classical information retrieval techniques with semantic understanding to deliver accurate and meaningful search results, highlighting the full pipeline from raw text processing to intelligent ranking and evaluation.