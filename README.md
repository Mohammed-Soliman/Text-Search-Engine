# Text Search Engine with Chatbot

This project implements a full-featured text search engine from scratch, combining classical **Information Retrieval techniques** with modern **semantic search methods**, including TF-IDF, BM25 ranking, and BERT-based query expansion, now integrated with an intelligent **chatbot assistant** for conversational retrieval and answer guidance.

Course: DSAI 201 - Data Mining and Information Retrieval  
Institution: Zewail City of Science and Technology  
Semester: Spring 2026  

---

## 🚀 Project Overview

The system is designed to index and retrieve relevant documents efficiently from a large corpus, supporting both lexical and semantic search, with a chatbot layer that helps users ask natural-language questions and receive context-aware responses.

The engine is implemented across multiple components:

- Core IR pipeline (preprocessing + indexing)
- Ranking models (TF-IDF, BM25)
- Query enhancement (WordNet, BERT, Rocchio)
- Conversational chatbot (query translation + search orchestration)
- Interactive UI with evaluation dashboard

---

## 1️⃣ Core IR Pipeline - Data Processing and Indexing

- Data collection from 20 Newsgroups / custom datasets  
- Text preprocessing (tokenization, normalization)  
- Stopword removal and stemming/lemmatization  
- Inverted index construction with term frequencies  

---

## 2️⃣ Retrieval Models - Ranking and Scoring

- TF-IDF vector space model  
- BM25 probabilistic ranking (Bonus)  
- Efficient query-document matching  

---

## 3️⃣ Query Processing - Expansion and Enhancement

- WordNet-based synonym expansion  
- BERT embedding-based semantic expansion (Bonus)  
- Rocchio relevance feedback (Bonus)  
- Query spell correction (Bonus)  

---

## 4️⃣ Chatbot Layer - Conversational Search Assistant

- Natural-language query understanding and translation  
- Chat-driven search flow connected to retrieval modules  
- Context-aware response generation from retrieved content  
- Modular chatbot pipeline (document processor, translator, search handler)  

---

## 5️⃣ User Interface and Evaluation

- Interactive Streamlit UI for search and chatbot interaction  
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
- NLTK and spaCy  
- Scikit-learn  
- Transformers (BERT)  

---

## 📚 Dataset

- 20 Newsgroups dataset (~18,000 documents across 20 categories)  
- Supports extension to TREC or custom corpora  

---

## 📚 References

- Manning et al., *Introduction to Information Retrieval*  
- Jurafsky and Martin, *Speech and Language Processing*  
- Scikit-learn Documentation  
- Hugging Face Transformers Documentation  

---

## 👨‍💻 Authors

- Mohammed Soliman  
- Ziad Shaker  

---

## 📌 Key Takeaway

This project demonstrates how a modern search engine combines classical information retrieval techniques with semantic understanding and conversational interaction to deliver accurate, meaningful, and user-friendly search experiences, covering the full pipeline from raw text processing to intelligent ranking, evaluation, and chatbot-assisted exploration.
