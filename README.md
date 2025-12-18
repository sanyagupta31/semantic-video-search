# 🔍 Semantic Video Search Engine (Multimodal AI)

## 📌 Project Overview
This project is a **Multimodal RAG (Retrieval-Augmented Generation)** application that enables natural language search within video content. 

Unlike traditional metadata search (which relies on tags), this engine uses **OpenAI's CLIP model** to generate 512-dimensional vector embeddings for video frames. It allows users to query video footage conceptually (e.g., "a sad moment" or "rainbow") without any prior labeling.

## 🛠️ Tech Stack
* **AI Model:** OpenAI CLIP (`clip-ViT-B-32`) for multimodal embeddings.
* **Vector Database:** ChromaDB (for high-performance similarity search).
* **Computer Vision:** OpenCV (for frame extraction and preprocessing).
* **Frontend:** Streamlit (for the interactive web interface).
* **Language:** Python 3.10+

## ⚙️ Architecture
1.  **Ingestion:** The system accepts `.mp4` video files.
2.  **Preprocessing:** OpenCV extracts frames at 1 FPS to optimize storage vs. granularity.
3.  **Embedding:** CLIP converts frames into high-dimensional vectors.
4.  **Indexing:** Vectors are stored in a persistent ChromaDB collection.
5.  **Retrieval:** User text queries are converted to vectors; Cosine Similarity finds the closest frame matches.

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/sanyagupta31/semantic-video-search.git](https://github.com/sanyagupta31/semantic-video-search.git)
    cd semantic-video-search
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the application:
    ```bash
    streamlit run app.py
    ```

## 📸 Screenshots
![rainbow](<Screenshot 2025-12-18 121211.png>)