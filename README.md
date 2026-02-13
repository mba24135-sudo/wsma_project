# Corporate Narrative & NLP Intelligence Dashboard

An institutional-grade Market Intelligence Dashboard designed to extract, analyze, and visualize corporate narratives using Advanced Natural Language Processing (NLP). Built with Python and Streamlit, it replicates the UI/UX of modern financial SaaS platforms like Concall.in.

## 🧠 Core Features
* **Entity-Level Sentiment Analysis (ELSA):** Utilizes `ProsusAI/finbert` and `bert-base-NER` to isolate financial news sentiment down to specific organizations and people.
* **Strategic Aspect Tagging:** Uses Zero-Shot Classification (`bart-large-mnli`) to categorize unstructured news into distinct business pillars (e.g., Regulatory, Digital Transformation).
* **Forward-Looking Statement (FLS) Extraction:** Parses dense earnings call transcripts to isolate and score management guidance and future projections.
* **Automated Executive Summaries:** Deploys `bart-large-cnn` to chunk and summarize 4-quarter earnings transcripts automatically.

## ⚙️ Tech Stack
* **Frontend:** Streamlit, Plotly, Custom CSS
* **Backend:** Python, Pandas, Numpy
* **Data Ingestion:** Refinitiv Eikon API
* **AI/Machine Learning:** Hugging Face Transformers, PyTorch

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)