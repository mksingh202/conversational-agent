# Conversational Intelligence Agent

A production-style GenAI conversational assistant that enables users to query PDF documents while maintaining conversational history:
- **OpenAI (GPT-4o-mini + Embeddings)**
- **LangGraph Agent Workflow**
- **PostgreSQL + pgvector**
- **Hybrid Retrieval (Vector + BM25)**
- **Semantic Chunking**
- **Citation-based Answers**
---

## Workflow Diagram
![Agentic Workflow](workflow.png)

## Project Structure
```
conversational-agent
├── app.py              # CLI entrypoint + chat loop
├── ingestion.py        # PDF loading, table extraction, semantic chunking
├── search.py           # Hybrid retrieval + RRF
├── bm25.py             # Keyword search index
├── agents.py           # Answer generation with citations
├── workflow.py         # Graph execution wrapper
├── graph.py            # LangGraph workflow definition
├── db.py               # PGVector integration
├── router.py           # Query classification
├── rewriter.py         # Follow-up question rewriting
├── requirements.txt    # Dependencies
└── README.md           # User Manual
```

## Prerequisites

- Python 3.9+
- PostgreSQL 14+
- pgvector extension
- OpenAI API Key

## Database Setup
Open PostgreSQL and run these command:
```sql
CREATE DATABASE adani_data;

CREATE EXTENSION IF NOT EXISTS vector;
```

## Create .env File
```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
DB_HOST=localhost
DB_PORT=5432
DB_NAME=<db_name>
DB_USER=<db_user>
DB_PASSWORD=<db_password>
```

## Install Dependencies
```pip install -r requirements.txt```

## Run the Streamlit Application
```bash 
python app.py <path_to_pdf>
```

## Example Output
```bash 
Answer:
Consolidated EBITDA in H1 FY26 is Rs. 7,688 crore [p1:c1.0].

Retrieved Chunks:
Rank  Citation     RRF      Snippet
```