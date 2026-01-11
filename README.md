# TED RAG Project

A Retrieval-Augmented Generation (RAG) system for querying TED Talks using natural language.

## Features

- **Semantic Search**: Find relevant TED talks using natural language queries
- **Context-Aware Responses**: Get answers based on actual TED talk transcripts
- **RESTful API**: Simple HTTP endpoints for integration
- **Deployed on Vercel**: Fast, serverless deployment

## API Endpoints

### `GET /api/stats`
Returns RAG system configuration.

**Response:**
````json
{
  "chunk_size": 1024,
  "overlap_ratio": 0.2,
  "top_k": 15
}
````

### `POST /api/prompt`
Query the TED talks database.

**Request:**
````json
{
  "question": "Which TED talk focuses on education?"
}
````

**Response:**
````json
{
  "response": "Natural language answer...",
  "context": [
    {
      "talk_id": "1234",
      "title": "Talk Title",
      "chunk": "Transcript excerpt...",
      "score": 0.85
    }
  ],
  "Augmented_prompt": {
    "System": "System prompt used...",
    "User": "User prompt used..."
  }
}
````

## Tech Stack

- **Backend**: FastAPI (Python)
- **Vector Database**: Pinecone
- **Embeddings & LLM**: LLMod API
- **Deployment**: Vercel
- **Data Processing**: Pandas

## Project Structure
````
ted_rag_project/
├── api/
│   └── index.py          # Main FastAPI application
├── app/
│   ├── rag.py           # RAG logic & retrieval
│   ├── utils.py         # Data processing utilities
│   └── embed_data.py    # Embedding generation
├── data/
│   └── ted_talks.csv    # TED talks dataset
├── index.html           # Frontend UI
├── style.css
├── script.js
├── vercel.json          # Vercel configuration
└── requirements.txt     # Python dependencies

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/NetaSilam/ted-rag-project.git
cd ted_rag_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file with your API keys:
````
LLMOD_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=ted-rag-index
````

4. Run locally:
```bash
uvicorn api.index:app --reload
```

## Deployment

Deployed on Vercel at: [https://ted-rag-zeta.vercel.app](https://ted-rag-zeta.vercel.app)

Environment variables are configured in Vercel Dashboard.

