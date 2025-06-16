# Resume Evaluation with RAG

A Retrieval-Augmented Generation (RAG) system for evaluating resumes against job descriptions or search queries.

## Overview

This project uses AI to analyze resumes and evaluate them against specific job requirements or search criteria. The system loads resume data, indexes it using embeddings, and then evaluates matches using natural language processing.

## System Architecture

The application consists of:

1. **FastAPI Backend**: Serves the API endpoints for resume evaluation
2. **Vector Database**: Stores resume embeddings using Chroma
3. **ML Models**: Uses language models for evaluating relevance
4. **Data Processing**: Processes resumes from JSON or PDF formats

## Key Components

- `main.py`: FastAPI server with endpoints
- `linkedin_evaluate.py`: Loads and processes resume data
- `ai_search_query.py`: Performs the query against indexed resumes
- `get_embedding_function.py`: Provides embeddings for vector search

## Setup Instructions

### Prerequisites

- Python 3.8+ and <3.10
- FastAPI
- LangChain
- Chroma DB
- Mistral 7B (or other LLM)

### Installation

1. Clone this repository
```bash
git clone <repository-url>
cd rag-tutorial-v2
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables (create a `.env` file)
```
# Add your model keys and configurations here
```

4. Create data directories
```bash
mkdir -p data_json
mkdir -p chroma
mkdir -p data
```

## Usage

### Preparing Resume Data

You can prepare resume data in two formats:

#### JSON Format
Place your resume data in the `data_json` folder as JSON files with the following structure:

```json
{
  "content": "Resume text content...",
  "metadata": {
    "name": "Candidate Name",
    "email": "email@example.com"
  }
}
```

Or as a list of documents:

```json
[
  {
    "content": "Section 1 of resume...",
    "metadata": { "section": "experience" }
  },
  {
    "content": "Section 2 of resume...",
    "metadata": { "section": "education" }
  }
]
```

#### PDF Format
You can also add PDF resume files directly to the `data` folder. The system will automatically extract text content from these files during processing.

### Running the Server

Start the FastAPI server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Command-Line Usage (Without Frontend)

If you prefer to use the system directly from the command line without a frontend:

1. Create or update the vector database with your resume data:
```bash
python3 linkedin_evaluate.py --reset
```

2. Run a query against the indexed resumes:
```bash
python3 ai_search_query.py "Looking for a senior software engineer with 5+ years experience in Python"
```

The results will be printed to the console in JSON format.

## API Endpoints

### POST /linkedin

Evaluates resumes against a query.

**Request:**
```json
{
  "query": "Looking for a senior software engineer with 5+ years of experience in Python and cloud infrastructure"
}
```

**Response:**
```json
{
  "result": ["resume_id", "Yes/No", "Explanation of match"]
}
```

### POST /search

Searches resumes for matching skills or keywords.

**Request:**
- Form data with "query" field and resume files

**Response:**
```json
{
  "result": ["Matched resumes with scores"]
}
```

## Example Workflow

1. Add resume JSON files to the `data_json` folder
2. Start the server with `uvicorn main:app --reload`
3. Send a query to the `/linkedin` endpoint:

```javascript
const response = await fetch('http://localhost:8000/linkedin', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({ query: "Need an experienced data scientist with ML expertise" })
});
```

4. Review the results to find matching candidates

## License

[Your license information]
