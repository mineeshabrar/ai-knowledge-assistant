# AI Knowledge Assistant

A local-first Retrieval-Augmented Generation (RAG) backend built with FastAPI,
Sentence Transformers, FAISS, and Ollama. Upload text, Markdown, PDF, or Word
documents, add notes, and ask questions about their contents. The API returns
answers with source excerpts and an explicit fallback when there is not enough
relevant context.

## Product direction

The current project is the backend foundation. The planned product is a full
personal knowledge **playground** with a web UI: a place to write and organize
notes, add documents, and work with an AI assistant that can answer questions
using that knowledge. The experience should bring the user's notes, uploaded
files, conversations, and cited AI responses together rather than making them
use the API directly.

Planned UI capabilities include:

- A notes workspace for creating, editing, organizing, and searching personal
  notes.
- A document library for uploading and browsing files, with clear indexing
  status and document management.
- An AI chat assistant that can answer from the user's notes and documents,
  show supporting source excerpts, and clearly say when it cannot find enough
  information.
- Conversation history and controls for choosing which notes or documents the
  assistant should use.
- Helpful knowledge-work actions such as summarizing notes, explaining a topic
  from saved material, and creating study questions or action items.

These are product goals, not features currently provided by the backend. The
current API supports adding notes/documents, listing ingested documents, and
asking questions; the web interface, editable note management, and conversation
history are future work.

## Database and storage

This project does **not** use a traditional relational database such as PostgreSQL
or SQLite. It uses a small, file-based storage layer:

- **FAISS** (`data/knowledge.faiss`) stores the dense embedding vectors and
  performs nearest-neighbor similarity search. The index is an in-memory
  `IndexFlatIP` index persisted to disk; because vectors are normalized before
  insertion, inner-product ranking is cosine-similarity ranking.
- **JSON metadata** (`data/chunks.json`) stores each chunk's text, source filename,
  document ID, and chunk number. FAISS returns a vector position; this JSON list
  at the same position tells the backend which source text and citation details
  belong to that vector.
- **JSON document catalog** (`data/documents.json`) stores each ingested
  document's ID, filename, and chunk count for `GET /documents`.

These files are created under `DATA_DIR`, which defaults to `data/` in the
process's working directory. The original uploaded file is **not** retained;
the extracted text chunks and their embeddings are what get persisted. Notes
are saved the same way as documents. The embedding model itself is downloaded
and cached by Sentence Transformers in its normal model cache, not in the FAISS
data directory.

When the service starts, it loads the FAISS index and JSON metadata. It checks
that the index vector count matches the number of saved chunks, and reports an
error if they are inconsistent rather than silently returning incorrect
citations. Writes use temporary files and replace the target files after writing.
The FAISS index is local and simple to run, but it does not provide SQL queries,
multi-user access control, document deletion/update endpoints, or database
transactions across all three files. Back up the complete `DATA_DIR` directory
to preserve the knowledge base.

## Backend walkthrough

### 1. Application startup

`app/config.py` reads environment-variable settings into one validated settings
object. `app/main.py` creates the FastAPI application and a `VectorStore`.
`app/vector_store.py` initializes the data directory, loads the persisted index
and metadata, and loads the embedding model the first time it is needed.

### 2. Ingesting documents and notes

Both ingestion routes turn submitted content into the same internal pipeline:

1. `POST /documents` accepts a multipart file upload. The backend limits its size,
   allows `.txt`, `.md`, `.pdf`, and `.docx`, and extracts text using UTF-8,
   `pypdf`, or `python-docx`.
2. `POST /notes` accepts a title and text as JSON. It bypasses file extraction
   but otherwise uses the same indexing process.
3. `app/documents.py` normalizes whitespace and splits text into overlapping
   chunks. Chunk size and overlap are configurable.
4. The Sentence Transformers model converts each chunk into a vector. Vectors are
   normalized and added to FAISS.
5. The store assigns a document ID, associates every vector with its source
   metadata, and persists the updated FAISS index and JSON files.

The upload response gives the new document ID, filename, and chunk count. Empty
documents, unsupported file types, and uploads over the configured limit are
rejected with an HTTP error.

### 3. Answering a question (RAG)

`POST /query` performs retrieval first and generation second:

1. Sentence Transformers embeds the question using the same model used for
   documents.
2. FAISS finds the `RETRIEVAL_TOP_K` nearest chunks. Their inner-product scores
   are cosine similarities because the vectors were normalized.
3. The backend discards matches below `MIN_SIMILARITY`. If no matches remain, it
   returns the insufficient-context fallback and does not call Ollama.
4. Otherwise, it sends the question and numbered source excerpts to Ollama's
   `/api/generate` endpoint. The prompt asks the model to use only the excerpts,
   treat excerpt text as reference material rather than instructions, and cite
   sources as `[1]`, `[2]`, and so on.
5. The response contains the generated answer and the exact retrieved source
   excerpts, filenames, chunk numbers, and similarity scores. Clients can show
   these alongside the answer to let users check its grounding.

The citations are requested from the language model in its answer; the separate
`sources` array is the authoritative mapping from citation numbers to retrieved
excerpts. As with any generative model, citations in the prose should be checked
against that array.

### Backend modules

- `app/main.py` — FastAPI app, request validation, HTTP routes, RAG orchestration,
  and calls to Ollama.
- `app/documents.py` — supported-file text extraction and overlapping chunk
  creation.
- `app/vector_store.py` — embedding model integration, FAISS similarity search,
  metadata associations, and local persistence.
- `app/config.py` — environment configuration and validation.
- `tests/test_documents.py` — tests for text extraction, chunking, empty content,
  and invalid chunk settings.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- An Ollama chat model pulled locally (default: `llama3.2`)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2
cp .env.example .env
uvicorn app.main:app --reload
```

The embedding model is downloaded the first time it is needed. Interactive API
documentation is available at `http://localhost:8000/docs`.

## API

### `GET /health`

Returns `{"status":"ok"}` when the API process is responding.

### `POST /documents`

Upload a `.txt`, `.md`, `.pdf`, or `.docx` file using
`multipart/form-data` with the field name `file`. The default maximum size is
15 MB.

```bash
curl -F "file=@meeting-notes.pdf" http://localhost:8000/documents
```

### `POST /notes`

Add plain text directly as a note:

```bash
curl -H "Content-Type: application/json" \
  -d '{"title":"Project decisions","text":"The launch is planned for June."}' \
  http://localhost:8000/notes
```

### `GET /documents`

Lists ingested document records, including IDs, filenames, and chunk counts.

### `POST /query`

Ask a question using JSON:

```bash
curl -H "Content-Type: application/json" \
  -d '{"question":"When is the launch planned?"}' \
  http://localhost:8000/query
```

Successful answers have this shape:

```json
{
  "answer": "The launch is planned for June [1].",
  "sources": [
    {
      "number": 1,
      "filename": "Project decisions.txt",
      "chunk_index": 0,
      "score": 0.72,
      "text": "The launch is planned for June."
    }
  ],
  "insufficient_context": false
}
```

When no source meets the relevance threshold, `answer` contains the fallback,
`sources` is empty, and `insufficient_context` is `true`. Invalid uploads return
client errors; failures communicating with Ollama return `502`.

## Configuration

Set these environment variables before starting Uvicorn. `.env.example` lists
the defaults; install a dotenv loader separately if you want Uvicorn/Python to
load a `.env` file automatically, or export the variables in your shell.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATA_DIR` | `data` | Directory for FAISS index and JSON metadata |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence Transformers embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Ollama text-generation model |
| `RETRIEVAL_TOP_K` | `5` | Maximum number of nearest chunks to retrieve |
| `MIN_SIMILARITY` | `0.30` | Minimum cosine similarity for a chunk to be used |
| `CHUNK_SIZE` | `900` | Approximate maximum characters per chunk |
| `CHUNK_OVERLAP` | `150` | Approximate overlapping characters between chunks |
| `MAX_UPLOAD_MB` | `15` | Maximum uploaded file size in megabytes |

## Run tests

```bash
python -m unittest discover -s tests -v
```

## Next steps

The following roadmap describes how to grow the current API into the planned
personal knowledge playground:

1. **Build the web playground UI** — add a responsive app with a notes area,
   document library, and AI chat in one workspace. Show source citations and
   excerpts alongside answers, plus clear loading, indexing, and error states.
2. **Add full note and document management** — implement create, read, update,
   delete, search, and organization features. Re-index changed content and
   prevent or identify duplicate uploads.
3. **Add persistent conversations and context controls** — save chat history and
   let users choose whether a question searches all knowledge or selected notes
   and documents.
4. **Improve retrieval and answer quality** — evaluate answers against their
   source passages, verify citation references, and explore hybrid semantic and
   keyword search for exact names, dates, and terms.
5. **Add useful assistant actions** — support grounded summaries, explanations,
   study flashcards or quizzes, and meeting/project action-item extraction.
   Keep generated results linked to their source material.
6. **Add tests and operational safeguards** — cover API ingestion and query
   flows, insufficient-context behavior, and Ollama failures. Add structured
   logging, authentication before exposing the app beyond a trusted local
   machine, and request/rate limits.
7. **Plan for growth** — back up the complete data directory. If the project
   grows into a multi-user service, move document ownership and metadata into a
   transactional database and consider PostgreSQL with pgvector for vector
   search and access control.
