# Agentic RAG Company Knowledge

An agentic retrieval-augmented generation (RAG) prototype that combines a private Pinecone knowledge base, LLM-based evidence grading, Tavily web search, query rewriting, and final answer generation.

The current work is being developed in [`rag_backend/agentic_rag.ipynb`](./rag_backend/agentic_rag.ipynb).

## Architecture

![Agentic RAG architecture](./Screenshot%20From%202026-10-26%2015-28-17.png)

The diagram describes the target flow currently being designed:

1. A user submits a question.
2. A router LLM decides whether the question should use the private knowledge base or receive a direct LLM response.
3. Relevant chunks are retrieved from Pinecone using semantic search.
4. An LLM grades whether the private evidence is sufficient.
5. If the private evidence is weak, Tavily searches the web for current information.
6. Another LLM grades the web evidence.
7. A final answer is generated from the private knowledge base when the evidence is good.
8. Otherwise, a final answer is generated from the web results when those results are good.
9. If both sources remain insufficient, the system returns a fallback response after query-rewrite retries.
10. The selected answer is returned to the user with its source type.

## Current notebook nodes

The notebook currently defines the following node functions and routing decisions:

| Node                               | Function              | Responsibility                                                 |
| ---------------------------------- | --------------------- | -------------------------------------------------------------- |
| 1. Router                          | `route_query`         | Classifies the question as `kb` or `direct`.                   |
| 2. Private KB retrieval            | `retrieve_kb`         | Retrieves the top documents from Pinecone.                     |
| 3. Private evidence grader         | `grade_kb_evidence`   | Grades retrieved knowledge-base context as `good` or `weak`.   |
| 4. Web search                      | `search_web`          | Uses Tavily as the fallback source for current information.    |
| 5. Web evidence grader             | `grade_web_evidence`  | Grades Tavily results as `good` or `weak`.                     |
| 6. Query rewriter                  | `rewrite_query`       | Rewrites a weak query before another retrieval attempt.        |
| 7. Private answer generator        | `generate_from_kb`    | Generates an answer using only private knowledge-base context. |
| 8. Web answer generator            | `generate_from_web`   | Generates an answer using Tavily context.                      |
| 9. Direct answer generator         | `direct_answer`       | Handles general questions outside the private knowledge base.  |
| 10. Insufficient-evidence fallback | `answer_insufficient` | Explains that the available evidence was not sufficient.       |

The notebook also defines:

- `RouteDecision` for router output.
- `EvidenceGrade` for evidence-grader output.
- `AgentState` for question, query, retrieved documents, web results, grades, answer, source, and retry state.
- `MAX_RETRIES = 2` for weak web-evidence retry handling.

### Workflow status

The node logic and conditional routing functions are currently being developed. The final notebook cell contains the initial LangGraph workflow scaffold, and the complete executable graph wiring is the next major step.

## Knowledge-base ingestion

The notebook currently:

1. Loads the LangChain Agentic RAG documentation with `WebBaseLoader`.
2. Splits the source into approximately 29 chunks with `RecursiveCharacterTextSplitter`.
3. Generates 384-dimensional embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
4. Runs embeddings on the CPU because the local NVIDIA MX130 is not supported by the installed CUDA PyTorch build.
5. Creates or reuses the Pinecone index `homa-rag-knowledge-base`.
6. Stores vectors in the `langchain-agentic-rag` namespace.

The ingestion cell uses deterministic document IDs. This prevents rerunning the notebook from creating duplicate vectors in the same namespace.

## Requirements

- Python `>=3.9,<3.14`
- A Pinecone API key
- A Tavily API key
- A Groq API key for the configured LLM

The Python version restriction is required because the pinned `langchain-pinecone==0.2.13` package does not support Python 3.14.

## Setup

From the repository root:

```bash
python3.12 -m venv rag_backend/ragenv
source rag_backend/ragenv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r rag_backend/requirements.txt
```

In VS Code, select `rag_backend/ragenv` as the notebook kernel.

## Environment variables

Create `rag_backend/.env` locally and do not commit it:

```dotenv
TAVILY_DEV_API_KEY=your_tavily_key
PINECONE_API_KEY=your_pinecone_key
GROQ_DEV_API_KEY=your_groq_key
```

The repository `.gitignore` excludes environment files. Never place real API keys in the notebook or README.

## Running the notebook

Run the cells in order:

1. Load and validate environment variables.
2. Load and split the source document.
3. Create CPU embeddings.
4. Create or reuse the Pinecone index.
5. Upload chunks using deterministic IDs.
6. Create the retriever.
7. Define the agent state and node functions.
8. Build and compile the LangGraph workflow once that cell is implemented.

If an older run uploaded duplicate vectors before deterministic IDs were added, clear the project namespace once and then rerun ingestion:

```python
pinecone_client.Index(INDEX_NAME).delete(
    delete_all=True,
    namespace=NAMESPACE_NAME,
)
```

Only run the deletion command when it is safe to remove all vectors in that namespace.

## Technology stack

- Python
- LangChain and LangGraph
- Pinecone
- Sentence Transformers
- Tavily
- Groq-compatible chat model
- Jupyter Notebook
