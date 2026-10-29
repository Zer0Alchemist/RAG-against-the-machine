```text
┏━┓┏━┓┏━╸   ┏━┓┏━╸┏━┓╻┏┓╻┏━┓╺┳╸   ╺┳╸╻ ╻┏━╸   ┏┳┓┏━┓┏━╸╻ ╻╻┏┓╻┏━╸
┣┳┛┣━┫┃╺┓   ┣━┫┃╺┓┣━┫┃┃┗┫┗━┓ ┃     ┃ ┣━┫┣╸    ┃┃┃┣━┫┃  ┣━┫┃┃┗┫┣╸
╹┗╸╹ ╹┗━┛╺━╸╹ ╹┗━┛╹ ╹╹╹ ╹┗━┛ ╹ ╺━╸ ╹ ╹ ╹┗━╸╺━╸╹ ╹╹ ╹┗━╸╹ ╹╹╹ ╹┗━╸
```
*This project has been created as part of the 42 curriculum by sbelabbe.*

## Description

RAG Against the Machine is a Retrieval-Augmented Generation (RAG) system designed to answer technical questions about the **vLLM 0.10.1** codebase and documentation.

The project combines a custom **BM25 search engine** with **Qwen3-0.6B** to retrieve relevant source code or documentation and generate answers grounded in those sources.

## System Architecture

The pipeline consists of two main stages:

**Indexing:**

Source Files → Chunking → Tokenization → Inverted Index → BM25 Statistics → Pickle Serialization

**Question Answering:**

User Query → Tokenization → BM25 Retrieval → Top-K Sources → Prompt Augmentation → Qwen3-0.6B → Answer

The system is divided into modules for indexing, retrieval, augmentation, generation, evaluation, and data validation using Pydantic.

## Chunking Strategy

Python, Markdown, and text files are divided using LangChain's `RecursiveCharacterTextSplitter`.

- Maximum chunk size: 2,000 characters.
- Default overlap: 100 characters.
- Language-aware separators for Python and Markdown.
- Each chunk preserves its original file path and character offsets.

This allows the system to retrieve specific sections of the corpus rather than entire files.

## Retrieval Method

The project implements **BM25 (Best Matching 25)** without relying on an external search engine.

The retrieval process consists of:

1. Normalizing text by handling camelCase, snake_case, punctuation, and stop words.
2. Building an inverted index containing token frequencies for each chunk.
3. Computing inverse document frequency (IDF) and document lengths.
4. Ranking chunks using BM25 with `k1 = 1.5` and `b = 0.75`.
5. Returning the Top-K most relevant sources.

The index is serialized using Python's `pickle` module to avoid rebuilding it for every search.

## Instructions

### Installation

The project uses Python and `uv` for dependency management.

```bash
uv sync
```

### Example Usage

Run commands from the project root.

**Build the index:**

```bash
uv run python -m src index --max_chunk_size 2000
```

**Search for relevant sources:**

```bash
uv run python -m src search "What is LoRA?" --k 5
```

**Search a complete dataset:**

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

**Generate an answer:**

```bash
uv run python -m src answer "What is LoRA?" --k 5
```

The CLI also supports `answer_dataset` for batch generation and `evaluate` for calculating retrieval recall.

## Performance Analysis

The system was evaluated using the project's automated examination pipeline.

| Test | Result |
|---|---|
| Indexing | 7 seconds |
| Retrieval of 200 questions | 40 seconds |
| Code Recall@1 (reported test) | 34% |
| Code Recall@3 (reported test) | 60% |
| Code Recall@5 (reported test) | 65% |
| Code Recall@10 (reported test) | 70% |
| Final private examination | **4/4 PASS** |

The private examination requires documentation Recall@5 of at least 80% and code Recall@5 of at least 50%. The final examination passed both requirements.

## Design Decisions

- **BM25:** Chosen for efficient keyword-based retrieval without requiring a vector database.
- **Recursive chunking:** Preserves useful document structure while respecting the maximum context length.
- **Custom tokenizer:** Normalizes natural language and programming identifiers to improve matching.
- **Pickle:** Stores the index for reuse.
- **Pydantic:** Validates data exchanged between pipeline stages.
- **Python Fire:** Exposes the project through a command-line interface.
- **Qwen3-0.6B:** Generates answers from retrieved context.

## Challenges Faced

The main challenge was improving retrieval accuracy, particularly for Python source code. Aswell as making the evaluator

Initially, punctuation normalization merged unrelated identifiers, negatively affecting token matching. Improving the tokenizer increased code Recall@5 from 49% to 65% in the reported tests.

Other challenges included handling chunk boundaries, validating input parameters, managing Python type annotations, and ensuring dataset output filenames were compatible with the automated evaluator.

## Resources

- [LangChain Documentation](https://python.langchain.com/docs/) — Recursive text splitting.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/) — Tokenization and language model generation.
- [BM25 — explanation](https://www.youtube.com/watch?v=hiJcEaiuw_E) — Probabilistic information retrieval.
- My peers: ybajdou, oerrami


### AI Usage

Ai was used as a learning and debugging assistant throughout development, particularly for understanding BM25, IDF, Recall@K, recursive chunking, anf for Python mypy type annotations.

AI assistance was also used to review implementation decisions, investigate retrieval errors, and prepare project documentation. The retrieval implementation was developed and tested through iterative debugging and evaluation.