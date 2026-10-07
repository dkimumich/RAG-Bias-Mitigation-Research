# RAG Bias Mitigation

A Python pipeline that decomposes and rewrites user queries, retrieves supporting context, and checks generated answers for social bias in retrieval-augmented generation (RAG) systems.

## Problem

RAG systems ground language model answers in retrieved documents, but retrieved context can carry and amplify social bias. This project explores how to detect that bias and reduce it before an answer reaches the user.

## Approach

The pipeline runs in three stages:

1. **Query decomposition and rewriting:** An input question is broken into smaller sub-queries using the OpenAI API, and each sub-query is rewritten to remove biased framing and improve retrieval, so each part can be retrieved and checked on its own.
2. **Retrieval:** Relevant context is retrieved for each sub-query using BM25 keyword search and embedding-based vector similarity scoring against labeled fairness benchmarks.
3. **Bias check:** Retrieved context and generated answers are evaluated for bias.

```
User query -> Decomposition + Rewrite -> Retrieval -> Bias check -> Final answer
```

## Datasets

The pipeline works with a wide range of datasets rather than a single benchmark. Some of the datasets used include:

- **Wikipedia:** used as a retrieval corpus for supporting context.
- **BBQ (Bias Benchmark for QA):** a question-answering benchmark for measuring social bias.
- **BibleQA:** a question-answering dataset used to examine bias in religious texts and how LLMs respond to them.

## Project structure

```
.
|-- main.py              # Entry point: run the full pipeline on a query
|-- src/
|   |-- decomposition.py         # Query decomposition [and rewriting -- confirm]
|   |-- embedders.py             # Embedding models for retrieval
|   |-- bias_detection.py        # Bias detection on retrieved context and answers
|   |-- bias_grps.py             # Bias group definitions
|   |-- metrics.py               # Fairness and evaluation metrics
|   |-- experiments.py           # Experiment runner
|   |-- baseline-ASRank.py       # Baseline for comparison
|   |-- multi_dataset_loader.py  # Loaders for multiple datasets (Wikipedia, BBQ, ...)
|   |-- client.py, db.py         # LLM client and storage helpers
|   |-- LLM_extract/             # LLM-based extraction pipeline and experiments
|   |-- updated/                 # Latest pipeline (new_pipeline.py) and full evaluations
|-- debiasing-rag/       # [one-line description of this folder]
|-- BBQ/                 # BBQ benchmark data
|-- BibleQA/             # BibleQA question-answering data
|-- corpus_data/         # Retrieval corpus data
|-- bm25_example.py      # BM25 retrieval example
|-- rag_example.txt      # Example RAG prompt/output
|-- setup.sh, experiment.sh, environment_check.py   # Environment setup and experiment scripts
|-- requirements.txt
```
