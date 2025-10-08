# Multimodal RAG AI Agent for Financial Statement Analysis
## Overview
This project implements an AI agent powered by Retrieval-Augmented Generation (RAG) to analyze and interpret financial statements across text, tables, and images.

## Key Features
1. Self-Adaptive RAG Pipeline
   
Dynamically determines whether retrieval is needed based on user queries. This agent decides when and how to use retrieval, rewrites ambiguous queries for better precision, and unifies multimodal data into a structured context for analysis.

If the question is ambiguous, the agent rewrites and refines it before performing RAG.

Produces contextually aligned answers grounded in financial report sections.

2. Multimodal Document Understanding
   
Supports text, tables, and images from corporate financial reports (PDFs).

Uses [MinerU](https://github.com/opendatalab/MinerU) to convert PDFs into Markdown format to preserve document hierarchy — enabling precise context retrieval by section and heading.

3. Intelligent Chunking & Vectorization
   
The add_mineru_document.py script segments Markdown files into semantically coherent chunks.

Each chunk is stored in a FAISS vector index (index.faiss) with associated metadata.

4. RAG Agent Construction 
   
Integrates LLMs with FAISS retriever for grounded financial Q&A.

Reconstructs context from document metadata for transparency and traceability.

5. Privacy-Friendly Local Processing
   
MinerU supports both local client and API modes. Local mode keeps extracted images and OCR data on-device, ensuring compliance with enterprise privacy requirements.

6. Interactive Chat Interface

Provides a friendly web-based chatbot UI powered by [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui).

Allows users to chat naturally with the financial analysis agent and visualize responses in real time.

Built upon [Multimode-RAG-Agent](https://github.com/AlexLIAOPOLY/Multimode-RAG-Agent)

## Example Use Cases

You can query the agent with complex financial questions such as:

`“Compare year-over-year revenue growth trends in the latest report.”`

`“Extract net cash flow and summarize contributing factors.”`

`“Explain anomalies between balance sheet and cash flow in Q2.”`

The agent will:

- Rewrite the query if unclear.

- Retrieve relevant financial report sections.

- Generate a clear, source-grounded answer with references.

## Tech Stack
Language Models: DeepSeek & OpenAI-compatible APIs

Retrieval Engine: FAISS

Multimodal Extraction: MinerU 

Document Processing: Python + Markdown Parser

Orchestration: LangGraph

## Usage

Run Backend (LangGraph RAG Agent)
```
cd Multimode-RAG-Agent
langgraph dev
```

Run Frontend (Agent Chat UI)
```
cd Multimode-RAG-Agent/agent-chat-ui
pnpm dev
```

Please install [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui) before use. 

Create .env file for APIs before use.
```
# OpenAI API Configuration 
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx

# DeepSeek API Configuration
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# LangGraph API Configuration
LANGGRAPH_API_URL=http://localhost:2024

# LangSmith Configuration
LANGSMITH_API_KEY=lsv2_xxxxxxxxxxxxxxxxxxxxxxxx
```
