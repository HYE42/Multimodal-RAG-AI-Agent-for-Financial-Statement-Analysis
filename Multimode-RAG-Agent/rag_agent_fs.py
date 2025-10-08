from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv 
load_dotenv(override=True)
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# LLM & Embeddings
# ---------------------------------------------------------------------------
MODEL_NAME = "deepseek-chat"
model = init_chat_model(model=MODEL_NAME, model_provider="deepseek", temperature=0)
grader_model = init_chat_model(model=MODEL_NAME, model_provider="deepseek", temperature=0)

embed = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY", "your-openai-api-key-here"),
    base_url="https://ai.devtool.tech/proxy/v1",
    model="text-embedding-3-small",
)

# ---------------------------------------------------------------------------
# Vector store & Retriever tool
# ---------------------------------------------------------------------------
VS_PATH = "fs_db"
MINERU_DIR = "File_fs" 

# If the FAISS index doesn't exist, build it from local markdown so the dev server can start.
def _ensure_faiss_index():
    index_file = os.path.join(VS_PATH, "index.faiss")
    if os.path.exists(index_file):
        return
    os.makedirs(VS_PATH, exist_ok=True)
    source_md = os.path.join(os.path.dirname(__file__), MINERU_DIR, "full_updated.md") 
    if not os.path.exists(source_md):
        # Fallback to another sample if available
        alt_md = os.path.join(os.path.dirname(__file__), MINERU_DIR, "full.md")
        source_md = alt_md if os.path.exists(alt_md) else None
    if not source_md:
        # Nothing to index; leave creation to the user
        return
    with open(source_md, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # Simple chunking to avoid extra dependencies
    chunk_size = 1000
    overlap = 200
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += max(1, chunk_size - overlap)
    if chunks:
        vs = FAISS.from_texts(chunks, embed)
        vs.save_local(VS_PATH)

_ensure_faiss_index()

vector_store = FAISS.load_local(
    folder_path=VS_PATH,
    embeddings=embed,
    allow_dangerous_deserialization=True,
)
retriever_tool = create_retriever_tool(
    vector_store.as_retriever(search_kwargs={"k": 3}),
    name="retrieve_fs",
    description="Search and return relevant sections from the financial statements materials.",
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are a technical assistant focused on corporate financial statements and reports. "
    "Answer ONLY questions about financial reporting and analysis (e.g., 10-K/10-Q, annual reports, income statement, "
    "balance sheet, cash flow statement, MD&A, notes, accounting policies, key ratios, segment data, risk factors). "
    "If the user question is NOT related to financial statements or corporate filings, reply: 'I cannot answer questions that are not related to financial reporting.'\n"
    "When additional context is required, you may call the provided tool `retriever_tool` to fetch relevant passages "
    "from filings or financial-reporting materials.\n\n"
    "Rules & Style:\n"
    "- Be precise, neutral, and source-grounded. Prefer figures and terms found in the filings.\n"
    "- State currency, units, and period (e.g., FY2023, Q2 2025) when citing numbers.\n"
    "- If GAAP vs. non-GAAP metrics appear, explain the difference briefly and cite the reconciliation if present.\n"
    "- If IFRS/US GAAP differences matter, highlight them succinctly.\n"
    "- For calculations (margins, growth rates, coverage ratios, common-size, etc.), show concise steps and formulas.\n"
    "- If information is not present in the provided context, say: 'I do not have the information'\n"
    "- This content is for educational purposes only and is not investment, legal, or tax advice."
    "You may call the provided tool `retriever_tool` when additional context is required."
)

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question about financial statements.\n"
    "Retrieved document:\n{context}\n\nUser question: {question}\n"
    "Return 'yes' if the document helps answer the question (contains figures, definitions, policies, disclosures, or "
    "exhibits related to the query), otherwise return 'no'."
)

REWRITE_PROMPT = (
    "You are rewriting user questions to make them more relevant to the financial reporting materials.\n"
    "Your job is to refine or clarify the question so it references specific statements, metrics, periods, or entities when possible "
    "(e.g., 'Q2 2025 revenue growth by segment', 'cash from operations vs. net income (FY2023)', "
    "'goodwill impairment policy under US GAAP'). Keep it concise and faithful to intent.\n\n"
    "Original question:\n{question}\nImproved question:"
)

ANSWER_PROMPT = (
    "You are an assistant for answering questions about corporate financial statements and filings. "
    "Use ONLY the provided context to answer as completely and accurately as possible. "
    "If necessary, you may ask `retriever_tool` for additional passages.\n\n"
    "Whenever relevant, include examples, code blocks, or image references that appear in the source material. "
    "Use standard Markdown format for your output.\n\n"
    
    "Guidelines:\n"
    "- Cite specific numbers with units, currency, and period (e.g., '$2.1B revenue in FY2024').\n"
    "- If the context includes tables or bullet lists, reproduce only the relevant parts for clarity.\n"
    "- Show brief calculation steps for ratios or deltas (formula → numbers → result).\n" #
    "- Define technical terms briefly when first used (e.g., 'FCF = CFO − CapEx').\n" #
    "- If the answer is unknown or not present in the context, say: 'I do not have the information.'\n"
    "- Use standard Markdown; prefer fenced code blocks (```) for formulas or multi-step calculations.\n\n"
    "- If the context includes Markdown images (e.g. ![alt](url)), and the image is relevant, you may include it in the response.\n"
    "- Keep the response structured and easy to read with proper Markdown sections if needed.\n"

    "Question: {question}\n"
    "Context: {context}"
)

# ---------------------------------------------------------------------------
# LangGraph Nodes
# ---------------------------------------------------------------------------
async def generate_query_or_respond(state: MessagesState):
    """LLM decides to answer directly or call retriever tool."""
    response = await model.bind_tools([retriever_tool]).ainvoke(
        [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            *state["messages"],
        ]
    )
    return {"messages": [response]}


class GradeDoc(BaseModel):
    binary_score: str = Field(description="Relevance score 'yes' or 'no'.")


async def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content  # original user question
    ctx = state["messages"][-1].content      # retriever output
    prompt = GRADE_PROMPT.format(question=question, context=ctx)
    result = await grader_model.with_structured_output(GradeDoc).ainvoke([
        {"role": "user", "content": prompt}
    ])
    return "generate_answer" if result.binary_score.lower().startswith("y") else "rewrite_question"


async def rewrite_question(state: MessagesState):
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    resp = await model.ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [{"role": "user", "content": resp.content}]}


async def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    ctx = state["messages"][-1].content
    prompt = ANSWER_PROMPT.format(question=question, context=ctx)
    resp = await model.ainvoke([{"role": "user", "content": prompt}])
    return {"messages": [resp]}

# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
workflow = StateGraph(MessagesState)
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "generate_query_or_respond")
workflow.add_edge("generate_query_or_respond", "retrieve")
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

rag_agent = workflow.compile(name="rag_agent")