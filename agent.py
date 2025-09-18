# file: tavily_langgraph_example.py
from __future__ import annotations

import os
from typing import TypedDict, List, Dict, Any, Optional, Callable, Annotated
import json
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------- State ----------
class GraphState(TypedDict):
    question: str
    query: str
    results: List[Dict[str, Any]]   # tavily result items (title, url, content)
    answer: str
    attempts: int
    max_attempts: int
    follow_up_query: str
    last_prompt: str
    raw_reason: str
    interactions: List[Dict[str, Any]]
    search_payload: Any
    messages: Annotated[List[Dict[str, Any]], add_messages]
    search_log_path: Optional[str]

# ---------- Tools / Models ----------
# Get Tavily API key and initialize retriever
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable is not set")

tavily = TavilySearch(max_results=5, tavily_api_key=tavily_api_key)  # returns structured results

# Get API key from environment variable
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514", 
    temperature=0,
    anthropic_api_key=api_key
)

# ---------- Nodes ----------
def search_node(state: GraphState) -> GraphState:
    q = (state.get("query") or state["question"]).strip()
    if not q:
        raise ValueError("Empty question.")
    # TavilySearch returns search results as a string or dict
    search_result = tavily.invoke(q)
    
    # Convert to list format for consistency
    if isinstance(search_result, str):
        # If it's a string, create a simple result structure
        results = [{"content": search_result, "title": "Search Result", "url": ""}]
    elif isinstance(search_result, list):
        results = search_result
    else:
        # If it's something else, wrap it
        results = [{"content": str(search_result), "title": "Search Result", "url": ""}]
    # Optional logging of full search payload per attempt
    try:
        search_log_path = state.get("search_log_path")
        if search_log_path:
            attempt_num = int(state.get("attempts", 0)) + 1
            with open(search_log_path, "a", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"Attempt: {attempt_num}\n")
                f.write(f"Query: {q}\n")
                f.write("Full Tavily Payload:\n")
                f.write(json.dumps(search_result, ensure_ascii=False, indent=2))
                f.write("\n")
    except Exception:
        pass

    return {**state, "results": results, "search_payload": search_result}

def reason_node(state: GraphState) -> GraphState:
    q = state["question"]
    results = state.get("results", [])
    # Build a compact context for the LLM
    # Keep it short to control token usage
    def fmt_item(i, item):
        title = item.get("title") or ""
        url = item.get("url") or ""
        snippet = (item.get("content") or "").strip()
        snippet = " ".join(snippet.split())[:800]  # trim snippet
        return f"[{i}] {title}\nURL: {url}\nSnippet: {snippet}\n"

    context_blocks = "\n\n".join(fmt_item(i+1, r) for i, r in enumerate(results[:5]))

    prompt = f"""
You are a careful researcher. Use ONLY the context below.

Question:
{q}

Context (web search results):
{context_blocks}

Instructions:
- If the context is sufficient, produce a concise answer with inline citations [1], [2], etc.
- If the context is insufficient, do NOT guess; instead propose ONE improved web search query that would likely retrieve the needed info.
- If question requires multiple queries, than divide the question into multiple questions and propose one query for each question.

Output format (must follow exactly):
ANSWER: <your answer or say "I don't know based on the context">
FOLLOW_UP_QUERY: <a single refined query or "none">
"""

    # Use accumulated messages as memory
    msgs: List[Dict[str, Any]] = list(state.get("messages", []))
    msgs.append({"role": "user", "content": prompt})
    resp = llm.invoke(msgs)
    raw = resp.content if hasattr(resp, "content") else str(resp)
    # Append assistant response to memory
    msgs.append({"role": "assistant", "content": raw})

    # Parse the two expected fields
    answer_text = raw
    follow_up = "none"
    for line in raw.splitlines():
        if line.strip().upper().startswith("FOLLOW_UP_QUERY:"):
            follow_up = line.split(":", 1)[1].strip() or "none"
            # Remove the FOLLOW_UP_QUERY line from the answer block
            answer_text = "\n".join(l for l in raw.splitlines() if l != line).strip()
            break

    # Strip ANSWER: header if present
    if answer_text.strip().upper().startswith("ANSWER:"):
        answer_text = answer_text.split(":", 1)[1].strip()

    # Prepare interaction logging
    def summarize_results(items: List[Dict[str, Any]]):
        summary = []
        for i, item in enumerate(items[:5]):
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            summary.append({"index": i + 1, "title": title, "url": url})
        return summary

    prior_attempts = int(state.get("attempts", 0))
    current_attempt_num = prior_attempts + 1
    interactions = list(state.get("interactions", []))
    interactions.append({
        "attempt": current_attempt_num,
        "query": state.get("query", ""),
        "results_summary": summarize_results(results),
        "prompt": prompt.strip(),
        "raw_reason": raw.strip(),
        "answer": answer_text,
        "follow_up_query": follow_up,
    })

    # Possibly schedule a retry
    attempts = prior_attempts
    max_attempts = int(state.get("max_attempts", 0))
    query_update = state.get("query", "")
    if (follow_up or "none").strip().lower() != "none" and attempts < max_attempts:
        attempts = attempts + 1
        query_update = follow_up

    return {
        **state,
        "answer": answer_text,
        "follow_up_query": follow_up,
        "last_prompt": prompt.strip(),
        "raw_reason": raw.strip(),
        "interactions": interactions,
        "attempts": attempts,
        "query": query_update,
        "messages": msgs,
    }

# ---------- Graph ----------
builder = StateGraph(GraphState)
builder.add_node("search", search_node)
builder.add_node("reason", reason_node)
builder.set_entry_point("search")
builder.add_edge("search", "reason")

def should_retry(state: GraphState) -> str:
    fu = (state.get("follow_up_query") or "none").strip().lower()
    attempts = int(state.get("attempts", 0))
    max_attempts = int(state.get("max_attempts", 0))
    return "retry" if fu != "none" and attempts < max_attempts else "end"

builder.add_conditional_edges("reason", should_retry, {"retry": "search", "end": END})

# Enable persistent memory via checkpointer
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


def run_multi_agent(question: str, max_attempts: int = 2, search_log_path: Optional[str] = None, thread_id: Optional[str] = None) -> GraphState:
    """Inter-agent loop: reasoner can request the retriever to try again with a refined query.

    Args:
        question: Original user question.
        max_attempts: Maximum number of follow-up retrieval attempts.

    Returns:
        Final GraphState including the best answer found and attempts used.
    """
    state: GraphState = {
        "question": question,
        "query": question,
        "results": [],
        "answer": "",
        "attempts": 0,
        "max_attempts": max_attempts,
        "follow_up_query": "none",
        "last_prompt": "",
        "raw_reason": "",
        "interactions": [],
        "search_payload": None,
        "messages": [
            {"role": "system", "content": "You are a careful researcher. Use only the provided context and cite sources like [1], [2]."}
        ],
        "search_log_path": search_log_path,
    }

    # Invoke the graph with a persistent thread id (for memory)
    config = {"configurable": {"thread_id": thread_id or "default-thread"}}
    final_state = graph.invoke(state, config=config)
    return final_state

if __name__ == "__main__":
    # Example question (change to whatever you want)
    question = "Who founded Safe Superintelligence Inc. and when was it announced?"
    # Use the inter-agent loop with up to 2 follow-up retrievals
    final = run_multi_agent(question, max_attempts=5)
    print("\n=== ANSWER ===\n")
    print(final["answer"])
