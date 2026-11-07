"""CrewAI orchestration for a multi-source Agentic RAG demonstration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

Route = Literal["pdf", "web", "generate"]


@dataclass(frozen=True)
class RouteDecision:
    """A traceable routing result."""

    route: Route
    rationale: str


def route_question(question: str) -> RouteDecision:
    """Return a deterministic route for the key-free demonstration.

    The live CrewAI router can handle broader language. These rules make the
    notebook reproducible and provide a testable fallback.
    """

    text = question.lower().strip()
    fresh_markers = (
        "latest",
        "current",
        "today",
        "recent",
        "this year",
        "news",
        "state of the art",
    )
    pdf_markers = (
        "paper",
        "pdf",
        "transformer",
        "attention",
        "encoder",
        "decoder",
        "multi-head",
        "positional encoding",
    )
    if any(marker in text for marker in fresh_markers):
        return RouteDecision("web", "The question asks for time-sensitive information.")
    if any(marker in text for marker in pdf_markers):
        return RouteDecision("pdf", "The question targets concepts in the domain PDF.")
    return RouteDecision("generate", "The question does not require a provided document or fresh web data.")


class TraceLogger:
    """Append machine-readable interaction events to JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


class AgenticRAGSystem:
    """Configure the router/retriever collaboration and required tools."""

    def __init__(self, pdf_path: Path, trace_path: Path = Path("logs/trace.jsonl")) -> None:
        self.pdf_path = Path(pdf_path)
        self.trace = TraceLogger(Path(trace_path))

    def route(self, question: str) -> RouteDecision:
        decision = route_question(question)
        self.trace.write("router_decision", question=question, **asdict(decision))
        return decision

    def build_crew(self, question: str):
        """Build a live sequential CrewAI workflow for *question*.

        Imports are intentionally local so the deterministic demo and unit tests
        run without optional API packages or credentials.
        """

        try:
            from crewai import Agent, Crew, Process, Task
            from crewai_tools import PDFSearchTool
            from langchain_community.tools.tavily_search import TavilySearchResults
        except ImportError as exc:
            raise RuntimeError("Install requirements.txt to run the live crew.") from exc

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for the live CrewAI path.")

        decision = self.route(question)
        tools = []
        if decision.route == "pdf":
            if not self.pdf_path.exists():
                raise FileNotFoundError(f"Place the domain PDF at {self.pdf_path}")
            tools = [PDFSearchTool(pdf=str(self.pdf_path))]
        elif decision.route == "web":
            if not os.getenv("TAVILY_API_KEY"):
                raise RuntimeError("TAVILY_API_KEY is required for web retrieval.")
            tools = [TavilySearchResults(max_results=5)]

        router = Agent(
            role="Router Agent",
            goal="Select the correct evidence path and explain the handoff.",
            backstory="You triage questions for a multi-source research assistant.",
            verbose=True,
            allow_delegation=False,
        )
        retriever = Agent(
            role="Retriever Agent",
            goal="Return a concise answer grounded only in the selected evidence.",
            backstory="You retrieve evidence, preserve source details, and disclose gaps.",
            tools=tools,
            verbose=True,
            allow_delegation=False,
        )
        routing_task = Task(
            description=(
                f"Question: {question}\n"
                f"Preliminary route: {decision.route}. Explain why this route fits and "
                "give the retriever a precise retrieval instruction."
            ),
            expected_output="A route, short rationale, and retrieval instruction.",
            agent=router,
        )
        retrieval_task = Task(
            description=(
                f"Answer this question: {question}\nUse the selected route ({decision.route}). "
                "Use the assigned tool when one is present. Ground each substantive claim "
                "in retrieved evidence, identify sources, and state when evidence is insufficient."
            ),
            expected_output="A grounded answer followed by a Sources section.",
            agent=retriever,
            context=[routing_task],
        )
        self.trace.write(
            "crew_built",
            question=question,
            route=decision.route,
            tool_count=len(tools),
        )
        return Crew(
            agents=[router, retriever],
            tasks=[routing_task, retrieval_task],
            process=Process.sequential,
            verbose=True,
        )


def offline_answer(question: str) -> dict[str, Any]:
    """Return a small, deterministic demonstration result."""

    decision = route_question(question)
    if decision.route == "pdf":
        answer = (
            "The Transformer replaces recurrence with attention. Multi-head attention "
            "projects queries, keys, and values into several learned subspaces, performs "
            "scaled dot-product attention in parallel, then concatenates the results."
        )
        sources = ["Vaswani et al., Attention Is All You Need, Sections 3.2–3.3"]
    elif decision.route == "web":
        answer = (
            "This route requires a live search because the answer can change. In offline "
            "mode the system records the intended Tavily query rather than inventing results."
        )
        sources = ["Tavily live-search placeholder"]
    else:
        answer = "The system can answer this stable, general question directly with the configured LLM."
        sources = ["Direct generation path"]
    return {"question": question, "route": decision.route, "rationale": decision.rationale, "answer": answer, "sources": sources}
