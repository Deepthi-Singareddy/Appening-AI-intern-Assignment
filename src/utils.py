import os
import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from typing import List, Dict, Any, Tuple
from src.config import settings

# Configure Industrial Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("rag_agentic_ai")


def compute_groundedness_score(query: str, context: List[str], answer: str) -> float:
    """
    Computes an empirical confidence / groundedness score [0.0 - 1.0].
    Evaluates:
    1. Availability of retrieved context.
    2. Overlap between answer terms and context terms.
    3. Explicit refusal detection (out-of-scope query rejection).
    """
    if not context or not answer:
        return 0.0

    refusal_phrases = [
        "cannot answer",
        "not available in the ebook",
        "does not contain",
        "lacks this information",
        "out of scope",
        "not mentioned",
        "i don't know",
        "no information"
    ]
    answer_lower = answer.lower()
    for phrase in refusal_phrases:
        if phrase in answer_lower:
            return 0.0

    # Calculate token overlap between answer and context
    context_text = " ".join(context).lower()
    context_words = set(w.strip(".,!?()[]\"'") for w in context_text.split() if len(w) > 3)
    answer_words = [w.strip(".,!?()[]\"'") for w in answer_lower.split() if len(w) > 3]

    if not answer_words:
        return 0.5

    matching_words = sum(1 for w in answer_words if w in context_words)
    overlap_ratio = matching_words / len(answer_words)

    # Base confidence calculation
    score = 0.5 + (0.45 * min(1.0, overlap_ratio * 1.2))
    return round(score, 2)


def generate_agentic_ai_pdf(output_path: Path):
    """
    Generates a structured, multi-page PDF document 'Ebook-Agentic-AI.pdf' 
    containing comprehensive Agentic AI ebook content for knowledge base ingestion.
    Uses basic PyPDF / raw PDF generation to guarantee zero extra binary tool dependencies.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sections = [
        ("CHAPTER 1: Definition and Core Scope of Agentic AI", """
Agentic AI refers to autonomous artificial intelligence systems capable of perceiving their environment, reasoning about complex objectives, making decisions, decomposing tasks, and executing multi-step workflows with minimal human intervention.

Unlike traditional generative AI chatbots that act as simple input-output completion engines, Agentic AI exhibits goal-directed behavior, dynamic tool utilization, long-term memory retrieval, and iterative self-correction. 

Core Characteristics of Agentic AI:
1. Autonomy: Operates independently to achieve open-ended goals.
2. Multi-step Planning: Decomposes high-level objectives into granular subtasks.
3. Tool Usage: Interfaces with APIs, databases, web search engines, and code execution environments.
4. Stateful Execution: Retains persistent context across complex multi-turn workflows.
5. Self-Reflection: Evaluates its own intermediate outputs and adjusts strategy upon failure.
"""),

        ("CHAPTER 2: Architectural Components and Paradigms", """
Building an Agentic AI architecture requires four primary components working in concert:

1. Brain / Reasoning Engine:
Powered by Large Language Models (LLMs) such as GPT-4o, Claude, or Llama 3. Responsible for Chain-of-Thought (CoT) reasoning, subtask planning, and decision-making.

2. Memory Subsystem:
- Short-Term Memory: Maintains active conversation context within the LLM window.
- Long-Term Memory: Retains episodic, semantic, and procedural knowledge over time using vector databases like Pinecone.

3. Tool & Action Suite:
Functions and external software APIs enabled for the agent (e.g., SQL execution, vector search, web scraping, python interpreters).

4. Environment & Observer Loop:
Perceives external state updates, logs execution results, and feeds observations back into the agent's reasoning loop for continuous control.
"""),

        ("CHAPTER 3: Industry Use Cases for Agentic AI", """
Real-World Enterprise Applications:

1. Autonomous Software Engineering:
Agents inspect repositories, identify bugs, generate tests, and issue pull requests (e.g., Devin, Antigravity AI, AutoGPT).

2. Enterprise Customer Operations:
Agents resolve complex multi-tiered customer requests by querying CRM databases, issuing refunds, and triggering logistical workflows.

3. Financial Analysis & Portfolio Management:
Agentic workflows parse thousands of SEC filings, perform quantitative risk modeling, and synthesize investment memos autonomously.

4. Healthcare & Clinical Knowledge Management:
Agents query clinical literature, cross-reference patient histories against medical knowledge bases, and assist physicians with differential diagnosis.

5. Supply Chain & Logistics Orchestration:
Agents monitor real-time shipment disruptions, re-route freight automatically, and re-negotiate vendor schedules.
"""),

        ("CHAPTER 4: Agentic AI vs Traditional Generative AI Chatbots", """
Comparison Matrix:

Feature | Traditional Generative AI Chatbot | Agentic AI Systems
-------------------------------------------------------------------
Interaction Pattern | Single-turn / Conversational QA | Multi-step Goal-Directed Execution
Statefulness | Stateless / Window-limited | Persistent Memory & State Graphs
Task Execution | Output Text Generation | API Execution, Database Writes, Tool Actions
Error Recovery | Halts or returns generic response | Self-grading, Reflection, and Retry
Planning Capability | None (Direct generation) | Subtask Decomposition & Dynamic Re-planning
Knowledge Source | Static pre-trained weights | Real-time RAG & Dynamic Vector Search
"""),

        ("CHAPTER 5: Challenges, Limitations, and Guardrails", """
Key Implementation Challenges:

1. Hallucination & Grounding Drift:
Agents may hallucinate facts or construct incorrect intermediate plans. Mitigation requires strict Retrieval-Augmented Generation (RAG) with groundedness evaluation nodes.

2. Infinite Loop Vulnerabilities:
Agents can become stuck in circular reasoning or repeated API calls. Mitigation requires loop-counting state graphs with deterministic execution limits.

3. Security & Prompt Injection:
Malicious inputs can trick agents into executing unauthorized tool calls. System prompts must enforce strict access boundaries.

4. Token Costs & Latency:
Iterative agentic loops increase LLM token consumption and call latency. Requires caching and lightweight model delegation (e.g., gpt-4o-mini).
""")
    ]

    # Simple plain-text formatted PDF generation helper using pure Python if reportlab unavailable
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=12
        )
        body_style = ParagraphStyle(
            'BodyStyle',
            parent=styles['BodyText'],
            fontSize=10,
            leading=14,
            spaceAfter=10
        )

        for heading, body in sections:
            story.append(Paragraph(f"<b>{heading}</b>", title_style))
            for para in body.strip().split("\n\n"):
                clean_text = para.replace("\n", "<br/>")
                story.append(Paragraph(clean_text, body_style))
            story.append(Spacer(1, 12))

        doc.build(story)
        logger.info(f"Successfully generated structured PDF document at: {output_path}")

    except ImportError:
        # Fallback raw PDF generator using reportlab-less FPDF approach or text file structure
        logger.info("ReportLab not present. Writing document content for parsing fallback.")
        text_content = "\n\n".join(f"{h}\n{'='*len(h)}\n{b}" for h, b in sections)
        # Create text backup
        txt_path = output_path.with_suffix(".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_content)
        
        # Write minimal PDF structure
        _write_minimal_pdf(output_path, sections)


def _write_minimal_pdf(output_path: Path, sections: List[Tuple[str, str]]):
    """Fallback simple PDF binary generator without external dependencies."""
    full_text = "Agentic AI eBook - Knowledge Base Document\n\n"
    for heading, body in sections:
        full_text += f"{heading}\n{body}\n\n"

    # Escape parentheses
    escaped_text = full_text.replace("(", "\\(").replace(")", "\\)")
    lines = escaped_text.split("\n")
    
    # Construct raw PDF pages
    pdf_content = (
        "%PDF-1.4\n"
        "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        "4 0 obj\n<< /Length 5000 >>\nstream\nBT\n/F1 10 Tf\n50 740 Td\n12 TL\n"
    )
    
    for line in lines[:50]:  # Basic page wrapping
        pdf_content += f"({line[:90]}) '\n"
        
    pdf_content += (
        "ET\nendstream\nendobj\n"
        "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        "xref\n0 6\n0000000000 65535 f \n"
        "trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%%EOF\n"
    )

    with open(output_path, "wb") as f:
        f.write(pdf_content.encode("latin-1", errors="ignore"))
    logger.info(f"Fallback PDF generated at: {output_path}")


def ensure_knowledge_base_pdf() -> Path:
    """Ensures Ebook-Agentic-AI.pdf exists in the data directory."""
    pdf_path = settings.DEFAULT_PDF_PATH
    if not pdf_path.exists():
        logger.info(f"Knowledge Base PDF not found at {pdf_path}. Generating default eBook...")
        generate_agentic_ai_pdf(pdf_path)
    return pdf_path
