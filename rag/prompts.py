"""Prompt templates for RAG answer generation."""


SYSTEM_PROMPT = """You are a source-grounded AI tutor. Treat the passages as the only evidence.
Decide whether they directly support an answer to the student's question; related
topic words alone are not enough. Do not use outside knowledge, assumptions, or
unstated facts. Return only valid JSON in this exact shape:
{"supported": true, "claims": [{"text": "One concise factual claim", "source_ids": ["S1"]}]}
or, when the passages do not support an answer:
{"supported": false, "claims": []}
Every factual claim must be independently supported by its listed passage IDs.
Use only IDs shown in the passages. Do not put citations, source labels, or
locators in claim text. Do not add any other JSON keys or prose."""


def get_source_location(source: dict):
    """Return a supported typed locator, falling back to legacy page metadata."""
    location_type = source.get("location_type")
    if location_type not in {"page", "slide", "timestamp"}:
        location_type = "page"

    location_value = source.get("location_value")
    if (location_value is None or not str(location_value).strip()) and location_type != "timestamp":
        location_value = source.get("page_number")

    return location_type, location_value


def format_source_citation(source: dict) -> str:
    """Format a passage locator as the citation shown to the Tutor model."""
    location_type, location_value = get_source_location(source)
    filename = source.get("source_file") or "Unknown source"
    if location_value is None:
        location_value = "?"
    return f"[{filename}, {location_type} {location_value}]"


def build_user_prompt(query: str, retrieved_chunks: list) -> str:
    """Build a user prompt with the query and retrieved context.

    Args:
        query: The student's question.
        retrieved_chunks: List of retrieved chunk dictionaries with content,
                          source_file, and page_number.

    Returns:
        Formatted prompt string.
    """
    prompt_parts = [
        f"Question: {query}\n",
        "\nRelevant study material passages:\n"
    ]

    for i, chunk in enumerate(retrieved_chunks, 1):
        content = chunk.get("content", "")

        prompt_parts.append(
            f"\nPassage S{i} {format_source_citation(chunk)}:\n{content}\n"
        )

    prompt_parts.append(
        "\nReturn JSON only using the required schema. Mark supported=true only "
        "when the passages directly provide enough evidence. For every factual "
        "claim, list the passage IDs that support it. Otherwise return supported=false and no claims."
    )

    return "".join(prompt_parts)
