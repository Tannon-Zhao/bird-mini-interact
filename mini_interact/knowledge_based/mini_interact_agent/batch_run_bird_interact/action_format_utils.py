"""Shared helpers for agent action validation and observation size control."""

from typing import Optional

UNKNOWN_ACTION_TRUNCATE_LEN = 500
ACTION_LOG_TRUNCATE_LEN = 500

VALID_ACTION_PREFIXES = (
    "execute(",
    "get_schema(",
    "get_all_column_meanings(",
    "get_column_meaning(",
    "get_all_external_knowledge_names(",
    "get_knowledge_definition(",
    "get_all_knowledge_definitions(",
    "ask(",
    "submit(",
)

_UNKNOWN_ENV_SUFFIX = (
    " Your availabel actions to Database are "
    "execute(sql): Execute a SQL query "
    "get_schema(): Get the schema of the database "
    "get_all_column_meanings(): Get all column meanings "
    "get_all_external_knowledge_names(): Get all external knowledge names "
    "get_knowledge_definition(knowledge_name): Get the definition of a specific knowledge "
    "get_all_knowledge_definitions(): Get all knowledge definitions"
)


def truncate_for_observation(text: str, max_len: int = UNKNOWN_ACTION_TRUNCATE_LEN) -> str:
    if not text:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[:max_len] + "..."


def truncate_action_for_log(action: str, max_len: int = ACTION_LOG_TRUNCATE_LEN) -> str:
    return truncate_for_observation(action, max_len=max_len)


def format_unknown_environment_observation(action: str) -> str:
    preview = truncate_for_observation(action)
    return f"Unknown Environment action (truncated): {preview}{_UNKNOWN_ENV_SUFFIX}"


def is_valid_parsed_action(action: str) -> bool:
    if not action or not action.strip():
        return False
    normalized = action.strip()
    if any(tag in normalized for tag in ("<thought>", "</thought>", "<action>", "</action>")):
        return False
    if normalized.startswith("Error:"):
        return False
    if "BadRequestError" in normalized or "maximum context length" in normalized.lower():
        return False
    return any(normalized.startswith(prefix) for prefix in VALID_ACTION_PREFIXES)


def classify_agent_api_failure(response: str) -> Optional[str]:
    """Return a short observation if the agent response is a worker/API failure."""
    if not response or not str(response).strip():
        return "[API error: empty response from agent model]"

    text = str(response).strip()
    lower = text.lower()

    if text.startswith("Error: Exception during processing for index"):
        if "maximum context length" in lower or "context length" in lower:
            return "[API error: context length exceeded]"
        if "badrequesterror" in lower or "error code: 400" in lower:
            return "[API error: request rejected by model API]"
        return "[API error: agent model request failed]"

    if text.startswith("Error: Processing failed unexpectedly for index"):
        return "[API error: agent model request failed]"

    if text.startswith("Error: Unexpected response format from call_api_model"):
        return "[API error: invalid response from agent model]"

    if "model response blocked" in lower:
        return "[API error: model response blocked]"

    if "maximum context length is" in lower and "badrequesterror" in lower:
        return "[API error: context length exceeded]"

    return None
