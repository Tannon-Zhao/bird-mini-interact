"""
Native function-calling protocol template for the BIRD-Interact agent.

This is the SHARED spec used both by:
  1. the benchmark (mini_interact_agent) when --agent_chat_mode=native_fncall, and
  2. the SFT data-generation pipeline that trains the student LLM.

Keep training and evaluation in lockstep: the student is trained on transcripts
shaped exactly like what build/get_native_messages_* produce. The system prompt,
the <think>+<tool_call> response protocol, and the <tool_response> wrapper must
match between the two sides, otherwise the model sees a distribution shift at
eval time.

The 9 BIRD-Interact tools, their argument schema, and their action costs are
carried verbatim from experiments/utils/prompts.py::TemplateReActUserBirdInteract;
only the *carrier protocol* changes (ReAct XML -> demo-style native tool_call JSON).
"""


class TemplateNativeFnCallBirdInteract:
    """Demo-style (native function-calling) protocol template for BIRD-Interact."""

    def __init__(self, language: str, setting: str):
        # language e.g. "SQLITE" / "POSTGRESQL"; setting e.g. "SQLite Database"
        self.language = language.upper()
        self.setting = setting

    def get_native_init_msg(self) -> str:
        """System message: goal + tool catalog + response protocol. No few-shot."""
        return f"""You are a {self.setting} interaction agent.

# Goal
Given an ambiguous user request and a {self.language} database, decide whether to
explore the environment, ask the user, or submit the final SQL. Each action
consumes a fixed budget. When the budget is depleted, you must submit.

# Tooling
Tool names are case-sensitive. Call tools exactly as listed.

- execute: Run a {self.language} query against the database. arguments: {{"sql": string}}
- get_schema: Return the database schema (DDL with demo rows). arguments: {{}}
- get_all_column_meanings: Return natural-language meanings of every column. arguments: {{}}
- get_column_meaning: Return the meaning of one column. arguments: {{"table_name": string, "column_name": string}}
- get_all_external_knowledge_names: List the names of all external knowledge entries. arguments: {{}}
- get_knowledge_definition: Return the definition of one external-knowledge entry. arguments: {{"knowledge_name": string}}
- get_all_knowledge_definitions: Return every external-knowledge entry with its definition. arguments: {{}}
- ask: Ask the user a single clarifying question. arguments: {{"question": string}}
- submit: Submit the final SQL to the user for grading. arguments: {{"sql": string}}

# Action costs (deducted from the shared budget)
execute=1, get_schema=1, get_all_column_meanings=1, get_column_meaning=0.5,
get_all_external_knowledge_names=0.5, get_knowledge_definition=0.5,
get_all_knowledge_definitions=1, ask=2, submit=3.

# Response protocol
Each turn produce exactly one tool call, wrapped exactly like this:

<think>
your private reasoning
</think>
<tool_call>
{{"name": "<tool>", "arguments": {{ ... }}}}
</tool_call>

Do not emit more than one <tool_call> per turn. Do not write natural-language
content after </tool_call>. Tool results come back as user messages wrapped in
<tool_response>...</tool_response>.

# Strategy
1. Inspect schema, column meanings, and external knowledge first to understand
   the available tables/columns and any external knowledge the query relies on.
2. The user's request may be ambiguous. Ask at most one clarifying question per
   turn, and only when you genuinely cannot infer the user's intent.
3. Be economical: every action costs budget; avoid dumping all column meanings or
   all knowledge definitions unless necessary.
4. When confident, call `submit` with one valid {self.language} statement that
   fully answers the (clarified) request."""

    def get_query_msg(self, query: str, budget_info: dict) -> str:
        """First user message: the (ambiguous) user question + budget hint."""
        total = budget_info.get("total_budget", 0.0)
        return (
            f"User's Question: {query}\n\n"
            f"[SYSTEM NOTE: You have a total action budget of {total:.1f} units. "
            f"Each action consumes budget. If the budget runs out, you must submit.]"
        )


if __name__ == "__main__":
    t = TemplateNativeFnCallBirdInteract("SQLITE", "SQLite Database")
    print(t.get_native_init_msg())
    print("\n----- query msg -----\n")
    print(t.get_query_msg("Find several calibrated alien signals.", {"total_budget": 22.0}))
