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

    def get_demos_native(self) -> str:
        """One native-format toy example, rendered as a text block.

        Mirrors TemplateReActUserBirdInteract.get_demos() but in the native
        protocol (<think>/<tool_call>{json}</tool_call> for the agent and
        <tool_response> for results). Placed inside the FIRST user message
        (before the real query) so it carries no training loss, exactly like
        the legacy ReAct demo did. Converter and benchmark share this verbatim.
        """
        return """### A TOY EXAMPLE INTERACTION (native function-calling format) ###

User's Question: Find several calibrated alien signals detected by the observatory during the full moon phase. Show their information.

<think>
This query is ambiguous. I should check the schema, learn how "full moon" is encoded, look up what "calibrated" means, then clarify "several" and which fields to return. Start with the schema.
</think>
<tool_call>
{"name": "get_schema", "arguments": {}}
</tool_call>

<tool_response>
<name>get_schema</name>
<id>chatcmpltooldemo000001</id>
<result>
Database schema showing tables: Observatories(ObservID, ObservStation, ObservDate, LunarStage, Signal), Signals(SignalID, SignalType, SignalStrength)...

[SYSTEM NOTE: Remaining budget: 21.0/22.0]
</result>
</tool_response>

<think>
There's a LunarStage field. I need to know what values represent the full moon.
</think>
<tool_call>
{"name": "get_column_meaning", "arguments": {"table_name": "Observatories", "column_name": "LunarStage"}}
</tool_call>

<tool_response>
<name>get_column_meaning</name>
<id>chatcmpltooldemo000002</id>
<result>
Full name: 'Lunar Stage'. Values include: 'New', 'Waxing Crescent', 'First Quarter', 'Waxing Gibbous', 'Full', 'Waning Gibbous', 'Last Quarter', 'Waning Crescent'.

[SYSTEM NOTE: Remaining budget: 20.5/22.0]
</result>
</tool_response>

<think>
"Full" is the full moon. Now I should understand "calibrated alien signals" via external knowledge.
</think>
<tool_call>
{"name": "get_knowledge_definition", "arguments": {"knowledge_name": "Calibrated Signal"}}
</tool_call>

<tool_response>
<name>get_knowledge_definition</name>
<id>chatcmpltooldemo000003</id>
<result>
{"knowledge": "Calibrated Signal", "definition": "Those signals with strength > 10."}

[SYSTEM NOTE: Remaining budget: 20.0/22.0]
</result>
</tool_response>

<think>
"Calibrated" means strength > 10. But "several" and the output fields are unclear. Ask one clarifying question.
</think>
<tool_call>
{"name": "ask", "arguments": {"question": "How many results do you want, and what information should I show for each signal?"}}
</tool_call>

<tool_response>
<name>ask</name>
<id>chatcmpltooldemo000004</id>
<result>
I want the top 3 results sorted by signal strength, and I just want the signal id.

[SYSTEM NOTE: Remaining budget: 18.0/22.0]
</result>
</tool_response>

<think>
Now I have everything: Full moon, strength > 10, top 3 by strength, return SignalID. Write and submit the SQL.
</think>
<tool_call>
{"name": "submit", "arguments": {"sql": "SELECT s.SignalID FROM Observatories o JOIN Signals s ON o.Signal = s.SignalID WHERE o.LunarStage = 'Full' AND s.SignalStrength > 10 ORDER BY s.SignalStrength DESC LIMIT 3"}}
</tool_call>

### END OF TOY EXAMPLE ###"""

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
