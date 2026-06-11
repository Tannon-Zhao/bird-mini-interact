"""
Native function-calling protocol template for the BIRD-Interact agent.

This is the SHARED spec used both by:
  1. the benchmark (mini_interact_agent) when --agent_chat_mode=native_fncall, and
  2. the SFT data-generation pipeline that trains the student LLM.

Keep training and evaluation in lockstep: the student is trained on transcripts
shaped exactly like what build/get_native_messages_* produce. The system prompt,
the <tool_call> response protocol, and the <tool_response> wrapper must
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
        """System message aligned with TemplateReActUserBirdInteract.get_init_msg(),
        only the response format differs (native <tool_call> instead of
        ReAct <thought>/<interaction_object>/<action>)."""
        return f"""You are a helpful {self.setting} agent that interacts with a user and a database to solve the user's question.

# Task Description
Your goal is to understand the user's ambiguous question involving the external knowledge retrieval and generate the correct SQL query to solve it. You can:
1. Interact with the user to ask clarifying questions to understand their request better or submit the SQL query to the user. The user will test your SQL correctness and give you feedback.
2. Interact with the {self.setting} environment ({self.language} db, column meaning file, external knowledge, and so on) to explore the database and get db relevant information.
- Termination condition: The interaction will end when you submit the correct SQL query or the user patience runs out.
- Cost of your action: each your action will cost a certain amount of user patience.

# Response Protocol (Function-Calling Format)
You call exactly one tool per turn. After each tool call you receive a tool response, then you act again. The cycle is: Tool Response -> Tool Call -> Tool Response -> Tool Call -> ...

## Response Format
Given previous interaction history and current tool response (or the user's request at the beginning), you should respond with exactly one tool call:

<tool_call>
{{"name": "<tool_name>", "arguments": {{...}}}}
</tool_call>

Do not emit more than one <tool_call> per turn. Do not write natural-language content outside of <tool_call>. Tool results come back as user messages wrapped in <tool_response>...</tool_response>.

## Tool Catalog and Action Costs
### Environment Tools
- **execute**: Execute a {self.language} query against the database.
    - arguments: {{"sql": string}} — {self.language} command to execute. Could contain multiple commands separated by semicolon. MUST BE IN ONE STRING.
    - output: fetched result from {self.language} database.
    - cost: 1 unit
- **get_schema**: Get the schema of the database.
    - arguments: {{}}
    - output: string of database schema in DDL format with demo data.
    - cost: 1 unit
- **get_all_column_meanings**: Get the meaning of all columns in the database.
    - arguments: {{}}
    - output: string of all column meanings.
    - cost: 1 unit
- **get_column_meaning**: Get the meaning of a specific column.
    - arguments: {{"table_name": string, "column_name": string}} — name of the table and column.
    - output: string of column meaning.
    - cost: 0.5 unit
- **get_all_external_knowledge_names**: Get all external knowledge names.
    - arguments: {{}}
    - output: list of string of external knowledge names.
    - cost: 0.5 unit
- **get_knowledge_definition**: Get external knowledge by name.
    - arguments: {{"knowledge_name": string}} — name of the external knowledge to get definition.
    - output: string of external knowledge definition.
    - cost: 0.5 unit
- **get_all_knowledge_definitions**: Get all external knowledge names with definitions.
    - arguments: {{}}
    - output: string of all external knowledge names with definitions.
    - cost: 1 unit

### User Tools
- **ask**: Ask user for clarification. If you find the user's question is ambiguous, you should ask user for clarification to figure out the user's real intent. TO REDUCE COST, YOU ARE ONLY ALLOWED TO ASK ONE QUESTION AT A TIME.
    - arguments: {{"question": string}} — question to ask user for clarification.
    - output: string of user's reply, to clarify the ambiguities in his/her question.
    - cost: 2 units
- **submit**: Submit the SQL to the user. The user will test the SQL and give feedback.
    - arguments: {{"sql": string}} — SQL to submit to the user. Could contain multiple commands separated by semicolon. MUST BE IN ONE STRING.
    - output: feedback from user about the submitted SQL.
    - cost: 3 units

After each tool call, you'll see a [SYSTEM NOTE] showing how much budget remains (e.g. "[SYSTEM NOTE: Remaining budget: 7.0/10.0]"). Pay close attention to this note as it indicates how many more interactions you can make. If budget runs out, the task ends and you'll need to submit your final answer.

# Important Strategy Tips
- First explore the database schema, column meaning and external knowledge to understand available tables, columns and user query's involved external knowledge.
- FIGURE OUT THE USER'S REAL INTENT BY ASKING CLARIFYING QUESTIONS! IF YOU CANNOT FIGURE OUT THE USER'S REAL INTENT, YOU WILL PRODUCE WRONG SQL AND CAUSE MILLION DOLLARS LOSS TO OUR COMPANY, THEN YOU WILL BE FIRED!!! (TO REDUCE COST OF USER PATIENCE, YOU ARE ONLY ALLOWED TO ASK ONE QUESTION AT A TIME.)
- FIGURE OUT THE USER'S REAL INTENT BY ASKING CLARIFYING QUESTIONS! IF YOU CANNOT FIGURE OUT THE USER'S REAL INTENT, YOU WILL PRODUCE WRONG SQL AND CAUSE MILLION DOLLARS LOSS TO OUR COMPANY, THEN YOU WILL BE FIRED!!! (TO REDUCE COST OF USER PATIENCE, YOU ARE ONLY ALLOWED TO ASK ONE QUESTION AT A TIME.)
- Be efficient with your actions to conserve user patience
- Make sure your submitted SQL is valid and addresses all aspects of the question
- Keep track of your remaining user patience and prioritize your actions accordingly
- Be careful with the action of frequently get all column meanings and external knowledge definitions. It will return a long context to you.
"""

    def get_demos_native(self) -> str:
        """One native-format toy example, rendered as a text block.

        Mirrors TemplateReActUserBirdInteract.get_demos() but in the native
        protocol (<tool_call>{json}</tool_call> for the agent and
        <tool_response> for results). Placed inside the FIRST user message
        (before the real query) so it carries no training loss, exactly like
        the legacy ReAct demo did. Converter and benchmark share this verbatim.

        The demo is expanded to match the non-native demo's coverage: same
        number of steps (schema -> column meaning -> knowledge names ->
        knowledge def -> ask -> ask -> execute -> submit) and same thought
        granularity.
        """
        return '''### A TOY EXAMPLE INTERACTION (native function-calling format) ###

User's Question: Find several calibrated alien signals detected by the observatory during the full moon phase. Show their information.

<tool_call>
{"name": "get_schema", "arguments": {}}
</tool_call>

<tool_response>
<name>get_schema</name>
<result>
Database schema showing tables: Observatories(ObservID, ObservStation, ObservDate, LunarStage, Signal), Signals(SignalID, SignalType, SignalStrength)...

[SYSTEM NOTE: Remaining budget: 21.0/22.0]
</result>
</tool_response>

<tool_call>
{"name": "get_column_meaning", "arguments": {"table_name": "Observatories", "column_name": "LunarStage"}}
</tool_call>

<tool_response>
<name>get_column_meaning</name>
<result>
Full name: 'Lunar Stage'. Explanation: Current lunar phase during observation. Values include: 'New', 'Waxing Crescent', 'First Quarter', 'Waxing Gibbous', 'Full', 'Waning Gibbous', 'Last Quarter', 'Waning Crescent'.

[SYSTEM NOTE: Remaining budget: 20.5/22.0]
</result>
</tool_response>

<tool_call>
{"name": "get_all_external_knowledge_names", "arguments": {}}
</tool_call>

<tool_response>
<name>get_all_external_knowledge_names</name>
<result>
["Alien Signal Classification", "Lunar Phase Effects", "Calibrated Signal"]

[SYSTEM NOTE: Remaining budget: 20.0/22.0]
</result>
</tool_response>

<tool_call>
{"name": "get_knowledge_definition", "arguments": {"knowledge_name": "Calibrated Signal"}}
</tool_call>

<tool_response>
<name>get_knowledge_definition</name>
<result>
{"id": 3, "knowledge": "Calibrated Signal", "description": "Definition of calibrated signal", "definition": "Those signal with strength > 10."}

[SYSTEM NOTE: Remaining budget: 19.5/22.0]
</result>
</tool_response>

<tool_call>
{"name": "ask", "arguments": {"question": "How many results do you want? What do you mean about \\"several\\"?"}}
</tool_call>

<tool_response>
<name>ask</name>
<result>
I want three results sorted by signal strength.

[SYSTEM NOTE: Remaining budget: 17.5/22.0]
</result>
</tool_response>

<tool_call>
{"name": "ask", "arguments": {"question": "Do you want to get the signal name and signal strength?"}}
</tool_call>

<tool_response>
<name>ask</name>
<result>
I just want the signal id.

[SYSTEM NOTE: Remaining budget: 15.5/22.0]
</result>
</tool_response>

<tool_call>
{"name": "execute", "arguments": {"sql": "SELECT s.SignalID FROM Observatories o JOIN Signals s ON o.Signal = s.SignalID WHERE o.LunarStage = 'Full' AND s.SignalStrength > 10 ORDER BY s.SignalStrength DESC LIMIT 3"}}
</tool_call>

<tool_response>
<name>execute</name>
<result>
[(12,), (11,), (10,)]

[SYSTEM NOTE: Remaining budget: 14.5/22.0]
</result>
</tool_response>

<tool_call>
{"name": "submit", "arguments": {"sql": "SELECT s.SignalID FROM Observatories o JOIN Signals s ON o.Signal = s.SignalID WHERE o.LunarStage = 'Full' AND s.SignalStrength > 10 ORDER BY s.SignalStrength DESC LIMIT 3"}}
</tool_call>

<tool_response>
<name>submit</name>
<result>
Your SQL is correct! Now, here's a follow-up question from user: Can you also count how many distinct calibrated signals were detected during full moon phases?

[SYSTEM NOTE: Remaining budget: 11.5/22.0]
</result>
</tool_response>
...

### END OF TOY EXAMPLE INTERACTION ###'''

    def get_query_msg(self, query: str, budget_info: dict) -> str:
        """First user message: the (ambiguous) user question + budget hint.
        Aligned with TemplateReActUserBirdInteract.get_query_msg() format."""
        total = budget_info.get("total_budget", 0.0)
        return (
            f"# -----TASK START-----\n"
            f"Now, let's start with the user's question that may exist ambiguities "
            f"and require external knowledge understanding to solve. "
            f"(EACH TIME GIVE ONE ROUND RESPONSE, END YOUR RESPONSE AT ... "
            f"'</tool_call>' OTHERWISE YOU WILL BE FIRED!!!) \n\n"
            f"User's Question: {query}\n:\n\n"
            f"[SYSTEM NOTE: You have a total action budget of {total:.1f} units. "
            f"Each action consumes budget. If the budget runs out, you must submit.]"
        )


if __name__ == "__main__":
    t = TemplateNativeFnCallBirdInteract("SQLITE", "SQLite Database")
    print(t.get_native_init_msg())
    print("\n----- demos -----\n")
    print(t.get_demos_native())
    print("\n----- query msg -----\n")
    print(t.get_query_msg("Find several calibrated alien signals.", {"total_budget": 22.0}))
