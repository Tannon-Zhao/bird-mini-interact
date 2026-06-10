import sys
import os
import re
import json
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from batch_run_bird_interact.sample_status import SampleStatus

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import necessary prompt templates and parsers
from experiments.utils.prompts import TemplateReActUserBirdInteract
from src.envs.user_simulator.prompts import USER_SIMULATOR_ENCODER, USER_SIMULATOR_DECODER
from src.envs.user_simulator.sql_parser import segment_sql # Assuming this can be imported

# Constants
SETTING_MAP = {
    "bird_interact_sql": "PostgreSQL Database",
    "mini_interact": "SQLite Database",
}

LANGUAGE_MAP = {
    "bird_interact_sql": "PostgreSQL",
    "mini_interact": "SQLite",
}

# Cache for segmented SQL
_segment_sql_cache: Dict[str, Any] = {}

# --- Agent Prompt --- #

def build_initial_agent_prompt(sample_status: 'SampleStatus', budget_info: Dict, env_type: str = "bird_interact_sql") -> str:
    # Initialize template (assuming 'bird_interact_sql' is the env type)
    # This might need to be initialized differently if handling multiple env types
    react_template = TemplateReActUserBirdInteract(LANGUAGE_MAP[env_type], SETTING_MAP[env_type])

    """Builds the initial prompt for the agent for turn 0."""
    system_prompt = react_template.get_init_msg()
    demos = react_template.get_demos()
    query = sample_status.original_data["amb_user_query"]
    query_msg = react_template.get_query_msg(query)

    budget_prompt = f"\n\n[SYSTEM NOTE: You have a total action budget of {budget_info['total_budget']:.1f} units. Each action consumes budget. If the budget runs out, you must submit.]"

    initial_prompt = system_prompt + demos + query_msg + budget_prompt
    sample_status.current_prompt = initial_prompt # Store the base prompt
    return initial_prompt

def get_agent_prompt_for_turn(sample_status: 'SampleStatus') -> str:
    """Constructs the full agent prompt for the current turn based on history."""
    # The base prompt (query + budget info) should be in sample_status.current_prompt
    # The interaction history is used to build the rest
    return sample_status.get_full_interaction_prompt()


def build_initial_agent_prompt_dispatch(sample_status: 'SampleStatus', budget_info: Dict,
                                        env_type: str = "bird_interact_sql",
                                        chat_mode: str = "react_text"):
    """Dispatch initial-prompt construction by agent chat mode.

    react_text  -> legacy single-string ReAct prompt (stored in current_prompt)
    native_fncall -> demo-style [system, user] messages (stored in native_base_messages)
    Returns whatever the chosen builder returns (str or messages list).
    """
    if chat_mode == "native_fncall":
        from batch_run_bird_interact.native_messages_builder import build_native_initial_messages
        msgs = build_native_initial_messages(sample_status, budget_info, env_type)
        sample_status.current_prompt = ""  # not used in native mode
        return msgs
    return build_initial_agent_prompt(sample_status, budget_info, env_type)

def _convert_tool_call_to_action(name: str, arguments: dict) -> Tuple[str, str]:
    """
    Convert a tool_call JSON (name + arguments) into (interaction_object, action_string).
    Handles both direct format {"name":"get_schema",...} and nested format
    {"name":"Environment","arguments":{"action":"get_schema"}}.
    """
    if name in ("Environment", "User"):
        interaction_object = name
        inner_action = arguments.get("action", "")
        if inner_action:
            inner_args = {k: v for k, v in arguments.items() if k != "action"}
            return interaction_object, _format_action_call(inner_action, inner_args)
        if "question" in arguments:
            return "User", f"ask('{arguments['question']}')"
        if "sql" in arguments:
            return "User", f'submit("{arguments["sql"]}")'
        return interaction_object, "get_schema()"

    if name in ("ask", "submit"):
        interaction_object = "User"
    else:
        interaction_object = "Environment"
    return interaction_object, _format_action_call(name, arguments)


def _format_action_call(name: str, arguments: dict) -> str:
    """Format a function name + arguments dict into a callable action string."""
    if name == "get_schema":
        return "get_schema()"
    elif name == "execute":
        sql = arguments.get("sql", "")
        return f'execute("{sql}")'
    elif name == "ask":
        question = arguments.get("question", "")
        return f"ask('{question}')"
    elif name == "submit":
        sql = arguments.get("sql", "")
        return f'submit("{sql}")'
    elif name == "get_all_column_meanings":
        return "get_all_column_meanings()"
    elif name == "get_column_meaning":
        table = arguments.get("table_name", "")
        column = arguments.get("column_name", "")
        return f"get_column_meaning('{table}', '{column}')"
    elif name == "get_all_external_knowledge_names":
        return "get_all_external_knowledge_names()"
    elif name == "get_knowledge_definition":
        kname = arguments.get("knowledge_name", "")
        return f"get_knowledge_definition('{kname}')"
    elif name == "get_all_knowledge_definitions":
        return "get_all_knowledge_definitions()"
    else:
        args_str = ", ".join(f"'{v}'" for v in arguments.values()) if arguments else ""
        return f"{name}({args_str})"


def _extract_json_from_text(text: str) -> Optional[dict]:
    """
    Extract the first valid JSON object containing a "name" key from text.
    Handles nested braces (e.g. "arguments": {}).
    """
    start = text.find('{"name"')
    if start == -1:
        start = text.find("{'name")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


# Known BIRD-Interact tool names (used by the native tool_call fallback parser).
_KNOWN_TOOLS = (
    "execute",
    "get_schema",
    "get_all_column_meanings",
    "get_column_meaning",
    "get_all_external_knowledge_names",
    "get_knowledge_definition",
    "get_all_knowledge_definitions",
    "ask",
    "submit",
)


def _extract_bare_json_dict(text: str) -> Optional[dict]:
    """Return the first balanced {...} that parses to a dict (any keys)."""
    start = text.find('{')
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
        start = text.find('{', start + 1)
    return None


def _try_parse_native_toolcall_fallback(response: str) -> Optional[Tuple[str, str, str, dict]]:
    """
    Tolerant fallback for provider-native tool-call styles that are NOT our
    {"name":..,"arguments":..} JSON. Notably GLM emits shapes like:
        <tool_call>get_schema
        <tool_call>execute</arg_value>arguments</arg_key><arg_value>{"sql":"..."}</arg_value>
        <tool_call>get_column_meaning\ntable_name: signals\ncolumn_name: interflvl
        <tool_call>execute("SELECT ...")
    Returns (thought, interaction_object, action, tool_data) or None.
    """
    if "<tool_call>" not in response:
        return None

    seg = response.split("<tool_call>", 1)[1]
    seg = seg.split("</tool_call>", 1)[0]  # cut trailing close tag if present

    thought = ""
    m = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if m:
        thought = m.group(1).strip()

    # Resolve the tool name: prefer the leading identifier, else first known tool word.
    name = None
    head = seg.lstrip()
    mname = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)', head)
    if mname and mname.group(1) in _KNOWN_TOOLS:
        name = mname.group(1)
    else:
        for tn in _KNOWN_TOOLS:
            if re.search(r'\b' + re.escape(tn) + r'\b', seg):
                name = tn
                break
    if name is None:
        return None

    # Resolve arguments. Priority:
    #   (1) GLM <arg_key>K</arg_key><arg_value>V</arg_value> pairs (V may be raw text)
    #   (2) embedded JSON dict   (3) name(...) literal   (4) YAML kv lines
    args: dict = {}
    _ARG_KEYS = ("sql", "question", "table_name", "column_name", "knowledge_name")

    # (1) <arg_key>/<arg_value> pairs; tolerate a missing closing </arg_value> (truncation).
    arg_pairs = re.findall(r'<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>', seg, re.DOTALL)
    if not arg_pairs:
        m2 = re.search(r'<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>(.*)$', seg, re.DOTALL)
        if m2:
            arg_pairs = [(m2.group(1), m2.group(2))]
    for k, v in arg_pairs:
        k = k.strip(); v = v.strip()
        if k in ("table_name", "column_name", "knowledge_name"):
            v = v.strip('\'"')
        if k in _ARG_KEYS:
            args[k] = v

    # (2) embedded JSON dict
    if not args:
        j = _extract_bare_json_dict(seg)
        if isinstance(j, dict):
            if isinstance(j.get("arguments"), dict):
                args = j["arguments"]
            elif "name" not in j:
                args = j

    # (3) name(...) literal
    if not args:
        mlit = re.search(re.escape(name) + r'\((.*)\)', seg, re.DOTALL)
        if mlit:
            inner = mlit.group(1).strip()
            if inner and inner != "{}":
                inner_s = inner.strip().strip('\'"')
                if name in ("execute", "submit"):
                    args = {"sql": inner_s}
                elif name == "ask":
                    args = {"question": inner_s}

    # (4) YAML-ish "key: value" lines
    if not args:
        for line in seg.splitlines():
            mkv = re.match(r'\s*([a-zA-Z_]+)\s*:\s*(.+)', line)
            if mkv and mkv.group(1) in _ARG_KEYS:
                args[mkv.group(1)] = mkv.group(2).strip().strip('\'"')

    interaction_object, action = _convert_tool_call_to_action(name, args)
    return thought, interaction_object, action, {"name": name, "arguments": args}


def _try_parse_tool_call_format(response: str) -> Optional[Tuple[str, str, str]]:
    """
    Try to parse <tool_call> JSON format (from function-calling SFT models).
    Returns (thought, interaction_object, action) if successful, None otherwise.
    """
    if "<tool_call>" not in response and '"name"' not in response:
        return None

    # Extract thought from <think> tags (Qwen-style thinking)
    thought = ""
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    if think_match:
        thought = think_match.group(1).strip()
    elif '<think>' in response:
        after_think = response.split('<think>', 1)[1]
        before_tool = after_think.split('<tool_call>', 1)[0] if '<tool_call>' in after_think else after_think
        candidate = before_tool.strip()
        if candidate and not candidate.startswith('{'):
            thought = candidate

    # Try to extract JSON: first from <tool_call>...</tool_call>, then from full response
    tool_data = None
    tc_match = re.search(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
    if tc_match:
        tool_data = _extract_json_from_text(tc_match.group(1))

    if tool_data is None:
        tool_data = _extract_json_from_text(response)

    if tool_data is None:
        return None

    name = tool_data.get("name", "")
    arguments = tool_data.get("arguments", {})
    if not name:
        return None
    if not isinstance(arguments, dict):
        arguments = {}

    interaction_object, action = _convert_tool_call_to_action(name, arguments)
    return thought, interaction_object, action


def parse_agent_response_ex(response: str) -> Tuple[str, str, str, Optional[dict]]:
    """
    Like parse_agent_response, but also returns the raw tool_call dict
    ({"name":..., "arguments":...}) when the response was in tool_call JSON
    format, else None. Used by the native_fncall path to log tool_call_args.
    """
    # --- Try tool_call format first (scaffold for SFT models) ---
    tool_call_result = _try_parse_tool_call_format(response)
    if tool_call_result is not None:
        thought, interaction_object, action = tool_call_result
        tool_data = _extract_json_from_text(response)
        return thought, interaction_object, action, tool_data

    # --- Tolerant fallback for provider-native tool-call styles (e.g. GLM) ---
    native_result = _try_parse_native_toolcall_fallback(response)
    if native_result is not None:
        return native_result

    thought, interaction_object, action = _parse_react_format(response)
    return thought, interaction_object, action, None


def parse_agent_response(response: str) -> Tuple[str, str, str]:
    """
    Parse the agent's response into thought, interaction object, and action.
    Supports both standard ReAct format (<thought>/<interaction_object>/<action>)
    and tool_call JSON format (<tool_call>{"name":...,"arguments":...}</tool_call>).
    """
    thought, interaction_object, action, _ = parse_agent_response_ex(response)
    return thought, interaction_object, action


def _parse_react_format(response: str) -> Tuple[str, str, str]:
    """Standard ReAct (<thought>/<interaction_object>/<action>) parsing."""
    # --- Standard ReAct format parsing ---
    thought = ""
    interaction_object = ""
    action = ""

    # Extract thought
    thought_match = re.search(r'<thought>(.*?)</thought>', response, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
    else:
        lines = response.split('\n')
        for line in lines:
            if line.strip().lower().startswith("thought:"):
                thought = line.split(":", 1)[1].strip()
                break
        if not thought:
            thought = lines[0] if lines else ""

    # Extract interaction object
    object_match = re.search(r'<interaction_object>(.*?)</interaction_object>', response, re.DOTALL)
    if object_match:
        interaction_object = object_match.group(1).strip()
    else:
        if any(kw in response for kw in ["ask(", "submit("]):
            interaction_object = "User"
        elif any(kw in response for kw in ["execute(", "get_schema(", "get_column_meaning(", "get_knowledge_definition("]):
             interaction_object = "Environment"
        else:
             interaction_object = "Environment"

    # Extract action
    action_match = re.search(r'<action>(.*?)</action>', response, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
    else:
        lines = response.split('\n')
        for line in reversed(lines):
            line_stripped = line.strip()
            if line_stripped.startswith(("execute(", "get_schema(", "get_all_column_meanings(",
                                       "get_column_meaning(", "get_all_external_knowledge_names(",
                                       "get_knowledge_definition(", "get_all_knowledge_definitions(",
                                       "ask(", "submit(")):
                action = line_stripped
                break
        if not action:
             action = lines[-1].strip() if lines else ""

    # Basic validation/cleanup
    if interaction_object not in ["User", "Environment"]:
        if action.startswith("ask(") or action.startswith("submit("):
            interaction_object = "User"
        else:
            interaction_object = "Environment"

    return thought, interaction_object, action

# --- User Simulator Prompts (Encoder/Decoder) --- #

def _get_sql_segments(sql: Union[str, List[str]]) -> str:
    """Segments SQL and caches the result."""
    sql_key = sql if isinstance(sql, str) else "\n===\n".join(sql)
    if sql_key in _segment_sql_cache:
        return _segment_sql_cache[sql_key]

    sql_list = [sql] if isinstance(sql, str) else sql
    sql_segments = ""
    for i, s in enumerate(sql_list):
        if i > 0:
            sql_segments += "\n===\n"
        try:
            for clause, text in segment_sql(s):
                sql_segments += clause + ":\n" + text + "\n\n"
        except Exception as e:
            # Fallback for segmentation error
            sql_segments += f"QUERY:\n{s}\n\n" # Treat whole query as one segment
            print(f"Warning: SQL segmentation failed: {e}. Using full query.")

    _segment_sql_cache[sql_key] = sql_segments.strip()
    return _segment_sql_cache[sql_key]

def build_user_encoder_prompt(question: str, sample_status: 'SampleStatus', db_schema: str, user_sim_prompt_version: str = 'v2') -> str:
    """Builds the prompt for the User Simulator Encoder."""
    prompt = USER_SIMULATOR_ENCODER[user_sim_prompt_version].replace('[[clarification_Q]]', question)

    record = sample_status.original_data
    user_query_ambiguity = record.get('user_query_ambiguity', {})
    knowledge_ambiguity = record.get('knowledge_ambiguity', [])

    ambiguity_json = {
        'user_query_ambiguity': user_query_ambiguity,
        'knowledge_ambiguity': knowledge_ambiguity
    }

    # Use phase-specific reference SQL and ambiguity info
    if sample_status.current_phase == 1:
        prompt = prompt.replace('[[amb_json]]', json.dumps(ambiguity_json, indent=2))
        reference_sql = record.get("sol_sql", "")
    else: # Phase 2
        prompt = prompt.replace('[[amb_json]]', json.dumps({}, indent=2)) # No ambiguity in phase 2
        reference_sql = record.get("follow_up", {}).get("sol_sql", "")

    sql_segments = _get_sql_segments(reference_sql)
    prompt = prompt.replace('[[SQL_Glot]]', sql_segments)
    prompt = prompt.replace('[[DB_schema]]', db_schema)

    return prompt

def build_user_decoder_prompt(question: str, encoded_action: str, sample_status: 'SampleStatus', db_schema: str, user_sim_prompt_version: str = 'v2') -> str:
    """Builds the prompt for the User Simulator Decoder."""
    prompt = USER_SIMULATOR_DECODER[user_sim_prompt_version].replace('[[clarification_Q]]', question)
    prompt = prompt.replace('[[Action]]', encoded_action)

    record = sample_status.original_data
    user_query_ambiguity = record.get('user_query_ambiguity', {})
    knowledge_ambiguity = record.get('knowledge_ambiguity', [])
    clear_query = record.get("query", record.get("amb_user_query")) # Fallback to amb query if clear not present

    ambiguity_json = {
        'user_query_ambiguity': user_query_ambiguity,
        'knowledge_ambiguity': knowledge_ambiguity
    }

    # Use phase-specific reference SQL, clear query, and ambiguity info
    if sample_status.current_phase == 1:
        prompt = prompt.replace('[[amb_json]]', json.dumps(ambiguity_json, indent=2))
        reference_sql = record.get("sol_sql", "")
        prompt = prompt.replace('[[clear_query]]', clear_query)
    else: # Phase 2
        prompt = prompt.replace('[[amb_json]]', json.dumps({}, indent=2)) # No ambiguity in phase 2
        reference_sql = record.get("follow_up", {}).get("sol_sql", "")
        # Use follow-up query as the 'clear query' context for phase 2
        follow_up_query = record.get("follow_up", {}).get("query", "")
        prompt = prompt.replace('[[clear_query]]', follow_up_query)

    prompt = prompt.replace('[[GT_SQL]]', reference_sql if isinstance(reference_sql, str) else '\n'.join(reference_sql))
    sql_segments = _get_sql_segments(reference_sql)
    prompt = prompt.replace('[[SQL_Glot]]', sql_segments)
    prompt = prompt.replace('[[DB_schema]]', db_schema)

    return prompt

def parse_encoder_response(response):
    cut_idx = response.find("</s>")
    if cut_idx != -1:
        extracted_response = response[:cut_idx].strip()
    else:
        extracted_response = response

    if "<s>" in extracted_response:
        cut_idx_1 = extracted_response.find("<s>")
        extracted_response = extracted_response[cut_idx_1:].replace("<s>", "").strip()

    return extracted_response


if __name__ == "__main__":
    # txt = "<thought>\nOkay, the user wants the Modulation Complexity Score (MCS) and the Signal-to-Noise Ratio (SNR) as signal details. They also clarified that \"several occurrences\" means more than 5 signals.\n\nNow I have a better understanding of the request. I need to:\n\n1.  Identify the tables containing signal encoding methods, MCS, SNR, complexity measure, and quality ratio.\n2.  Join the necessary tables.\n3.  Group the results by signal encoding method type.\n4.  Calculate the count, average complexity measure, and average quality ratio for each method.\n5.  Include the MCS and SNR in the output.\n6.  Filter for methods with more than 5 signals.\n\nLet's start by exploring the database schema to identify the relevant tables and columns.\n</thought>\n<interaction_object>Environment</interaction_object>\n<action>get_schema()</action>\n\nObservation: ```\nDatabase schema:\nCREATE TABLE Signals (\nSignalID INTEGER PRIMARY KEY,\nSignalType VARCHAR(50),\nSignalStrength DECIMAL(10, 2),\nFrequency DECIMAL(10, 2),\nEncodingMethod VARCHAR(50),\nComplexityMeasure INTEGER,\nQualityRatio DECIMAL(10, 2),\nMCS INTEGER,\nSNR DECIMAL(10, 2)\n);\nCREATE TABLE Observatories (\nObservID INTEGER PRIMARY KEY,\nObservStation VARCHAR(50),\nObservDate DATE,\nLunarStage VARCHAR(20),\nSignal INTEGER,\nFOREIGN KEY (Signal) REFERENCES Signals(SignalID)\n);\n```\n\n[SYSTEM NOTE: Remaining budget: 19.0/22.0]\n"
    txt = "<thought>\nBased on your clarifications:\n1. We will compute the score using the formula:\n   p.TechSigProb * (1 - p.NatSrcProb) * p.SigUnique * (0.5 + p.AnomScore/10)\n2. The classification is:\n   • 'Low' if the computed score is less than 0.25\n   • 'Medium' if the computed score is between 0.25 and 0.75 (i.e., less than 0.75)\n   • 'High' if the computed score is greater than or equal to 0.75.\n3. The BFR measure is computed from the Signals table as:\n   s.BwHz / (s.CenterFreqMhz * 1000000)\n4. The anomaly metric is the standard deviation of p.AnomScore.\n5. We assume these two tables (SignalProbabilities and Signals) are related by a common column (assumed here as SignalID).\n\nBelow is the SQL query that calculates the classification, counts the signals in each classification group, computes the average BFR measure, and the standard deviation of the anomaly metric.\n\n</thought>\n<interaction_object>User</interaction_object>\n<action>\nsubmit(\"SELECT \\\n  CASE \\\n    WHEN (p.TechSigProb * (1 - p.NatSrcProb) * p.SigUnique * (0.5 + p.AnomScore/10)) < 0.25 THEN 'Low' \\\n    WHEN (p.TechSigProb * (1 - p.NatSrcProb) * p.SigUnique * (0.5 + p.AnomScore/10)) < 0.75 THEN 'Medium' \\\n    ELSE 'High' \\\n  END AS classification, \\\n  COUNT(*) AS signal_count, \\\n  AVG(s.BwHz / (s.CenterFreqMhz * 1000000)) AS avg_bfr_measure, \\\n  STDDEV(p.AnomScore) AS anomaly_std_dev \\\nFROM SignalProbabilities p \\\nJOIN Signals s ON p.SignalID = s.SignalID \\\nGROUP BY classification;\")\n</action>"
    # print(txt)
    thought, obj, action = parse_agent_response(txt)
    print("thought: ", thought)
    print("obj: ", obj)
    print("action: ", action)


