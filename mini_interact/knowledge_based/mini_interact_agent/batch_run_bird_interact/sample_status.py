from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class SampleStatus:
    """Holds the status and interaction history for a single sample."""
    idx: int
    original_data: Dict[str, Any]
    current_prompt: str = ""
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    remaining_budget: float = 0.0
    total_budget: float = 0.0
    phase1_completed: bool = False
    phase2_completed: bool = False
    task_finished: bool = False
    current_turn: int = 0
    current_phase: int = 1 # 1 or 2
    # Fields to store temporary results between steps
    last_agent_response: Optional[str] = None
    parsed_action_object: Optional[str] = None
    parsed_action: Optional[str] = None
    parsed_thought: Optional[str] = None
    last_observation: Optional[str] = None
    last_reward: Optional[float] = None
    last_user_response: Optional[str] = None
    force_submit: bool = False # Flag if budget runs out
    successful_phase1_sql: Optional[str] = None # Added field
    # Native function-calling chat mode: base [system, user(query)] messages.
    # None for legacy react_text mode (back-compatible on resume).
    native_base_messages: Optional[List[Dict[str, str]]] = None

    # Add fields for budget tracking categories if needed
    # env_interact_used: float = 0.0
    # submit_used: float = 0.0
    # user_patience_used: float = 0.0

    # You might add methods here for updating status, budget, etc.
    def add_turn_log(self, thought: str, interaction_object: str, action: str, observation: str, reward: float, budget_info: Dict,
                     raw_assistant: Optional[str] = None,
                     tool_call_id: Optional[str] = None,
                     tool_call_args: Optional[Dict] = None):
        """Adds a log entry for the completed turn.

        raw_assistant / tool_call_id / tool_call_args are only populated in
        native_fncall mode; they are omitted from the log when None so legacy
        react_text logs are unchanged.
        """
        entry = {
            "turn": self.current_turn,
            "phase": self.current_phase,
            "thought": thought,
            "interaction_object": interaction_object,
            "action": action,
            "observation": observation,
            "reward": reward, # Reward *received* in this turn (usually 0 unless it's the final submit)
            "budget_after_action": budget_info
        }
        # Fall back to per-turn "pending" native extras if not passed explicitly.
        # In native_fncall mode the main loop stamps these once per turn so every
        # add_turn_log site records them without threading kwargs through each call.
        if raw_assistant is None:
            raw_assistant = getattr(self, "_pending_raw_assistant", None)
        if tool_call_id is None:
            tool_call_id = getattr(self, "_pending_tool_call_id", None)
        if tool_call_args is None:
            tool_call_args = getattr(self, "_pending_tool_call_args", None)
        if raw_assistant is not None:
            entry["raw_assistant"] = raw_assistant
        if tool_call_id is not None:
            entry["tool_call_id"] = tool_call_id
        if tool_call_args is not None:
            entry["tool_call_args"] = tool_call_args
        self.interaction_history.append(entry)

    def get_full_interaction_prompt(self) -> str:
        """Constructs the full prompt history for the agent."""
        prompt = self.current_prompt # Initial query + budget info
        for turn_log in self.interaction_history:
            prompt += f"""<thought>{turn_log['thought']}</thought>
<interaction_object>{turn_log['interaction_object']}</interaction_object>
<action>{turn_log['action']}</action>

Observation: {turn_log['observation']}

"""
        return prompt

    def get_truncated_interaction_prompt(self, max_prompt_chars: int) -> str:
        """Build prompt with oldest observations truncated to fit within char budget.
        
        Truncation strategy: keep the initial prompt (system+query) and all
        thought/action lines intact; progressively shorten the oldest
        observations (which are usually the largest, e.g. schema dumps) until
        the prompt fits.
        """
        base = self.current_prompt
        if not self.interaction_history:
            return base

        turn_blocks = []
        for turn_log in self.interaction_history:
            block = (
                f"<thought>{turn_log['thought']}</thought>\n"
                f"<interaction_object>{turn_log['interaction_object']}</interaction_object>\n"
                f"<action>{turn_log['action']}</action>\n\n"
                f"Observation: {turn_log['observation']}\n\n"
            )
            turn_blocks.append(block)

        full = base + "".join(turn_blocks)
        if len(full) <= max_prompt_chars:
            return full

        TRUNC_MARKER = "\n[... earlier output truncated ...]\n"
        for trunc_idx in range(len(turn_blocks)):
            obs = self.interaction_history[trunc_idx]['observation']
            action_header = (
                f"<thought>{self.interaction_history[trunc_idx]['thought']}</thought>\n"
                f"<interaction_object>{self.interaction_history[trunc_idx]['interaction_object']}</interaction_object>\n"
                f"<action>{self.interaction_history[trunc_idx]['action']}</action>\n\n"
                f"Observation: "
            )
            turn_blocks[trunc_idx] = action_header + TRUNC_MARKER

            candidate = base + "".join(turn_blocks)
            if len(candidate) <= max_prompt_chars:
                return candidate

        return base + "".join(turn_blocks)