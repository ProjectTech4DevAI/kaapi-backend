EXTRACTION_METHOD = "json_schema"

EXTRACTION_INSTRUCTION = (
    "You extract data from a chatbot conversation. You do not talk to the user.\n"
    "The admin flow prompt below defines the goal, the fields to collect, and the rules.\n"
    "1. Identify every field the flow prompt asks to collect; name each in snake_case. "
    "Treat every field as required unless the flow prompt explicitly marks it optional.\n"
    "2. collected_fields: every field whose value the user has clearly provided in the "
    "conversation, including those in the previously collected map. Reuse the exact "
    "keys of the previously collected map; change a value only if the user corrected it.\n"
    "3. Never invent, guess, or infer a value the user did not give.\n"
    "4. missing_fields: the required fields the user has not answered yet.\n"
    "5. is_complete: true only when no required field is missing."
)
EXTRACTION_FLOW_PROMPT_TEMPLATE = "\n\n<flow_prompt>\n{flow_prompt}\n</flow_prompt>"
EXTRACTION_PRIOR_FIELDS_TEMPLATE = (
    "\n\n<previously_collected>\n{prior_fields}\n</previously_collected>"
)

PROGRESS_NOTE_PREFIX = "[Field collection progress] "
PROGRESS_COLLECTED_TEMPLATE = (
    "Already collected (do not ask for these again): {collected}. "
)
PROGRESS_MISSING_TEMPLATE = "Still missing: {missing}. Ask only for the missing fields."
PROGRESS_COMPLETE_NOTE = "Every required field is collected; confirm and wrap up as the flow prompt describes."
