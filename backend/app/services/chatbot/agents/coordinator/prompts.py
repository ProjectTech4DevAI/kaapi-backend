ROUTING_INSTRUCTION = (
    "\n\nYou route the user to the right assistant. Read the conversation and decide "
    "which of the intents below the user's latest message expresses.\n"
    "1. intent: copy exactly one intent name from the list when the user's need clearly "
    "matches it; otherwise null. Never invent an intent that is not listed.\n"
    "2. clarifying_question: when intent is null, a short, friendly question in the "
    "user's language that helps them say which of the listed options they need; "
    "otherwise null."
)
ROUTES_TEMPLATE = "\n\n<intents>\n{routes}\n</intents>"

FALLBACK_CLARIFYING_QUESTION = (
    "Could you tell me a bit more about what you need help with?"
)
