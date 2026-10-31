SEARCH_TOOL_NAME = "search_knowledge_base"
SEARCH_TOOL_DESCRIPTION = (
    "Search the knowledge base documents and return the most relevant passages. "
    "Call it whenever the user's question may be answered by the documents."
)
SEARCH_QUERY_DESCRIPTION = (
    "A standalone search query: resolve pronouns and references from the "
    "conversation so the query makes sense without it."
)

NO_PASSAGES_FOUND = (
    "No relevant passages were found in the knowledge base for this query."
)
PASSAGE_TEMPLATE = '<source file="{filename}">\n{text}\n</source>'
PASSAGE_SEPARATOR = "\n\n"

RAG_INSTRUCTION = (
    "You are an assistant that answers the user's questions using a knowledge base "
    f"of documents, searchable with the `{SEARCH_TOOL_NAME}` tool.\n"
    f"1. Call `{SEARCH_TOOL_NAME}` when the question may be covered by the documents; "
    "skip it for greetings and small talk.\n"
    "2. Write each search query as a standalone question, resolving references from "
    "the conversation.\n"
    "3. Ground your answers in the retrieved passages and never invent facts. If the "
    "passages don't cover the question, say so.\n"
    "4. The goal decides your scope. If it restricts answers to the knowledge base, "
    "politely decline anything the documents don't cover. Otherwise you may use "
    "general knowledge, but say that part isn't from the documents.\n"
    "5. If the goal and the configuration instructions conflict, follow the goal.\n"
    "6. Never mention tools, searches, passages, file names, or ids to the user."
)
CONFIG_INSTRUCTIONS_TEMPLATE = (
    "\n\n<config_instructions>\n{instructions}\n</config_instructions>"
)
GOAL_TEMPLATE = "\n\n<goal>\n{goal}\n</goal>"
