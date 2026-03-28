"""
Optional pre-LLM context (currently unused).
Menu knowledge is loaded via the get_menu_items tool + pgvector in tool_executor.
"""


class RetrievalService:
    def retrieve(self, query: str, top_k: int = 3) -> list:
        return []
