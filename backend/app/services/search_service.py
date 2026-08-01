from tavily import TavilyClient
from app.config.settings import settings

class SearchService:

    def __init__(self):
        self.client = TavilyClient(api_key=settings.TAVILY_API_KEY)

    def search(self, query: str):
        from app.core.exceptions import SearchError
        from app.core.logger import logger
        try:
            logger.info(f"Initiating Web Search for query: '{query}'")
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=settings.SEARCH_MAX_RESULTS,
            )
            results = response.get("results", [])
            logger.info(f"Web Search completed. Found {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}", exc_info=True)
            raise SearchError(f"Search service failed: {str(e)}")