from app.services.search_service import SearchService
from app.tools.base import BaseTool

class WebTool(BaseTool):
    def __init__(self):
        self.search = SearchService()
    
    def invoke(self,query:str):
        return self.search.search(query)
    
    def name(self):
        return "web"