from app.langchain_tools.rag import search_documents
from app.langchain_tools.web import search_web
from app.langchain_tools.maintenance import maintenance_wokflow

TOOLS = [
    search_documents,
    search_web,
    maintenance_wokflow,

]