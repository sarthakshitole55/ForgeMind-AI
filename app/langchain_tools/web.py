from langchain_core.tools import tool
from app.tools.web_tools import WebTool

web = WebTool()

@tool
def search_web(question:str):
    """
    Search the internet for recent and real-time information.
    """
    return web.invoke(question)
    