from langchain_core.tools import tool
from app.tools.maintenance_tool import MaintenanceTool

maintenance = MaintenanceTool()

@tool
def maintenance_wokflow(question:str):
    """
    Execute maintenance troubleshooting workflow for industrial machines.
    """
    return maintenance.invoke(question)