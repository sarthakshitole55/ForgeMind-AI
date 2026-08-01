from app.tools.rag_tools import RAGTool
from app.tools.web_tools import WebTool
from app.tools.maintenance_tool import MaintenanceTool

print("=" * 80)
print("Testing RAG Tool")
print("=" * 80)

rag = RAGTool()

print(
    rag.invoke(
        "What is competitive programming?"
    )
)

print("\n")

print("=" * 80)
print("Testing Web Tool")
print("=" * 80)

web = WebTool()

print(
    web.invoke(
        "Latest Python release"
    )
)

print("\n")

print("=" * 80)
print("Testing Maintenance Tool")
print("=" * 80)

maintenance = MaintenanceTool()

print(
    maintenance.invoke(
        "Motor vibration after bearing replacement"
    )
)