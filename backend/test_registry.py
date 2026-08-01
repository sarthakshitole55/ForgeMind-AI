from app.langchain_tools.registry import TOOLS

for tool in TOOLS:
    print(tool.name)
    print(tool.description)
    print("=" * 80)