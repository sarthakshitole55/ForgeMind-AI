from app.agents.graph import graph

result = graph.invoke(
    {
        "question": "Building a Modern CNN ResNet"
    }
)

print("\n===== FINAL ANSWER =====")
print(result["answer"])