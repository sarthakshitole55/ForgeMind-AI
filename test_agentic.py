from app.agents.graph import graph

result = graph.invoke(
    {
        "question": "Motor vibration after replacing bearings"
    }
)

print(result["answer"])