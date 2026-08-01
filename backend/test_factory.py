from langchain_core.messages import HumanMessage

from app.llms.factory import LLMFactory
from app.config.settings import settings
print("Provider:", settings.DEFAULT_PROVIDER)
print("Model:", settings.DEFAULT_MODEL)
print("API Key:", settings.GROQ_API_KEY)

provider = LLMFactory.get_provider()

llm = provider.get_llm()

response = llm.invoke(
    [
        HumanMessage(
            content="Explain predictive maintenance in one paragraph."
        )
    ]
)

print(provider.get_name())
print(response.content)