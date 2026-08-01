from app.config.settings import settings
from app.llms.providers.groq import GroqProvider

PROVIDERS = {
    "groq": GroqProvider,
}


class LLMFactory:

    @staticmethod
    def get_provider():
        provider_name = settings.DEFAULT_PROVIDER.lower()

        provider_class = PROVIDERS.get(provider_name)

        if provider_class is None:
            raise ValueError(f"Unsupported provider: {provider_name}")

        return provider_class()