import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

load_dotenv()

DEFAULT_QUESTION = "Ada lowongan engineer di Singapore yang nyebutin gaji? Rupiahin gajinya."

SYSTEM = (
    "You help Vincent, a backend engineer in Jakarta, look at job openings abroad. "
    "Use the tools to find openings and convert salaries to IDR. "
    "Only state salaries the tools returned. Answer in casual Indonesian."
)


def make_model(model: str | None = None, **kwargs) -> BaseChatModel:
    """Anthropic when ANTHROPIC_API_KEY is set, otherwise OpenRouter. MODEL overrides the id."""
    kwargs.setdefault("max_tokens", 2048)
    if os.getenv("ANTHROPIC_API_KEY"):
        return init_chat_model(
            model or os.getenv("MODEL", "claude-haiku-5-5"), model_provider="anthropic", **kwargs
        )
    # OpenRouter speaks the OpenAI API, so the openai provider works with a different base_url.
    return init_chat_model(
        model or os.getenv("MODEL", "anthropic/claude-haiku-5.5"),
        model_provider="openai",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        **kwargs,
    )
