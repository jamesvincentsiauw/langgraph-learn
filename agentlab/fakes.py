"""A scripted chat model so graphs can be tested without an API key.

ScriptedChatModel returns the given messages in order, one per model call. It accepts
bind_tools (the tools are ignored, the script decides what gets called) and
with_structured_output (the next scripted message's content is parsed as JSON into the schema).
"""

import json
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class ScriptedChatModel(GenericFakeChatModel):
    def __init__(self, script: Sequence[BaseMessage | str], **kwargs: Any):
        msgs: Iterator[BaseMessage] = iter(
            AIMessage(content=m) if isinstance(m, str) else m for m in script
        )
        super().__init__(messages=msgs, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        def parse(_input: Any) -> Any:
            reply = self.invoke(_input)
            payload = json.loads(reply.content)
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                return schema.model_validate(payload)
            return payload

        return RunnableLambda(parse)
