"""A scripted chat model so graphs can be tested without an API key.

ScriptedChatModel returns the given messages in order, one per model call. It accepts
bind_tools (the tools are ignored, the script decides what gets called) and
with_structured_output (the next scripted message's content is parsed as JSON into the schema).
Every prompt the model receives is kept in `seen`, so a test can assert what the graph
actually sent (history included) and not only what came back. It streams too: text comes
out one whitespace-separated token at a time and a tool call as a single chunk, so
stream_mode="messages" and astream_events work in tests.

KeyedChatModel picks its reply by which key appears in the prompt, for graphs whose model
calls run in parallel (Send) and so arrive in no fixed order.
"""

import json
import re
from collections.abc import Iterator, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def structured(model: BaseChatModel, schema: Any) -> Runnable:
    """Parse the fake's next reply as JSON into the schema, like with_structured_output does."""

    def parse(_input: Any) -> Any:
        reply = model.invoke(_input)
        payload = json.loads(reply.content)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(payload)
        return payload

    return RunnableLambda(parse)


class ScriptedChatModel(GenericFakeChatModel):
    seen: list[list[BaseMessage]] = []

    def __init__(self, script: Sequence[BaseMessage | str], **kwargs: Any):
        msgs: Iterator[BaseMessage] = iter(
            AIMessage(content=m) if isinstance(m, str) else m for m in script
        )
        super().__init__(messages=msgs, seen=[], **kwargs)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.seen.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        # GenericFakeChatModel yields nothing for a message with empty content, which makes
        # langchain raise "No generations found in stream" on every scripted tool call.
        message = self._generate(messages, stop=stop, run_manager=run_manager).generations[0]
        message = message.message
        pieces = re.split(r"(\s)", message.content) if message.content else [""]
        tool_chunks = [
            {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
            for i, c in enumerate(getattr(message, "tool_calls", []))
        ]
        for i, piece in enumerate(pieces):
            last = i == len(pieces) - 1
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(
                    content=piece,
                    id=message.id,
                    tool_call_chunks=tool_chunks if last else [],
                    usage_metadata=message.usage_metadata if last else None,
                    chunk_position="last" if last else None,
                )
            )
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        return structured(self, schema)


class KeyedChatModel(BaseChatModel):
    replies: dict[str, str]

    @property
    def _llm_type(self) -> str:
        return "keyed-fake"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any):
        prompt = "\n".join(str(m.text) for m in messages)
        for key, reply in self.replies.items():
            if key in prompt:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])
        raise KeyError(f"no reply keyed for this prompt: {prompt[:120]!r}")

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        return structured(self, schema)
