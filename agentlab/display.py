from langchain_core.messages import BaseMessage


def describe(msg: BaseMessage) -> str:
    """One line per message: tool calls, tool results, or the model's text."""
    if getattr(msg, "tool_calls", None):
        return "; ".join(f"calls {c['name']}({c['args']})" for c in msg.tool_calls)
    if msg.type == "tool":
        return f"{msg.name} -> {str(msg.content)[:160]}"
    return f"answer:\n{msg.text}"


def print_updates(stream) -> None:
    """Print each graph step so the model -> tools -> model loop is visible."""
    for update in stream:
        for node, output in update.items():
            if not isinstance(output, dict):
                continue
            for msg in output.get("messages", []):
                print(f"[{node}] {describe(msg)}")
