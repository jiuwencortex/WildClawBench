from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

JIUWENSWARM_SESSIONS = "/root/.jiuwenswarm/agent/sessions"
OUTPUT_TRANSCRIPT_PATH = "/root/.openclaw/agents/main/sessions/chat.jsonl"


def _load_history(session_id: str) -> list[dict[str, Any]]:
    history_path = Path(JIUWENSWARM_SESSIONS) / session_id / "history.jsonl"
    if not history_path.exists():
        history_path = Path(JIUWENSWARM_SESSIONS) / session_id / "history.json"
    if not history_path.exists():
        return []

    if history_path.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        for line in history_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if isinstance(item, dict):
                records.append(item)
        return records
    else:
        try:
            data = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return data if isinstance(data, list) else []


def _assistant_entry(record: dict[str, Any]) -> dict[str, Any]:
    content = record.get("content", "")
    tool_calls = record.get("tool_calls")
    usage = record.get("usage", {})

    if not isinstance(tool_calls, list) or not tool_calls:
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": content,
                "usage": usage if isinstance(usage, dict) else {},
            },
        }

    content_blocks: list[dict[str, Any]] = []
    if isinstance(content, str) and content.strip():
        content_blocks.append({"type": "text", "text": content})

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_payload = tool_call.get("function", {})
        if not isinstance(function_payload, dict):
            function_payload = {}
        raw_arguments = function_payload.get("arguments", "")
        parsed_arguments: Any = raw_arguments
        if isinstance(raw_arguments, str):
            try:
                parsed_arguments = json.loads(raw_arguments)
            except Exception:
                parsed_arguments = raw_arguments
        content_blocks.append({
            "type": "tool_use",
            "name": str(function_payload.get("name", "")),
            "input": parsed_arguments,
            "id": str(tool_call.get("id", "")),
        })

    return {
        "type": "message",
        "message": {
            "role": "assistant",
            "content": content_blocks,
            "usage": usage if isinstance(usage, dict) else {},
        },
    }


def _user_entry(record: dict[str, Any]) -> dict[str, Any]:
    content = record.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return {
        "type": "message",
        "message": {"role": "user", "content": content},
    }


def _tool_entry(record: dict[str, Any]) -> dict[str, Any]:
    content = record.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return {
        "type": "toolResult",
        "toolResult": {
            "content": content,
            "tool_call_id": str(record.get("tool_call_id", "")),
        },
    }


def _to_openclaw_messages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for record in records:
        role = str(record.get("role", ""))
        if role == "assistant":
            converted.append(_assistant_entry(record))
        elif role == "user":
            converted.append(_user_entry(record))
        elif role == "tool":
            converted.append(_tool_entry(record))
    return converted


def main() -> int:
    session_id = os.environ.get("JIUWENSWARM_BENCH_SESSION", "bench-session")
    output_path = Path(OUTPUT_TRANSCRIPT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_records = _load_history(session_id)
    converted = _to_openclaw_messages(source_records)
    payload = ""
    if converted:
        payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in converted) + "\n"
    output_path.write_text(payload, encoding="utf-8")

    print(
        f"Wrote compat transcript to {OUTPUT_TRANSCRIPT_PATH} "
        f"({len(converted)} messages from {len(source_records)} source items)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
