from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from trace_store import TraceStore


@dataclass
class TraceRecorder:
    store: TraceStore
    latest_white_node_ids: tuple[str, ...]
    pending_tool_parents: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        prompt: str,
        eval_run_id: str | None = None,
        eval_target: Any | None = None,
    ) -> "TraceRecorder":
        store = TraceStore(eval_run_id=eval_run_id)
        root_node_id = f"node_{uuid4().hex}"
        store.add_node(
            node_id=root_node_id,
            kind="problem",
            color="white",
            content=prompt,
            metadata={},
        )
        if eval_target is not None:
            store.add_eval_target(
                task_name=eval_target.task_name,
                doc_id=eval_target.doc_id,
                prompt_text=eval_target.prompt_text,
                expected_answer=eval_target.expected_answer,
                doc=eval_target.doc,
                metadata=eval_target.metadata,
            )
        return cls(store=store, latest_white_node_ids=(root_node_id,))

    def record_response(self, resp: Any) -> None:
        response_node_id = f"node_{uuid4().hex}"
        self.store.add_node(
            node_id=response_node_id,
            kind="response",
            color="black",
            content=resp.output_text,
            metadata=_serialize_response(resp),
        )
        for white_node_id in self.latest_white_node_ids:
            self.store.add_edge(
                source=white_node_id,
                target=response_node_id,
                relation="prompted_response",
                metadata={"response_id": resp.id},
            )

        self.latest_white_node_ids = ()
        self.pending_tool_parents = {
            output_item.call_id: response_node_id
            for output_item in (resp.output or [])
            if getattr(output_item, "type", None) == "function_call"
        }

    def record_tool_result(
        self,
        *,
        call_id: str,
        code: str,
        tool_output: dict[str, Any],
    ) -> None:
        tool_node_id = f"node_{uuid4().hex}"
        self.store.add_node(
            node_id=tool_node_id,
            kind="tool_output",
            color="white",
            content=tool_output,
            metadata={
                "call_id": call_id,
                "tool_name": "python_code_interpreter",
                "code": code,
            },
        )
        parent_node_id = self.pending_tool_parents.get(call_id)
        if parent_node_id:
            self.store.add_edge(
                source=parent_node_id,
                target=tool_node_id,
                relation="tool_result",
                metadata={"call_id": call_id},
            )
        self.latest_white_node_ids = (*self.latest_white_node_ids, tool_node_id)


def _serialize_response(resp: Any) -> dict[str, Any]:
    return {
        "id": resp.id,
        "output_text": resp.output_text,
        "output": [
            {
                "id": getattr(item, "id", None),
                "type": getattr(item, "type", None),
                "name": getattr(item, "name", None),
                "call_id": getattr(item, "call_id", None),
                "arguments": getattr(item, "arguments", None),
            }
            for item in (resp.output or [])
        ],
    }
