import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from functools import wraps
import json
import logging
from pathlib import Path
import re
from string import Template
import sys
from typing import Any, Awaitable, Callable
from uuid import uuid4

import openai
from openai import AsyncOpenAI

from trace_store import TraceStore


ALLOW_TOOL_CALLS = True
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
SYSTEM_PROMPT_TEMPLATE = TEMPLATES_DIR / "system_prompt.txt"
TOOL_USE_PROMPT_TEMPLATE = TEMPLATES_DIR / "tool_use_prompt.txt"


logger = logging.getLogger(__name__)
FINAL_ANSWER_RE = re.compile(r"final answer:\s*(.+)", re.IGNORECASE)
MAX_TOKENS = 20000
MAX_TOOL_CALLS = 20
LOCAL_CODE_TIMEOUT_SECONDS = 2
API_TIMEOUT_SECONDS = 30.0
API_MAX_ATTEMPTS = 3
CODE_INTERPRETER_TOOL = {
    "type": "function",
    "name": "python_code_interpreter",
    "description": (
        "Run Python code locally for calculations, symbolic checks, "
        "or quick experiments."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute in the sandbox.",
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    },
    "strict": True,
}


@dataclass(frozen=True)
class ConversationState:
    input: str | list[dict]
    prev_response_id: str | None = None
    trace: "TraceStore | None" = None
    trace_id: str | None = None
    latest_white_node_ids: tuple[str, ...] = ()
    pending_tool_parents: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Final:
    text: str


@dataclass(frozen=True)
class ToolRequest:
    call_id: str
    code: str


@dataclass(frozen=True)
class Calls:
    requests: list[ToolRequest]


@dataclass(frozen=True)
class EvalTarget:
    task_name: str | None
    doc_id: int | None
    prompt_text: str | None
    expected_answer: str | None
    doc: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


StepResult = Final | Calls
StepFn = Callable[[ConversationState], Awaitable[tuple[StepResult, ConversationState]]]


class ResponseStateMachine:
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        eval_run_id: str | None = None,
    ):
        self.client = client
        self.model = model
        self.eval_run_id = eval_run_id

    async def solve(self, prompt: str, eval_target: EvalTarget | None = None) -> str:
        return await self._unfold(
            self._build_step(),
            init_state(prompt, eval_run_id=self.eval_run_id, eval_target=eval_target),
        )

    def _build_step(self) -> StepFn:
        @with_trace
        @with_retry
        @with_timeout
        async def base_step(state: ConversationState):
            resp = await self.client.responses.create(
                **build_request_kwargs(self.model, state)
            )
            trace_info = record_response_node(state, resp)
            next_state = ConversationState(
                input=state.input,
                prev_response_id=resp.id,
                trace=state.trace,
                trace_id=state.trace_id,
                latest_white_node_ids=state.latest_white_node_ids,
                pending_tool_parents=trace_info.get("tool_parent_map", {}),
            )
            return classify_response(resp), next_state

        return base_step

    async def _unfold(self, step_fn: StepFn, state: ConversationState) -> str:
        for _ in range(MAX_TOOL_CALLS):
            result, next_state = await step_fn(state)
            match result:
                case Final(text):
                    return text
                case Calls(requests):
                    tool_outputs = await asyncio.gather(
                        *(self._run_tool_request(next_state, request) for request in requests)
                    )
                    state = advance_state(next_state, tool_outputs)

        logger.warning("Exceeded maximum tool-call loop count for state: %r", state)
        return ""

    async def _run_tool_request(
        self, state: ConversationState, request: ToolRequest
    ) -> dict:
        if not request.code.strip():
            output = "Tool argument error: `code` must be a non-empty string."
        else:
            output = await self._run_code_locally(request.code)
        tool_output = tool_output_input(request.call_id, output)
        record_tool_output_node(state, request, tool_output)
        return tool_output

    async def _run_code_locally(self, code: str) -> str:
        try:
            logger.info("Executing code tool call.")
            logger.debug("Code to execute:\n%s", code)
            # Warning: running untrusted code here. TODO: sandboxing
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            communicate_task = asyncio.create_task(process.communicate())
            stdout, stderr = await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=LOCAL_CODE_TIMEOUT_SECONDS,
            )
            logger.debug(
                "Code execution completed. stdout: %r stderr: %r", stdout, stderr
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Code execution timed out after %ss", LOCAL_CODE_TIMEOUT_SECONDS
            )
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            stdout, stderr = b"", b""
            with suppress(asyncio.CancelledError):
                stdout, stderr = await communicate_task
            return json.dumps(
                build_code_result(process.returncode, stdout, stderr, timed_out=True),
                ensure_ascii=True,
            )
        except Exception as e:
            logger.exception("Local code execution failed")
            return f"Local code execution failed: {e}"

        return json.dumps(
            build_code_result(process.returncode, stdout, stderr, timed_out=False),
            ensure_ascii=True,
        )


def render_system_prompt(*, allow_tool_calls: bool) -> str:
    tool_instructions = ""
    if allow_tool_calls:
        tool_instructions = Template(
            TOOL_USE_PROMPT_TEMPLATE.read_text(encoding="utf-8")
        ).substitute(
            tool_name="python_code_interpreter"
        )

    template = Template(SYSTEM_PROMPT_TEMPLATE.read_text(encoding="utf-8"))
    return template.substitute(tool_instructions=tool_instructions).strip()


def init_state(
    prompt: str,
    *,
    eval_run_id: str | None = None,
    eval_target: EvalTarget | None = None,
) -> ConversationState:
    trace = TraceStore(eval_run_id=eval_run_id)
    root_node_id = f"node_{uuid4().hex}"
    trace.add_node(
        node_id=root_node_id,
        kind="problem",
        color="white",
        content=prompt,
        metadata={},
    )
    if eval_target is not None:
        trace.add_eval_target(
            task_name=eval_target.task_name,
            doc_id=eval_target.doc_id,
            prompt_text=eval_target.prompt_text,
            expected_answer=eval_target.expected_answer,
            doc=eval_target.doc,
            metadata=eval_target.metadata,
        )
    return ConversationState(
        input=prompt,
        trace=trace,
        trace_id=trace.trace_id,
        latest_white_node_ids=(root_node_id,),
    )


def extract_final_answer(text: str) -> str:
    if not text:
        logger.warning("Model returned empty content; no final answer found.")
        return ""

    matches = FINAL_ANSWER_RE.findall(text)
    if matches:
        answer = matches[-1].strip()
        if answer:
            return answer
        logger.warning(
            "Found 'final answer:' marker but no answer content. Raw response: %r",
            text,
        )
        return ""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        logger.warning(
            "No 'final answer:' marker found; falling back to last non-empty line: %r",
            lines[-1],
        )
        return lines[-1]

    logger.warning("Response had no non-empty lines; no final answer found.")
    return ""


def build_request_kwargs(model: str, state: ConversationState) -> dict:
    request_kwargs = {
        "model": model,
        "instructions": render_system_prompt(allow_tool_calls=ALLOW_TOOL_CALLS),
        "input": state.input,
        "max_output_tokens": MAX_TOKENS,
        "reasoning": {"effort": "high"},
        "previous_response_id": state.prev_response_id,
    }
    if ALLOW_TOOL_CALLS:
        request_kwargs["tools"] = [CODE_INTERPRETER_TOOL]
        request_kwargs["tool_choice"] = "auto"
    return request_kwargs


def parse_tool_request(output_item) -> ToolRequest:
    if output_item.name != "python_code_interpreter":
        logger.warning("Unsupported tool call requested: %s", output_item.name)
        return ToolRequest(call_id=output_item.call_id, code="")

    try:
        arguments = json.loads(output_item.arguments or "{}")
    except json.JSONDecodeError as e:
        logger.warning(
            "Invalid tool arguments from model: %r (%s)",
            output_item.arguments,
            e,
        )
        return ToolRequest(call_id=output_item.call_id, code="")

    return ToolRequest(
        call_id=output_item.call_id,
        code=arguments.get("code", ""),
    )


def classify_response(resp) -> StepResult:
    if not resp.output:
        logger.warning("API response had no output items: %r", resp)
        return Final("")

    calls = [
        parse_tool_request(output_item)
        for output_item in resp.output
        if output_item.type == "function_call"
    ]
    if calls:
        return Calls(calls)

    if not resp.output_text:
        logger.warning("API response had no output text: %r", resp)
        return Final("")
    return Final(resp.output_text)


def serialize_response(resp) -> dict:
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


def record_response_node(state: ConversationState, resp) -> dict[str, object]:
    if state.trace is None:
        return {}

    response_node_id = f"node_{uuid4().hex}"
    state.trace.add_node(
        node_id=response_node_id,
        kind="response",
        color="black",
        content=resp.output_text,
        metadata=serialize_response(resp),
    )
    for white_node_id in state.latest_white_node_ids:
        state.trace.add_edge(
            source=white_node_id,
            target=response_node_id,
            relation="prompted_response",
            metadata={"response_id": resp.id},
        )

    tool_parent_map = {}
    for output_item in resp.output or []:
        if getattr(output_item, "type", None) == "function_call":
            tool_parent_map[output_item.call_id] = response_node_id

    return {
        "response_node_id": response_node_id,
        "tool_parent_map": tool_parent_map,
    }


def record_tool_output_node(
    state: ConversationState, request: ToolRequest, tool_output: dict
) -> str | None:
    if state.trace is None:
        return None

    tool_node_id = f"node_{uuid4().hex}"
    state.trace.add_node(
        node_id=tool_node_id,
        kind="tool_output",
        color="white",
        content=tool_output,
        metadata={
            "call_id": request.call_id,
            "tool_name": "python_code_interpreter",
            "code": request.code,
        },
    )
    parent_node_id = state.pending_tool_parents.get(request.call_id)
    if parent_node_id:
        state.trace.add_edge(
            source=parent_node_id,
            target=tool_node_id,
            relation="tool_result",
            metadata={"call_id": request.call_id},
        )
    return tool_node_id


def tool_output_input(call_id: str, output: str) -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


def advance_state(
    state: ConversationState, tool_outputs: list[dict]
) -> ConversationState:
    latest_white_node_ids: list[str] = []
    if state.trace is not None and tool_outputs:
        for tool_output in tool_outputs:
            call_id = tool_output.get("call_id")
            node_id = state.trace.find_node_id_by_call_id(call_id)
            if node_id is not None:
                latest_white_node_ids.append(node_id)

    return ConversationState(
        input=tool_outputs,
        prev_response_id=state.prev_response_id,
        trace=state.trace,
        trace_id=state.trace_id,
        latest_white_node_ids=tuple(latest_white_node_ids),
    )


def build_code_result(
    returncode: int | None, stdout: bytes, stderr: bytes, *, timed_out: bool
) -> dict:
    return {
        "stdout": stdout.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "returncode": returncode,
        "timed_out": timed_out,
    }


def with_timeout(step_fn: StepFn) -> StepFn:
    @wraps(step_fn)
    async def wrapper(state: ConversationState):
        return await asyncio.wait_for(step_fn(state), timeout=API_TIMEOUT_SECONDS)

    return wrapper


def with_retry(step_fn: StepFn) -> StepFn:
    @wraps(step_fn)
    async def wrapper(state: ConversationState):
        for attempt in range(API_MAX_ATTEMPTS):
            try:
                return await step_fn(state)
            except openai.BadRequestError as e:
                logger.error(
                    "Bad request (will not retry). state: %r error: %s",
                    state,
                    e,
                )
                return Final(""), state
            except openai.AuthenticationError as e:
                logger.error("Authentication failed, aborting: %s", e)
                raise
            except (
                asyncio.TimeoutError,
                openai.RateLimitError,
                openai.APIStatusError,
                openai.APIConnectionError,
            ) as e:
                if attempt == API_MAX_ATTEMPTS - 1:
                    logger.error(
                        "API step failed after %d attempts: %s",
                        API_MAX_ATTEMPTS,
                        e,
                    )
                    return Final(""), state
                wait = 2**attempt
                logger.warning(
                    "Retrying in %ds (attempt %d/%d): %s",
                    wait,
                    attempt + 1,
                    API_MAX_ATTEMPTS,
                    e,
                )
                await asyncio.sleep(wait)

    return wrapper


def with_trace(step_fn: StepFn) -> StepFn:
    @wraps(step_fn)
    async def wrapper(state: ConversationState):
        result, new_state = await step_fn(state)
        logger.debug("Step state=%r result=%r next_state=%r", state, result, new_state)
        return result, new_state

    return wrapper
