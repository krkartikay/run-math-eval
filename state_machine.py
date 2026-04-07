import asyncio
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
import json
import logging
import re
import sys
from typing import Awaitable, Callable

import openai
from openai import AsyncOpenAI


ALLOW_TOOL_CALLS = True

SYSTEM = """You are a Math problem solver. Your task is to solve math problems.

Solve the problem step by step.
On a separate final line, write exactly:
final answer: <answer>

Keep the final answer concise and in canonical mathematical form when possible, for example:
2
\\frac{3}{4}
\\sqrt{74}
"""

if ALLOW_TOOL_CALLS:
    SYSTEM += """
You may use the `python_code_interpreter` tool for calculations or verification.
Use it when it helps, then continue reasoning from the tool result.
"""

SYSTEM += """
When you are done, respond normally with the required `final answer:` line.
"""


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


StepResult = Final | Calls
StepFn = Callable[[ConversationState], Awaitable[tuple[StepResult, ConversationState]]]


class ResponseStateMachine:
    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
    ):
        self.client = client
        self.model = model

    async def solve(self, prompt: str) -> str:
        return await self._unfold(self._build_step(), init_state(prompt))

    def _build_step(self) -> StepFn:
        @with_trace
        @with_retry
        @with_timeout
        async def base_step(state: ConversationState):
            resp = await self.client.responses.create(
                **build_request_kwargs(self.model, state)
            )
            next_state = ConversationState(input=state.input, prev_response_id=resp.id)
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
                        *(self._run_tool_request(request) for request in requests)
                    )
                    state = advance_state(next_state, tool_outputs)

        logger.warning("Exceeded maximum tool-call loop count for state: %r", state)
        return ""

    async def _run_tool_request(self, request: ToolRequest) -> dict:
        if not request.code.strip():
            output = "Tool argument error: `code` must be a non-empty string."
        else:
            output = await self._run_code_locally(request.code)
        return tool_output_input(request.call_id, output)

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


def init_state(prompt: str) -> ConversationState:
    return ConversationState(input=prompt)


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
        "instructions": SYSTEM,
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


def tool_output_input(call_id: str, output: str) -> dict:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


def advance_state(
    state: ConversationState, tool_outputs: list[dict]
) -> ConversationState:
    return ConversationState(
        input=tool_outputs,
        prev_response_id=state.prev_response_id,
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
