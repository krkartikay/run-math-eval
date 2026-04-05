import asyncio
from contextlib import suppress
import json
import logging
import os
import re
import sys

import openai
from openai import AsyncOpenAI
from lm_eval.api.model import LM


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


class OpenAINanoMathLM(LM):
    """
    Minimal lm_eval wrapper around an OpenAI nano model.

    GPT-5 nano may reject the `stop` parameter, so we generate normally and
    apply lm_eval's stop strings client-side.
    """

    def __init__(self, model: str = "gpt-5.4-nano"):
        super().__init__()
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    @property
    def batch_size(self):
        return 1

    @property
    def max_length(self):
        return 128_000

    def loglikelihood(self, requests):
        raise NotImplementedError(
            "This minimal wrapper only supports generation tasks."
        )

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "This minimal wrapper only supports generation tasks."
        )

    def _extract_final_answer(self, text: str) -> str:
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

    async def _call_one(self, prompt, max_tokens):
        response_input = prompt
        previous_response_id = None
        request_kwargs = {
            "model": self.model,
            "instructions": SYSTEM,
            "input": response_input,
            "max_output_tokens": max_tokens,
            "reasoning": {"effort": "high"},
            "previous_response_id": previous_response_id,
        }
        if ALLOW_TOOL_CALLS:
            request_kwargs["tools"] = [CODE_INTERPRETER_TOOL]
            request_kwargs["tool_choice"] = "auto"

        for attempt in range(3):
            for _ in range(MAX_TOOL_CALLS):
                try:
                    request_kwargs["input"] = response_input
                    request_kwargs["previous_response_id"] = previous_response_id
                    resp = await self.client.responses.create(**request_kwargs)
                    if not resp.output:
                        logger.warning("API response had no output items: %r", resp)
                        return ""

                    previous_response_id = resp.id
                    tool_outputs = []
                    for output_item in resp.output:
                        if output_item.type != "function_call":
                            continue

                        tool_name = output_item.name
                        if tool_name != "python_code_interpreter":
                            tool_output = f"Unsupported tool call: {tool_name}"
                        else:
                            tool_output = await self._run_code_locally(
                                output_item.arguments,
                            )

                        tool_outputs.append(
                            {
                                "type": "function_call_output",
                                "call_id": output_item.call_id,
                                "output": tool_output,
                            }
                        )

                    if not tool_outputs:
                        if not resp.output_text:
                            logger.warning("API response had no output text: %r", resp)
                            return ""
                        return resp.output_text

                    response_input = tool_outputs
                except openai.BadRequestError as e:
                    logger.error(
                        "Bad request (will not retry). prompt: %r error: %s",
                        prompt,
                        e,
                    )
                    return ""
                except openai.AuthenticationError as e:
                    logger.error("Authentication failed, aborting: %s", e)
                    raise
                except (
                    openai.RateLimitError,
                    openai.APIStatusError,
                    openai.APIConnectionError,
                ) as e:
                    if attempt == 2:
                        logger.error("API call failed after 3 attempts: %s", e)
                        return ""
                    wait = 2**attempt
                    logger.warning(
                        "Retrying in %ds (attempt %d/3): %s",
                        wait,
                        attempt + 1,
                        e,
                    )
                    await asyncio.sleep(wait)
                    break

        logger.warning("Exceeded maximum tool-call loop count for prompt: %r", prompt)
        return ""

    async def _run_code_locally(self, raw_arguments: str) -> str:
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError as e:
            logger.warning("Invalid tool arguments: %r (%s)", raw_arguments, e)
            return f"Tool argument error: {e}"

        code = arguments.get("code", "")
        if not code.strip():
            return "Tool argument error: `code` must be a non-empty string."

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
                {
                    "stdout": stdout.decode("utf-8", errors="replace"),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                    "returncode": process.returncode,
                    "timed_out": True,
                },
                ensure_ascii=True,
            )
        except Exception as e:
            logger.exception("Local code execution failed")
            return f"Local code execution failed: {e}"

        payload = {
            "stdout": stdout.decode("utf-8", errors="replace"),
            "stderr": stderr.decode("utf-8", errors="replace"),
            "returncode": process.returncode,
            "timed_out": False,
        }
        return json.dumps(payload, ensure_ascii=True)

    def generate_until(self, requests):
        async def run_all():
            tasks = [
                self._call_one(
                    req.args[0],
                    req.args[1].get("max_gen_toks", MAX_TOKENS),
                )
                for req in requests
            ]
            return await asyncio.gather(*tasks)

        texts = asyncio.run(run_all())
        return [self._extract_final_answer(t) for t in texts]
