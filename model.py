import asyncio
import logging
import os
import re

import openai
from openai import AsyncOpenAI
from lm_eval.api.model import LM


SYSTEM = """You are a Math problem solver. Your task is to solve math problems.

Solve the problem step by step.
On a separate final line, write exactly:
final answer: <answer>

Keep the final answer concise and in canonical mathematical form when possible, for example:
2
\\frac{3}{4}
\\sqrt{74}
"""


logger = logging.getLogger(__name__)
FINAL_ANSWER_RE = re.compile(r"final answer:\s*(.+)", re.IGNORECASE)
MAX_TOKENS = 4096


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
        for attempt in range(3):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=max_tokens,
                )
                if not resp.choices or not resp.choices[0].message.content:
                    logger.warning("API response had no content: %r", resp)
                    return ""
                return resp.choices[0].message.content
            except openai.BadRequestError as e:
                logger.error("Bad request (will not retry). prompt: %r error: %s", prompt, e)
                return ""
            except openai.AuthenticationError as e:
                logger.error("Authentication failed, aborting: %s", e)
                raise
            except (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
                if attempt == 2:
                    logger.error("API call failed after 3 attempts: %s", e)
                    return ""
                wait = 2 ** attempt
                logger.warning("Retrying in %ds (attempt %d/3): %s", wait, attempt + 1, e)
                await asyncio.sleep(wait)

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
