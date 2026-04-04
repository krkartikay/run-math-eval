import logging
import os
import re

from openai import OpenAI
from lm_eval.api.model import LM
from tqdm import tqdm


SYSTEM = """You are being evaluated by an automatic parser.

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


class OpenAINanoMathLM(LM):
    """
    Minimal lm_eval wrapper around an OpenAI nano model.

    GPT-5 nano may reject the `stop` parameter, so we generate normally and
    apply lm_eval's stop strings client-side.
    """

    def __init__(self, model: str = "gpt-5.4-nano"):
        super().__init__()
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
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

    def generate_until(self, requests):
        outputs = []
        for req in tqdm(requests):
            prompt, decoding_config = req.args
            max_tokens = decoding_config.get("max_gen_toks", 512)

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=max_tokens,
            )

            text = resp.choices[0].message.content or ""
            outputs.append(self._extract_final_answer(text))
        return outputs
