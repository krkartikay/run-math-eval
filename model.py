from dotenv import load_dotenv
import os

from openai import OpenAI
from lm_eval.api.model import LM


SYSTEM = """You are being evaluated by an automatic parser.

Return only the final answer.
Do not show steps.
Do not include explanations.
Do not include words like "Answer:".
Output exactly one mathematical expression and nothing else.
Prefer plain canonical forms like:
2
\\frac{3}{4}
\\sqrt{74}
"""


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

    def generate_until(self, requests):
        outputs = []
        print(f"Requests: {requests}")
        for req in requests:
            prompt, decoding_config = req.args
            stop = decoding_config.get("until") or None
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
            outputs.append(text)
        print(f"Outputs: {outputs}")
        return outputs
