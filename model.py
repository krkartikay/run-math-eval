import asyncio
import os

from openai import AsyncOpenAI
from lm_eval.api.model import LM

from state_machine import ResponseStateMachine, extract_final_answer


class OpenAINanoMathLM(LM):
    """
    Minimal lm_eval wrapper around an OpenAI nano model.

    GPT-5 nano may reject the `stop` parameter, so we generate normally and
    apply lm_eval's stop strings client-side.
    """

    def __init__(self, model: str = "gpt-5.4-nano"):
        super().__init__()
        self.state_machine = ResponseStateMachine(
            client=AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]),
            model=model,
        )

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

    async def _call_one(self, prompt: str) -> str:
        return await self.state_machine.solve(prompt)

    def generate_until(self, requests):
        async def run_all():
            return await asyncio.gather(*(self._call_one(req.args[0]) for req in requests))

        return [extract_final_answer(text) for text in asyncio.run(run_all())]
