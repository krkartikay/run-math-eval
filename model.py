import asyncio
import os

from openai import AsyncOpenAI
from lm_eval.api.model import LM

from state_machine import EvalTarget, ResponseStateMachine, extract_final_answer
from trace_store import TraceStore


class OpenAINanoMathLM(LM):
    """
    Minimal lm_eval wrapper around an OpenAI nano model.

    GPT-5 nano may reject the `stop` parameter, so we generate normally and
    apply lm_eval's stop strings client-side.
    """

    def __init__(self, model: str = "gpt-5.4-nano", eval_run_id: str | None = None):
        super().__init__()
        self.eval_run_id = eval_run_id or TraceStore.create_eval_run(
            metadata={"model": model}
        )
        self.state_machine = ResponseStateMachine(
            client=AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]),
            model=model,
            eval_run_id=self.eval_run_id,
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

    async def _call_one(
        self, prompt: str, eval_target: EvalTarget | None = None
    ) -> str:
        return await self.state_machine.solve(prompt, eval_target=eval_target)

    def generate_until(self, requests):
        def build_eval_target(request) -> EvalTarget:
            doc = dict(getattr(request, "doc", {}) or {})
            prompt_text = _extract_prompt_text(request, doc)
            return EvalTarget(
                task_name=getattr(request, "task_name", None),
                doc_id=getattr(request, "doc_id", None),
                prompt_text=prompt_text,
                expected_answer=_extract_expected_answer(doc),
                doc=doc,
                metadata={
                    "request_index": getattr(request, "idx", None),
                    "request_type": getattr(request, "request_type", None),
                },
            )

        async def run_all():
            return await asyncio.gather(
                *(
                    self._call_one(req.args[0], eval_target=build_eval_target(req))
                    for req in requests
                )
            )

        return [extract_final_answer(text) for text in asyncio.run(run_all())]


def _extract_prompt_text(request, doc: dict) -> str | None:
    args = getattr(request, "args", ())
    if args and isinstance(args[0], str):
        return args[0]

    for key in ("problem", "question", "prompt", "query"):
        value = doc.get(key)
        if isinstance(value, str):
            return value
    return None


def _extract_expected_answer(doc: dict) -> str | None:
    for key in (
        "answer",
        "solution",
        "boxed",
        "final_answer",
        "target",
        "label",
        "gold",
    ):
        value = doc.get(key)
        if isinstance(value, str):
            return value
        if value is not None:
            return str(value)
    return None
