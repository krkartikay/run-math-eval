import json
import logging
import lm_eval
from uuid import uuid4

from dotenv import load_dotenv

from model import OpenAINanoMathLM
from trace_store import TraceStore


def main():
    load_dotenv()  # reads .env
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    eval_run_id = TraceStore.create_eval_run(
        eval_run_id=f"eval_{uuid4().hex}",
        metadata={
            "tasks": ["hendrycks_math"],
            "num_fewshot": 0,
            "limit": 10,
            "log_samples": True,
        },
    )
    logging.info("Started eval run %s", eval_run_id)
    lm = OpenAINanoMathLM(eval_run_id=eval_run_id)

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["hendrycks_math"],
        num_fewshot=0,
        limit=10,  # small smoke test; remove or raise later
        log_samples=True,
    )

    print(json.dumps(results["results"], indent=2))


if __name__ == "__main__":
    main()
