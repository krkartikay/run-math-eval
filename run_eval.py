import json
import logging
import lm_eval
from uuid import uuid4

from dotenv import load_dotenv
from lm_eval.tasks import TaskManager

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
            "tasks": ["aime25"],
            "num_fewshot": 0,
            "limit": 100,
            "log_samples": True,
        },
    )
    logging.info("Started eval run %s", eval_run_id)
    lm = OpenAINanoMathLM(model="gpt-5.4-nano", eval_run_id=eval_run_id)
    task_manager = TaskManager(include_path="tasks")

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["aime25"],
        num_fewshot=0,
        limit=10,  # small smoke test; remove or raise later
        log_samples=True,
        task_manager=task_manager,
    )

    print(json.dumps(results["results"], indent=2))


if __name__ == "__main__":
    main()
