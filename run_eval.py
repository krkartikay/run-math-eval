import json
import logging
import lm_eval

from dotenv import load_dotenv

from model import OpenAINanoMathLM


def main():
    load_dotenv()  # reads .env
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    lm = OpenAINanoMathLM()

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
