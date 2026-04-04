import json
import lm_eval

from dotenv import load_dotenv

from model import OpenAINanoMathLM


def main():
    load_dotenv()  # reads .env
    lm = OpenAINanoMathLM()

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["hendrycks_math"],
        num_fewshot=0,
        limit=1,  # small smoke test; remove or raise later
        log_samples=True,
    )

    print(json.dumps(results["results"], indent=2))


if __name__ == "__main__":
    main()
