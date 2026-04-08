from __future__ import annotations

from typing import Any

from math_verify import parse, verify


def math_verify_filter(
    resps: list[list[str]], docs: list[dict[str, Any]]
) -> list[list[str]]:
    filtered_resps: list[list[str]] = []

    for resp_group, doc in zip(resps, docs, strict=True):
        gold_solution = doc.get("solution")
        gold_answer = doc.get("answer")

        if not isinstance(gold_solution, str) or not isinstance(gold_answer, str):
            filtered_resps.append(resp_group)
            continue

        verified_group: list[str] = []
        parsed_gold = parse(gold_solution)

        for resp in resp_group:
            try:
                if verify(parsed_gold, parse(resp)):
                    verified_group.append(gold_answer)
                else:
                    verified_group.append(resp)
            except Exception:
                verified_group.append(resp)

        filtered_resps.append(verified_group)

    return filtered_resps
