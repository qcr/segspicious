"""Train all (model × scenario) candidates.

Iterates the cross product of models and scenarios, calling
``train_or_load`` for each.  Already-trained candidates are loaded
from cache; new ones are trained and checkpointed.

Usage:
    python -m experiments.benchmarking.train
"""

from __future__ import annotations

import segspicious
from experiments.benchmarking.models import get_models
from experiments.benchmarking.scenarios import get_scenarios
from segspicious import train_or_load

REPO_ID = "alistair-english/segspicious-checkpoints"


def main() -> None:
    segspicious.configure(repo_id=REPO_ID)
    scenarios = get_scenarios()

    print(f"=== Training candidates ({len(scenarios)} scenario(s)) ===\n")

    for scenario in scenarios:
        print(f"Scenario: {scenario.name}")
        print(f"  train: {scenario.train.name}")
        print(f"  val:   {scenario.val.name if scenario.val else '(none)'}")
        print(f"  test:  {scenario.test.name}")
        print()

        for model in get_models():
            print(f"  [{model.name}] on [{scenario.train.name}]")

            candidate = train_or_load(
                model,
                scenario.train,
                validation_data=scenario.val,
            )

            print(f"    → Candidate ready: {candidate.name}")
            print()

    print("=== All candidates ready ===")


if __name__ == "__main__":
    main()
