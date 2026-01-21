"""Aggregate robustness scores across corruptions."""
def aggregate(results):
    # results: dict of corruption -> metric
    return sum(results.values()) / max(1, len(results))
