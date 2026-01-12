"""Competition-level scoring aggregation."""
def aggregate(scores, weights):
    total = 0.0
    for k, v in scores.items():
        w = weights.get(k, 0.0)
        total += w * v
    return total
