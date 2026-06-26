"""Offline per-block bit allocation (the MixDQ-style init).

Spends high bits on high-sensitivity blocks under an average-bit budget.
Greedy is dependency-free and exact for the two-level {4,8} case; an optional
ILP via PuLP is provided for >2 levels.
"""


def allocate_bits_greedy(sensitivity, cost, legal_bits, target_avg_bit):
    """
    sensitivity: {block_id: float}  (higher = more sensitive, e.g. quant MSE)
    cost:        {block_id: float}  (relative compute weight, e.g. BitOPs at 1-bit)
    legal_bits:  iterable of ints, e.g. (4, 8)
    target_avg_bit: cost-weighted average bit budget, e.g. 5.0
    returns: {block_id: bit}
    """
    legal = sorted(int(b) for b in legal_bits)
    lo, hi = legal[0], legal[-1]
    blocks = list(sensitivity.keys())
    bits = {k: lo for k in blocks}

    total_cost = sum(cost[k] for k in blocks)
    budget = target_avg_bit * total_cost  # cost-weighted bit budget

    # rank by sensitivity per unit upgrade cost; promote toward hi until budget spent
    def spent():
        return sum(cost[k] * bits[k] for k in blocks)

    order = sorted(blocks, key=lambda k: sensitivity[k] / (cost[k] + 1e-12), reverse=True)
    for k in order:
        for b in legal[1:]:
            extra = cost[k] * (b - bits[k])
            if spent() + extra <= budget:
                bits[k] = b
            else:
                break
    return bits


def allocate_bits_ilp(sensitivity, cost, legal_bits, target_avg_bit):
    """Exact ILP (requires `pip install pulp`). Falls back to greedy if unavailable."""
    try:
        import pulp
    except Exception:
        return allocate_bits_greedy(sensitivity, cost, legal_bits, target_avg_bit)

    legal = sorted(int(b) for b in legal_bits)
    blocks = list(sensitivity.keys())
    total_cost = sum(cost[k] for k in blocks)
    budget = target_avg_bit * total_cost

    prob = pulp.LpProblem("bit_alloc", pulp.LpMaximize)
    x = {(k, b): pulp.LpVariable(f"x_{k}_{b}", cat="Binary") for k in blocks for b in legal}
    # objective: reward bits where sensitivity is high
    prob += pulp.lpSum(sensitivity[k] * b * x[(k, b)] for k in blocks for b in legal)
    for k in blocks:
        prob += pulp.lpSum(x[(k, b)] for b in legal) == 1
    prob += pulp.lpSum(cost[k] * b * x[(k, b)] for k in blocks for b in legal) <= budget
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    out = {}
    for k in blocks:
        out[k] = max(legal, key=lambda b: pulp.value(x[(k, b)]) or 0)
    return out


def avg_bit(bits, cost):
    tot = sum(cost[k] for k in bits)
    return sum(cost[k] * bits[k] for k in bits) / (tot + 1e-12)
