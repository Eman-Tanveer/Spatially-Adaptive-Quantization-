"""Online budgeted bit policy.

Replaces AdaBM's global +/-1 scalar. Given the per-image latent complexity, it
chooses HOW MANY of the most sensitive blocks run at the high bit, keeping the
test-set average bit centered on the budget. Only hardware-legal bits {4,8} are
ever produced, so every realized config has a kernel.
"""


class BudgetedBitPolicy:
    def __init__(self, sensitivity, cost, legal_bits=(4, 8),
                 target_avg_bit=5.0, swing=0.15, comp_lo=None, comp_hi=None):
        """
        sensitivity/cost: {block_id: float}
        target_avg_bit:   cost-weighted budget the test set should average to
        swing:            +/- fraction of blocks that may flip vs the base config
        comp_lo, comp_hi: complexity-score clamp range (set from calibration)
        """
        self.legal = sorted(int(b) for b in legal_bits)
        self.lo_bit, self.hi_bit = self.legal[0], self.legal[-1]
        self.cost = dict(cost)
        # blocks ranked most -> least sensitive: promotions go to the top of this list
        self.order = sorted(sensitivity.keys(), key=lambda k: sensitivity[k], reverse=True)
        self.target_avg_bit = target_avg_bit
        self.comp_lo = comp_lo
        self.comp_hi = comp_hi

        total = sum(cost.values())
        # base number of high-bit blocks that hits the budget on average
        self._base_n8 = self._n8_for_budget(target_avg_bit, total)
        delta = max(1, int(round(swing * len(self.order))))
        self.n8_min = max(0, self._base_n8 - delta)
        self.n8_max = min(len(self.order), self._base_n8 + delta)

    def _n8_for_budget(self, avg_bit, total):
        # greedily count high-bit blocks (top of order) until cost-weighted avg ~ budget
        budget = avg_bit * total
        spent = self.lo_bit * total
        n = 0
        for k in self.order:
            extra = self.cost[k] * (self.hi_bit - self.lo_bit)
            if spent + extra <= budget:
                spent += extra
                n += 1
            else:
                break
        return n

    def set_complexity_range(self, comp_lo, comp_hi):
        self.comp_lo, self.comp_hi = comp_lo, comp_hi
        return self

    def assign(self, complexity_value):
        """complexity_value: scalar from ComplexityScore.score -> {block_id: bit}."""
        if self.comp_lo is None or self.comp_hi == self.comp_lo:
            p = 0.5
        else:
            p = (float(complexity_value) - self.comp_lo) / (self.comp_hi - self.comp_lo)
            p = min(1.0, max(0.0, p))
        n8 = int(round(self.n8_min + p * (self.n8_max - self.n8_min)))
        promoted = set(self.order[:n8])
        return {k: (self.hi_bit if k in promoted else self.lo_bit) for k in self.order}

    def realized_avg_bit(self, bits):
        tot = sum(self.cost[k] for k in bits)
        return sum(self.cost[k] * bits[k] for k in bits) / (tot + 1e-12)
