"""
Independent ground-truth repair process. NEW / UNVALIDATED layer.

This process is computed entirely separately from p53 -- p53 never sees it and it
never sees p53. That separation is what makes "is the fate decision calibrated to
true recoverability?" a meaningful question rather than a tautology.

Output per cell:
  - breaks_fn(t): step function giving open-break count over time (fed to the generator)
  - repair_time: when all breaks resolved (or T if never)
  - labels: dict of unrecoverability under SEVERAL metrics (see RECOVERABILITY_METRICS)

The recoverability metric is deliberately NOT fixed. The headline calibration number
moves with this choice, so we expose it as an independent variable and report how
calibration depends on it, rather than smuggling in one definition.
"""
import numpy as np

RECOVERABILITY_METRICS = ('any_misrepair', 'misrepair_burden', 'persistence')


def simulate_repair(D0, r, misrepair_hazard, T=72.0, dt=0.002,
                    deadline=24.0, burden_frac=0.25, rng=None):
    """Returns (breaks_fn, repair_time, labels_dict).

    D0: initial double-strand breaks. r: per-break correct-repair rate (/h).
    misrepair_hazard: per-break per-hour hazard of irreversible misrepair (mutation).
    deadline: hours past which unresolved breaks count as failure (persistence metric).
    burden_frac: fraction of initial breaks misrepaired above which 'misrepair_burden' fires.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = int(T / dt)
    open_breaks = int(D0)
    misrepaired = 0
    repair_time = T
    # record break count on a coarse grid for the breaks_fn step function
    grid_t = np.linspace(0, T, n + 1)
    grid_b = np.zeros(n + 1)
    for k in range(n):
        grid_b[k] = open_breaks
        if open_breaks <= 0:
            repair_time = k * dt
            grid_b[k:] = 0
            break
        p_rep = 1 - np.exp(-r * open_breaks * dt)
        if rng.random() < p_rep:
            open_breaks -= 1
        p_mis = 1 - np.exp(-misrepair_hazard * open_breaks * dt)
        if rng.random() < p_mis:
            misrepaired += 1
            open_breaks = max(0, open_breaks - 1)
    else:
        grid_b[-1] = open_breaks

    def breaks_fn(t):
        idx = min(int(t / dt), n)
        return grid_b[idx]

    labels = {
        'any_misrepair':    int((misrepaired > 0) or (open_breaks > 0)),
        'misrepair_burden': int(misrepaired >= max(1, burden_frac * D0)),
        'persistence':      int(repair_time > deadline),
    }
    return breaks_fn, repair_time, labels
