"""
Main experiment: sweep a heterogeneous cell population, run each cell through the
validated pulse generator under a chosen damage encoder, apply the fate readout, and
score calibration of the commitment decision against an independent recoverability
label -- for every (encoder x metric) combination.

Outputs a reliability table and ECE/Brier per combination, plus reliability-diagram
PNGs. Run `python experiment.py --help` for options.

SEAM REMINDER: generator.py is borrowed/validated; repair.py, fate.py, and the
recoverability labels are introduced here and validated against nothing. The
calibration result is a property of this added layer, not of real cells.
"""
import argparse
import numpy as np

from generator import simulate
from repair import simulate_repair, RECOVERABILITY_METRICS
from fate import fate_readout


def run_population(n_cells, encoder, dt, T, fate_kwargs, seed=0):
    rng = np.random.default_rng(seed)
    rows = []  # (C_max, committed, labels_dict)
    for i in range(n_cells):
        D0 = int(rng.integers(2, 40))
        r = rng.uniform(0.05, 0.6)
        mh = rng.uniform(0.005, 0.06)  # widened so misrepair labels are better populated
        breaks_fn, rt, labels = simulate_repair(D0, r, mh, T=T, dt=dt, rng=rng)
        t, h = simulate(T=T, dt=dt, breaks_fn=breaks_fn, encoder=encoder)
        fr = fate_readout(t, h[:, 1], **fate_kwargs)
        rows.append((fr['C_max'], fr['committed'], labels))
        if (i + 1) % 50 == 0:
            print(f"  [{encoder}] {i+1}/{n_cells} cells")
    return rows


def calibrate(rows, metric):
    """Logistic-map commitment score C to probability, then ECE/Brier vs label."""
    from scipy.optimize import minimize
    C = np.array([row[0] for row in rows])
    y = np.array([row[2][metric] for row in rows])
    if y.sum() == 0 or y.sum() == len(y):
        return None  # degenerate; widen sweep
    from scipy.special import expit  # numerically stable logistic, no overflow warning
    xs = (C - C.mean()) / (C.std() + 1e-9)

    def fit_probs(C_arr, y_arr):
        xs_ = (C_arr - C_arr.mean()) / (C_arr.std() + 1e-9)

        def nll(p):
            pr_ = np.clip(expit(p[0] + p[1] * xs_), 1e-6, 1 - 1e-6)
            return -np.sum(y_arr * np.log(pr_) + (1 - y_arr) * np.log(1 - pr_))

        r = minimize(nll, [0.0, 1.0])
        return expit(r.x[0] + r.x[1] * xs_)

    def ece_of(pr_, y_arr):
        bins_ = np.linspace(0, 1, 11)
        e = 0.0
        for i in range(10):
            hi = bins_[i + 1] if i < 9 else 1.0 + 1e-9
            m = (pr_ >= bins_[i]) & (pr_ < hi)
            if m.sum() == 0:
                continue
            e += (m.sum() / len(y_arr)) * abs(pr_[m].mean() - y_arr[m].mean())
        return e

    pr = fit_probs(C, y)

    bins = np.linspace(0, 1, 11)
    ece = 0.0; table = []
    for i in range(10):
        m = (pr >= bins[i]) & (pr < (bins[i + 1] if i < 9 else 1.0 + 1e-9))
        if m.sum() == 0:
            continue
        conf = pr[m].mean(); acc = y[m].mean(); w = m.sum() / len(y)
        ece += w * abs(conf - acc)
        table.append((bins[i], conf, acc, int(m.sum())))
    brier = float(np.mean((pr - y) ** 2))
    disc = float(np.corrcoef(C, y)[0, 1])
    commit_rate = float(np.mean([row[1] for row in rows]))

    # Bootstrap CIs over cells (resample with replacement)
    n_boot = 300
    rb = np.random.default_rng(12345)
    ece_b = np.empty(n_boot); disc_b = np.empty(n_boot)
    for b in range(n_boot):
        idx = rb.integers(0, len(y), len(y))
        Cb, yb = C[idx], y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            ece_b[b] = np.nan; disc_b[b] = np.nan; continue
        prb = fit_probs(Cb, yb)
        ece_b[b] = ece_of(prb, yb)
        disc_b[b] = np.corrcoef(Cb, yb)[0, 1]
    ece_lo, ece_hi = np.nanpercentile(ece_b, [2.5, 97.5])
    disc_lo, disc_hi = np.nanpercentile(disc_b, [2.5, 97.5])

    return dict(ece=ece, brier=brier, disc=disc, commit_rate=commit_rate,
                true_rate=float(y.mean()), table=table, pr=pr, y=y,
                n_pos=int(y.sum()),
                ece_ci=(float(ece_lo), float(ece_hi)),
                disc_ci=(float(disc_lo), float(disc_hi)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000, help='cells per encoder')
    ap.add_argument('--dt', type=float, default=0.002)
    ap.add_argument('--T', type=float, default=72.0)
    ap.add_argument('--encoders', nargs='+',
                    default=['recurrent', 'linear', 'saturating'])
    ap.add_argument('--theta_C', type=float, default=6.0)
    ap.add_argument('--g_slow', type=float, default=0.06)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--plot', action='store_true')
    args = ap.parse_args()

    fate_kwargs = dict(theta_C=args.theta_C, g_slow=args.g_slow)
    print(f"Sweep: {args.n} cells x {len(args.encoders)} encoders, dt={args.dt}, T={args.T}")
    print(f"Fate knobs: theta_C={args.theta_C}, g_slow={args.g_slow}\n")

    results = {}
    for enc in args.encoders:
        rows = run_population(args.n, enc, args.dt, args.T, fate_kwargs, seed=args.seed)
        for metric in RECOVERABILITY_METRICS:
            res = calibrate(rows, metric)
            results[(enc, metric)] = res

    print("\n=== Calibration summary (encoder x recoverability metric) ===")
    print(f"{'encoder':<12}{'metric':<18}{'ECE':>6}{'disc':>7}"
          f"{'disc 95% CI':>18}{'n_pos':>7}{'commit':>8}")
    for (enc, metric), res in results.items():
        if res is None:
            print(f"{enc:<12}{metric:<18}{'(degenerate label)':>30}")
            continue
        ci = res['disc_ci']
        flag = "  <- CI spans 0" if ci[0] < 0 < ci[1] else ""
        print(f"{enc:<12}{metric:<18}{res['ece']:>6.3f}{res['disc']:>7.3f}"
              f"   [{ci[0]:>6.3f},{ci[1]:>6.3f}]{res['n_pos']:>7}"
              f"{res['commit_rate']:>8.3f}{flag}")

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        encs = args.encoders
        fig, axes = plt.subplots(len(encs), len(RECOVERABILITY_METRICS),
                                 figsize=(4*len(RECOVERABILITY_METRICS), 3.2*len(encs)),
                                 squeeze=False)
        for i, enc in enumerate(encs):
            for j, metric in enumerate(RECOVERABILITY_METRICS):
                ax = axes[i][j]; res = results[(enc, metric)]
                ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
                if res:
                    tb = np.array(res['table'])
                    if len(tb):
                        ax.plot(tb[:, 1], tb[:, 2], 'o-', ms=4)
                    ax.set_title(f"{enc}/{metric}\nECE={res['ece']:.3f}", fontsize=8)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                if i == len(encs)-1: ax.set_xlabel('model p', fontsize=8)
                if j == 0: ax.set_ylabel('true freq', fontsize=8)
        plt.tight_layout(); plt.savefig('reliability_grid.png', dpi=120)
        print("\nsaved reliability_grid.png")


if __name__ == "__main__":
    main()
