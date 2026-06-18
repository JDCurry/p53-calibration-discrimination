# p53 fate-decision calibration study

Computational analysis asking whether the p53 DNA-damage fate decision is
*calibrated* to true cellular recoverability — not just whether it *discriminates*
recoverable from unrecoverable cells. Built entirely on published models; dry-lab only.

## The seam (read first)

This code has two clearly separated layers:

- **Borrowed and validated** — `generator.py` reimplements the Batchelor et al. 2008
  recurrent-initiation DDE model (Mol Cell 30:277), parameters from its Supplemental
  Table S1. It reproduces the paper's Figure 4B (sustained ~6 h pulses) and the digital
  count code (pulse number scales with damage persistence). `python generator.py` runs
  the validation check.

- **Introduced here and validated against nothing** — `repair.py` (independent
  ground-truth repair process + recoverability labels), `fate.py` (two-timescale
  effector readout), and the calibration scoring in `experiment.py`. The entire
  calibration result lives in this layer. Any writeup must say so plainly; do not
  present the added layer as if it carried the generator's empirical grounding.

## Two design choices deliberately left as variables

Rather than hard-coding contestable assumptions, the study sweeps over them:

1. **Damage encoder** (`--encoders`): `recurrent` (Batchelor persistence code),
   `linear` (Loewer 2013, Signal ∝ break count), `saturating` (Kim & Jackson 2013,
   Hill function of breaks). These three disagree in the literature; calibration is
   reported under each.
2. **Recoverability metric** (all three scored automatically): `any_misrepair`,
   `misrepair_burden`, `persistence`. The headline number moves with this choice, so
   it is an axis of the study, not a buried constant.

## Run

```
pip install numpy scipy matplotlib
python generator.py                      # validation check (should print ~8 pulses, ~6 h)
python experiment.py --n 2000 --plot     # full sweep, writes reliability_grid.png
```

Useful flags: `--n` cells per encoder, `--dt` timestep (0.002 for real runs, 0.02 for
quick checks), `--theta_C` and `--g_slow` (the commitment-arm knobs), `--seed`.

## Tuning before the real run

The smoke-test defaults over-commit (commit rate near 1.0) because `theta_C`/`g_slow`
are untuned. NOTE: `theta_C` only affects the reported `commit` column; it does NOT
enter the calibration scoring (ECE/disc are computed from the continuous commitment
score via a logistic fit). So calibration results are threshold-independent and you do
not need to tune `theta_C` to trust them.

Each (encoder x metric) cell now reports a bootstrap 95% CI on discrimination (300
resamples over cells) and the positive-class count `n_pos`. Rows where the disc CI
spans 0 are flagged; treat their discrimination as indistinguishable from zero rather
than reporting the point estimate (this is what rescues the spurious negative
misrepair_burden values seen in early runs). The misrepair_hazard sweep range was
widened to 0.005-0.06 so the rare misrepair labels are adequately populated.

## What a result looks like

The defensible claim is *not* "p53 is miscalibrated." It is: under recoverability
metric M and damage encoder E, the commitment decision deviates from true
unrecoverability by ECE = X, and the deviation is governed by (g_slow / repair-rate)
and inherited from the encoder. The negative result — "for parameter region R the
system is well-calibrated" — is equally reportable. If apparent miscalibration is
large, consider that the system may be calibrated to a different objective
(organism-level cancer risk vs cell-level recoverability) rather than broken.

## Files
- `generator.py` — validated pulse generator (3 encoders)
- `repair.py` — independent ground-truth repair + recoverability labels
- `fate.py` — two-timescale fate readout
- `experiment.py` — sweep + calibration (ECE/Brier/discrimination) + reliability grid

## Key references
- Batchelor et al. 2008, Mol Cell 30:277 (generator + Table S1 params)
- Purvis et al. 2012, Science 336:1440 (dynamics control fate)
- Hanson/Porter/Batchelor 2019, JCB 218:1282 (target decay kinetics → readout split)
- Kim & Jackson 2013, PLoS ONE 8:e65242 (saturating ATM encoder)
- Loewer et al. 2013, BMC Biol 11:114 (linear, thresholdless break encoding)
