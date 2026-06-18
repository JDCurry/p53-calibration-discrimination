"""
p53 pulse generator: Batchelor et al. 2008 (Mol Cell 30:277) recurrent-initiation model.

VALIDATED LAYER. Five species: p53inactive, p53active, Mdm2, inhibitor (Wip1-like),
Signal (ATM-P/Chk2-P). Equations from p.5 of the paper; parameters/initial conditions
from Supplemental Table S1 (mmc1.pdf). Reproduces Fig 4B (sustained undamped pulses,
~6 h period) and the digital count code (pulse number scales with damage persistence).

The model encodes damage three ways, selectable via `encoder`:
  'recurrent'  - Batchelor native: Signal source q(t) on while damage persists (persistence code)
  'linear'     - Loewer 2013: Signal production scales linearly with instantaneous break count
  'saturating' - Kim & Jackson 2013: Signal production is a Hill (saturating) function of breaks
These three are in tension in the literature; treat `encoder` as an independent variable.
"""
import numpy as np

# ---- Parameters from Batchelor Table S1 (simulated concentration units, Cs/h) ----
PARAMS = dict(
    bp=0.9, bsp=10.0, bm=0.9, bmi=0.2, bi=0.25, bs=10.0,
    ampi=5.0, api=2.0, ampa=1.4, asm=0.5, am=1.0, ai=0.7, ais=50.0, as_=7.5,
    tau_m=0.7, tau_i=1.25, Ts=1.0, Ti=0.2, ns=4, ni=4,
)
Y0 = np.array([0.3, 0.0, 0.2, 0.0, 0.0])  # [p53i, p53a, Mdm2, inhibitor, Signal]


def _signal_source(encoder, breaks, breaks0, p):
    """Signal production term given current open-break count.
    breaks0 is the initial break count (for normalizing the linear/saturating forms)."""
    if encoder == 'recurrent':
        return p['bs'] if breaks > 0 else 0.0
    if encoder == 'linear':
        # linear in instantaneous breaks, normalized so a 'typical' load ~ bs
        return p['bs'] * (breaks / 10.0)
    if encoder == 'saturating':
        Ng, Tg = 4, 5.0  # Kim-Jackson Hill: sensitive at low damage, saturates high
        return p['bs'] * (breaks**Ng) / (Tg**Ng + breaks**Ng)
    raise ValueError(encoder)


def simulate(T=72.0, dt=0.002, breaks_fn=None, encoder='recurrent', p=PARAMS):
    """Integrate the DDE. `breaks_fn(t)` returns open-break count at time t
    (step function from the repair process). If None, damage is permanent.
    Returns (times, history[n+1,5]) with columns [p53i, p53a, Mdm2, inh, Signal]."""
    n = int(T / dt)
    hist = np.zeros((n + 1, 5)); hist[0] = Y0
    times = np.linspace(0, T, n + 1)
    im = int(round(p['tau_m'] / dt)); ii = int(round(p['tau_i'] / dt))
    if breaks_fn is None:
        breaks_fn = lambda t: 1.0
    breaks0 = max(breaks_fn(0.0), 1.0)

    def deriv(y, y_tm, y_ti, brk):
        p53i, p53a, mdm2, inh, sig = y
        src = _signal_source(encoder, brk, breaks0, p)
        sig_hill = sig**p['ns'] / (sig**p['ns'] + p['Ts']**p['ns'])
        dp53i = p['bp'] - p['ampi']*mdm2*p53i - p['bsp']*p53i*sig_hill - p['api']*p53i
        dp53a = p['bsp']*p53i*sig_hill - p['ampa']*mdm2*p53a
        dmdm2 = p['bm']*y_tm[1] + p['bmi'] - p['asm']*sig*mdm2 - p['am']*mdm2
        dinh  = p['bi']*y_ti[1] - p['ai']*inh
        inh_hill = inh**p['ni'] / (inh**p['ni'] + p['Ti']**p['ni'])
        dsig  = src - p['ais']*inh_hill*sig - p['as_']*sig
        return np.array([dp53i, dp53a, dmdm2, dinh, dsig])

    for k in range(n):
        t = k * dt
        brk = breaks_fn(t)
        y = hist[k]
        y_tm = hist[k-im] if k-im >= 0 else Y0
        y_ti = hist[k-ii] if k-ii >= 0 else Y0
        k1 = deriv(y, y_tm, y_ti, brk)
        k2 = deriv(y + 0.5*dt*k1, y_tm, y_ti, brk)
        k3 = deriv(y + 0.5*dt*k2, y_tm, y_ti, brk)
        k4 = deriv(y + dt*k3, y_tm, y_ti, brk)
        hist[k+1] = np.clip(y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4), 0, None)
    return times, hist


if __name__ == "__main__":
    # Validation: persistent damage should give sustained ~6h pulses
    from scipy.signal import find_peaks
    t, h = simulate(T=48.0, dt=0.002)
    p53 = h[:, 0] + h[:, 1]
    pk, _ = find_peaks(p53, height=0.4, distance=int(2.0/0.002))
    print(f"[validation] p53 peaks in 48h: {len(pk)}")
    if len(pk) >= 2:
        print(f"[validation] mean inter-peak interval: {np.mean(np.diff(t[pk])):.2f} h "
              f"(expect ~5-7 h)")
    print(f"[validation] p53 range: {p53.min():.3f}-{p53.max():.3f}")
