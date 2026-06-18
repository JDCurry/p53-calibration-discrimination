"""
Fate-readout layer. NEW / UNVALIDATED.

Two effector arms over the validated p53active trace, following the decay-kinetics
result of Hanson/Porter/Batchelor 2019 (JCB): arrest-arm targets have fast mRNA/protein
decay (track pulses, reset between them); commitment-arm targets have slow decay
(integrate across pulses, do not reset). Commitment fires when the slow arm crosses
threshold. These four parameters are the knobs the calibration study turns.
"""
import numpy as np


def fate_readout(times, p53active, k_A=1.0, g_fast=1.5,
                 k_C=1.0, g_slow=0.06, theta_C=6.0):
    """Returns dict with arrest peak A_max, commitment score C_max, committed bool."""
    dt = times[1] - times[0]
    A = 0.0; C = 0.0; A_max = 0.0; C_max = 0.0
    for k in range(len(p53active) - 1):
        A = max(0.0, A + dt * (k_A * p53active[k] - g_fast * A))
        C = max(0.0, C + dt * (k_C * p53active[k] - g_slow * C))
        A_max = max(A_max, A); C_max = max(C_max, C)
    return dict(A_max=A_max, C_max=C_max, committed=int(C_max > theta_C))
