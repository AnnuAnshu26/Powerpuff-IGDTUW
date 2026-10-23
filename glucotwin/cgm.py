"""Stream 2 (dynamic / real-time): simulated wearable + CGM time series at 5-minute resolution.

Mimics a continuous glucose monitor, a smartwatch (steps, heart rate) and a logbook
(meals, insulin). Physiology is driven by each patient's EHR so the two streams are linked.
"""
import numpy as np
import pandas as pd
from .ehr import physiology

STEPS_PER_DAY = 288


def _kernel(tau, length):
    k = np.arange(length)
    return k / tau * np.exp(1 - k / tau)


def simulate_patient(row, days: int = 14, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ph = physiology(row, rng)
    n = days * STEPS_PER_DAY
    carbs, ins, steps = np.zeros(n), np.zeros(n), np.zeros(n)
    for d in range(days):
        b = d * STEPS_PER_DAY
        for t0, mu in [(90, 57), (156, 65), (234, 65)]:
            t = b + t0 + int(round(rng.normal(0, 4)))
            cb = round(mu + rng.uniform(-12, 12)); carbs[t] += cb
            dose = cb / ph["icr"] * (1 + rng.normal(0, .18)) * (1.35 if rng.random() < .18 else 1)
            ins[t] += round(dose * 2) / 2
        if rng.random() < .4:
            carbs[b + 190 + int(rng.integers(0, 20))] += 20
        if rng.random() < .75:
            t = b + (170 if rng.random() < .5 else 205) + int(rng.integers(0, 15))
            steps[t:t + 6 + int(rng.integers(0, 4))] = 350 + rng.random() * 250
    kc, ki = _kernel(12, 150), _kernel(ph["tau_ins"], 150 if ph["tau_ins"] < 28 else 200)
    eff = np.convolve(carbs * ph["cg"], kc)[:n] - np.convolve(ins * ph["ig"], ki)[:n]
    act = np.convolve(steps, np.ones(18))[:n] * 0.008
    noise = np.zeros(n); e = rng.normal(0, 2.2, n)
    for i in range(1, n):
        noise[i] = .93 * noise[i - 1] + e[i]
    tod = (np.arange(n) % STEPS_PER_DAY) / STEPS_PER_DAY
    g = np.clip(ph["base"] + 8 * np.sin(tod * 2 * np.pi - 1.2) + eff - act + 2 * noise, 40, 350)
    hr = 62 + 0.035 * np.convolve(steps, np.ones(3))[:n] + 4 * (carbs > 0) + rng.normal(0, 2, n)
    hour = tod * 24
    sleeping = ((hour >= 22.5) | (hour < 6.5)).astype(int)
    return pd.DataFrame({"glucose": g, "carbs": carbs, "insulin": ins, "steps": steps,
                         "heart_rate": np.where(sleeping == 1, hr - 8, hr), "sleeping": sleeping, "tod": tod})
