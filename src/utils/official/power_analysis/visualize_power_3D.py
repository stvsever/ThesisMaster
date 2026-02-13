"""
3D Power surface: Power as a function of sample size (N participants) and expected effect size (Cohen's d)
Z-axis: power

PLUS: 2D plot of power vs number of items (m) for fixed N=100 and d=0.30 across multiple ICC (rho) values,
and it prints the minimum m needed to reach 80% power (if attainable within the searched range).

Key point about "number of items":
- Items affect power ONLY if your d is defined at the ITEM/TRIAL level and you aggregate items per participant.
- If your d is already defined on the PARTICIPANT MEAN level, items do NOT affect power (for a participant-level t-test).

This script supports both interpretations via EFFECT_DEFINED_AT:
- "item": d is item-level -> averaging m items increases d_eff -> power increases with m
- "participant_mean": d is participant-mean-level -> d_eff == d -> power does not depend on m

Test: two-sided one-sample t-test, power via noncentral t distribution.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import nct, t


# =========================
# User-configurable settings
# =========================

# Core test settings
ALPHA = 0.05  # two-sided

# Effect-size interpretation:
# - "item": d refers to item/trial-level standardized effect size
# - "participant_mean": d refers to participant-mean standardized effect size
EFFECT_DEFINED_AT = "item"  # <-- change if needed

# ---------- 3D surface settings (X=N, Y=d, Z=power) ----------
# Items per participant are FIXED for the 3D surface (items are not an axis in this plot)
M_ITEMS_3D = 30
RHO_3D = 0.30

# X-axis grid (sample size)
N_MIN, N_MAX, N_STEP = 20, 300, 2

# Y-axis grid (effect size)
D_MIN, D_MAX, D_STEP = 0.05, 0.60, 0.01

# ---------- 2D items curve settings (power vs m) ----------
N_FIXED = 100
D_FIXED = 0.30
TARGET_POWER = 0.80

M_MIN, M_MAX = 1, 400  # search range for "minimum items needed"

# Plot multiple rho curves for intuition (must include your main rho)
RHO_VALUES = [0.0, 0.1, 0.3, 0.5]
RHO_MAIN_FOR_PRINT = 0.30  # which rho to use for the "minimum items needed" printout

# ---------- Output ----------
OUTPUT_DIR = "/Users/stijnvanseveren/PythonProjects/MASTERPROEF/utils/official/power_analysis/visuals"
OUTPUT_3D_FILENAME = "power_surface_N_effect_power.png"
OUTPUT_2D_FILENAME = "power_vs_items_fixedN_fixedD.png"


# =========================
# Core math
# =========================

def d_effective(d: np.ndarray, m_items: int, rho: float, effect_defined_at: str) -> np.ndarray:
    """
    Map expected d to the participant-mean scale used in the participant-level t-test.

    If effect_defined_at == "participant_mean":
        d is already on participant means -> items do not change it.

    If effect_defined_at == "item":
        Assume item-level variance is standardized to 1 with ICC decomposition:
            between-person variance = rho
            within-person variance  = 1 - rho
        Participant mean variance:
            Var(mean) = rho + (1-rho)/m
        So participant-mean standardized effect:
            d_eff = d / sqrt(rho + (1-rho)/m)
    """
    if effect_defined_at == "participant_mean":
        return np.asarray(d, dtype=float)

    m = max(int(m_items), 1)
    denom = np.sqrt(rho + (1.0 - rho) / m)
    return np.asarray(d, dtype=float) / denom


def power_two_sided_one_sample_t(d: np.ndarray, n: np.ndarray, alpha: float) -> np.ndarray:
    """
    Two-sided one-sample t-test power using the noncentral t distribution.
    Noncentrality: nc = d * sqrt(n)
    Power = P(|T| > tcrit) under noncentral t
    """
    d = np.asarray(d, dtype=float)
    n = np.asarray(n, dtype=float)

    # guard
    power = np.full(np.broadcast(d, n).shape, np.nan, dtype=float)
    valid = n > 2
    if not np.any(valid):
        return power

    df = (n - 1.0)
    nc = d * np.sqrt(n)

    # tcrit depends on df
    tcrit = t.ppf(1 - alpha / 2.0, df)

    # P(|T| <= tcrit) = CDF(tcrit) - CDF(-tcrit)
    p_between = nct.cdf(tcrit, df, nc) - nct.cdf(-tcrit, df, nc)
    power = 1.0 - p_between
    return power


def min_items_for_target_power(
    n_fixed: int,
    d_fixed: float,
    rho: float,
    alpha: float,
    target_power: float,
    effect_defined_at: str,
    m_min: int,
    m_max: int,
) -> int | None:
    """
    Returns the smallest m in [m_min, m_max] such that power(m) >= target_power.
    If not reached, returns None.
    """
    ms = np.arange(m_min, m_max + 1, dtype=int)
    d_eff = d_effective(d_fixed, 1, rho, effect_defined_at)

    # If effect is participant_mean, d_eff doesn't depend on m, so power is constant in m.
    if effect_defined_at == "participant_mean":
        p_const = float(power_two_sided_one_sample_t(d_eff, n_fixed, alpha))
        return m_min if p_const >= target_power else None

    # item-level: compute power over m
    powers = []
    for m in ms:
        d_eff_m = float(d_effective(d_fixed, int(m), rho, effect_defined_at))
        powers.append(float(power_two_sided_one_sample_t(d_eff_m, n_fixed, alpha)))
    powers = np.array(powers, dtype=float)

    idx = np.where(powers >= target_power)[0]
    if len(idx) == 0:
        return None
    return int(ms[idx[0]])


# =========================
# Styling (professional defaults)
# =========================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})


# =========================
# 1) 3D surface: N (x) vs d (y) vs power (z) with fixed m
# =========================

Ns = np.arange(N_MIN, N_MAX + 1, N_STEP, dtype=int)
Ds = np.arange(D_MIN, D_MAX + 1e-12, D_STEP, dtype=float)
Ngrid, Dgrid = np.meshgrid(Ns, Ds)

D_eff_grid = d_effective(Dgrid, M_ITEMS_3D, RHO_3D, EFFECT_DEFINED_AT)
P_grid = power_two_sided_one_sample_t(D_eff_grid, Ngrid, ALPHA)

fig1 = plt.figure(figsize=(12, 7), dpi=180)
ax1 = fig1.add_subplot(111, projection="3d")

surf = ax1.plot_surface(
    Ngrid, Dgrid, P_grid,
    cmap=cm.viridis,      # gradient colormap
    linewidth=0,
    antialiased=True,
    rcount=200, ccount=200
)

ax1.set_xlabel("Sample size (N participants)")
ax1.set_ylabel("Expected effect size (Cohen's d)")
ax1.set_zlabel("Power")
ax1.set_title(
    f"Power surface (items/participant = {M_ITEMS_3D}, ICC rho = {RHO_3D}, alpha = {ALPHA}, d defined at: {EFFECT_DEFINED_AT})"
)
ax1.set_zlim(0, 1)
ax1.view_init(elev=28, azim=-135)

cbar = fig1.colorbar(surf, ax=ax1, pad=0.08, shrink=0.75)
cbar.set_label("Power")

fig1.tight_layout()


# =========================
# 2) 2D plot: power vs items m (fixed N=100 and d=0.30) for multiple rho
# =========================

ms = np.arange(M_MIN, M_MAX + 1, dtype=int)

fig2 = plt.figure(figsize=(11, 5.5), dpi=180)
ax2 = fig2.add_subplot(111)

for rho in RHO_VALUES:
    if EFFECT_DEFINED_AT == "participant_mean":
        # flat line (items irrelevant)
        p = float(power_two_sided_one_sample_t(d_effective(D_FIXED, 1, rho, EFFECT_DEFINED_AT), N_FIXED, ALPHA))
        powers = np.full_like(ms, p, dtype=float)
    else:
        # item-level: d_eff depends on m
        d_eff_m = np.array([float(d_effective(D_FIXED, int(m), rho, EFFECT_DEFINED_AT)) for m in ms], dtype=float)
        n_vec = np.full_like(d_eff_m, N_FIXED, dtype=float)
        powers = power_two_sided_one_sample_t(d_eff_m, n_vec, ALPHA)

    ax2.plot(ms, powers, label=f"rho = {rho:g}")

# 80% power threshold line (no explicit color specified)
ax2.axhline(TARGET_POWER, linestyle="--", linewidth=1)

ax2.set_xlabel("Number of items (m) per participant")
ax2.set_ylabel("Power")
ax2.set_title(
    f"Power vs items (N = {N_FIXED}, d = {D_FIXED}, alpha = {ALPHA}, d defined at: {EFFECT_DEFINED_AT})"
)
ax2.set_ylim(0, 1)
ax2.grid(True, linestyle=":", linewidth=0.8)
ax2.legend(loc="lower right", frameon=True)

fig2.tight_layout()


# =========================
# Minimum items needed for 80% power (prints)
# =========================

m_needed = min_items_for_target_power(
    n_fixed=N_FIXED,
    d_fixed=D_FIXED,
    rho=RHO_MAIN_FOR_PRINT,
    alpha=ALPHA,
    target_power=TARGET_POWER,
    effect_defined_at=EFFECT_DEFINED_AT,
    m_min=M_MIN,
    m_max=M_MAX,
)

# Also print the power at m=1 for context
p_m1 = float(
    power_two_sided_one_sample_t(
        float(d_effective(D_FIXED, 1, RHO_MAIN_FOR_PRINT, EFFECT_DEFINED_AT)),
        N_FIXED,
        ALPHA
    )
)

print("========== Power vs items summary ==========")
print(f"Fixed N = {N_FIXED}, d = {D_FIXED}, alpha = {ALPHA}, rho = {RHO_MAIN_FOR_PRINT}, d defined at = {EFFECT_DEFINED_AT}")
print(f"Power at m = 1 item: {p_m1:.3f}")

if m_needed is None:
    print(f"Minimum items needed to reach {TARGET_POWER:.0%} power within m in [{M_MIN}, {M_MAX}]: NOT REACHED")
else:
    p_needed = float(
        power_two_sided_one_sample_t(
            float(d_effective(D_FIXED, m_needed, RHO_MAIN_FOR_PRINT, EFFECT_DEFINED_AT)),
            N_FIXED,
            ALPHA
        )
    )
    print(f"Minimum items needed to reach {TARGET_POWER:.0%} power within m in [{M_MIN}, {M_MAX}]: {m_needed}")
    print(f"Power at m = {m_needed}: {p_needed:.3f}")

print("===========================================")


# =========================
# Save figures
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

out_3d = os.path.join(OUTPUT_DIR, OUTPUT_3D_FILENAME)
fig1.savefig(out_3d, bbox_inches="tight")
print(f"Saved 3D figure to: {out_3d}")

out_2d = os.path.join(OUTPUT_DIR, OUTPUT_2D_FILENAME)
fig2.savefig(out_2d, bbox_inches="tight")
print(f"Saved 2D figure to: {out_2d}")

#plt.show()
