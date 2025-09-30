"""Problem 1: Autocorrelation analysis and small-sample bias simulation.

Regression-based estimation of autocorrelation with heteroskedasticity-robust (HC1) SEs.
Adds asymptotic bias approximation (-1/n) and simulated (bootstrap) small-sample bias/SE
by resampling returns with replacement (destroying serial dependence but retaining
marginal heteroskedasticity structure).

Extended: Computes joint statistics (first-j) Box-Pierce Q(j), Variance Ratio VR(j),
and long-run return regression coefficient delta(j) with both asymptotic and
bootstrap (iid null) p-values. Bootstrap samples are generated once and reused across
all statistics for efficiency.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats  # new for chi-square / normal cdf

@dataclass
class AutoCorrResult:
    autocorr: pd.Series      # regression slope estimates b_h (lag index)
    se_asy: pd.Series        # robust HC1 standard errors
    bias_asy: pd.Series      # asymptotic bias approximation (-1/n)
    bias_sim: pd.Series      # simulated (bootstrap) bias (mean of boot slopes)
    se_sim: pd.Series        # simulated (bootstrap) standard deviation of slopes
    # Joint statistics indexed by j (j corresponds to using first j autocorrs)
    q_stat: pd.Series        # Box-Pierce Q(j)
    q_p_asy: pd.Series       # Asymptotic p-value (chi-square)
    q_p_sim: pd.Series       # Bootstrap p-value under iid null
    vr: pd.Series            # Variance Ratio VR(j)
    vr_p_asy: pd.Series      # Asymptotic p-value (normal approximation)
    vr_p_sim: pd.Series      # Bootstrap p-value
    delta: pd.Series         # Long-run regression coefficient delta(j)
    delta_p_asy: pd.Series   # Asymptotic p-value (normal approximation)
    delta_p_sim: pd.Series   # Bootstrap p-value
    n: int
    mean_return: float
    reps: int | None = None

# ----------------------------------------------------------------------------------
# Core computation
# ----------------------------------------------------------------------------------

def compute_autocorrelation(
    returns: pd.Series,
    max_lag: int = 20,
    bootstrap: bool = True,
    reps: int = 500,
    seed: int | None = 123,
) -> AutoCorrResult:
    """Estimate autocorrelations via univariate regressions r_{t+h} on r_t + constant.

    For each h in 1..max_lag run OLS: r_{t+h} = a_h + b_h r_t + e_{t,h}.
    Report slope b_h as autocorrelation estimate and White (HC1) robust SE.

    Additionally compute:
    - bias_asy: constant series (-1/n) approximation to small-sample bias under white noise.
    - bias_sim / se_sim: bootstrap (with replacement) distribution of slope estimates where
      each replication draws n observations i.i.d. from the empirical distribution.
    - Joint statistics for j = 1..L (L = max computed lag):
        * Q(j) = N * sum_{k=1}^j rho(k)^2 (Box-Pierce) with chi-square(j) asymptotic p.
        * VR(j) = 2 * sum_{k=1}^{j-1} ((j-k)/j) * rho(k) with asymptotic variance
          Omega_VR(j)/N where Omega_VR(j)=4*sum_{k=1}^{j-1}((j-k)/j)^2
        * delta(j) (long-run return regression coefficient) using sample scaling factor
          Var(r) / ((1/j)*Var(sum_{k=0}^{j-1} r_{t-k})). Under iid null scaling=1 and
          sqrt(N)*delta(j) ~ N(0,1/j).
      For each, both asymptotic and bootstrap p-values (two-sided) under iid null.

    Parameters
    ----------
    returns : pd.Series
        Input return series.
    max_lag : int
        Maximum horizon h.
    bootstrap : bool
        Whether to run bootstrap simulation.
    reps : int
        Number of bootstrap repetitions (ignored if bootstrap=False).
    seed : int | None
        RNG seed for bootstrap.
    """
    if returns.isna().any():
        returns = returns.dropna()
    r = returns.to_numpy()
    n = len(r)

    betas: dict[int, float] = {}
    ses: dict[int, float] = {}

    for h in range(1, max_lag + 1):
        if n - h < 5:  # not enough overlapping pairs
            break
        y = r[h:]
        x = r[:-h]
        X = sm.add_constant(x)
        model = sm.OLS(y, X)
        res = model.fit(cov_type='HC1')
        betas[h] = float(res.params[1])
        ses[h] = float(res.bse[1])

    lags = sorted(betas.keys())
    if not lags:
        raise ValueError("No lags could be computed (increase series length or reduce max_lag).")

    L = max(lags)
    ac_series = pd.Series(betas, name='autocorr')
    se_series = pd.Series(ses, name='se_asy')
    bias_asy_series = pd.Series({h: -1.0/n for h in lags}, name='bias_asy')

    # Preallocate joint statistics containers (index j = 1..L)
    idx_j = pd.Index(range(1, L + 1), name='j')
    q_stat = pd.Series(np.nan, index=idx_j, name='q_stat')
    vr = pd.Series(np.nan, index=idx_j, name='vr')
    delta = pd.Series(np.nan, index=idx_j, name='delta')

    # Compute cumulative forms from sample autocorrelations
    rho_vals = ac_series.reindex(range(1, L + 1))  # may include NaN for missing
    # Compute Q(j)
    running_sum_q = 0.0
    for j in range(1, L + 1):
        rj = rho_vals.iloc[j-1]
        if np.isnan(rj):
            break
        running_sum_q += rj * rj
        q_stat.iloc[j-1] = n * running_sum_q
        # Variance Ratio
        if j >= 2:
            weights = [(j - k) / j for k in range(1, j)]  # k=1..j-1
            vr.iloc[j-1] = 2.0 * float(np.nansum([w * rho_vals.iloc[k-1] for k, w in enumerate(weights, start=1)]))
        elif j == 1:
            vr.iloc[j-1] = 0.0  # convention
        # delta(j)
        # scaling factor: Var(r) / ((1/j)*Var(sum_{k=0}^{j-1} r_{t-k}))
        if j == 1:
            factor = 1.0
        else:
            # Overlapping sums S_t for t=j-1..n-1 inclusive
            csum = np.concatenate([[0.0], np.cumsum(r)])
            S = csum[j:] - csum[:-j]
            var_S = np.var(S, ddof=1) if len(S) > 1 else np.nan
            var_r = np.var(r, ddof=1)
            if var_S == 0 or np.isnan(var_S):
                factor = np.nan
            else:
                factor = var_r / ((1.0 / j) * var_S)
        delta.iloc[j-1] = factor * float(np.nanmean(rho_vals.iloc[:j]))

    # Asymptotic p-values
    # Box-Pierce Q(j) right-tail p-value under chi-square_j
    q_p_asy = pd.Series({j: (1 - stats.chi2.cdf(q_stat.loc[j], j)) if not np.isnan(q_stat.loc[j]) else np.nan for j in q_stat.index}, name='q_p_asy')

    vr_p_asy_vals = {}
    delta_p_asy_vals = {}
    for j in range(1, L + 1):
        # VR variance Omega_VR(j)/N
        vr_j = vr.loc[j]
        if j >= 2 and not np.isnan(vr_j):
            omega_vr = 4.0 * sum(((j - k) / j) ** 2 for k in range(1, j))
            sd_vr = np.sqrt(omega_vr / n)
            z_vr = vr_j / sd_vr if sd_vr > 0 else np.nan
            vr_p_asy_vals[j] = 2 * (1 - stats.norm.cdf(abs(z_vr))) if not np.isnan(z_vr) else np.nan
        else:
            vr_p_asy_vals[j] = np.nan  # j=1 or undefined
        # Delta variance 1/(j*N) under iid null
        d_j = delta.loc[j]
        if not np.isnan(d_j):
            sd_d = np.sqrt(1.0 / (j * n))
            z_d = d_j / sd_d if sd_d > 0 else np.nan
            delta_p_asy_vals[j] = 2 * (1 - stats.norm.cdf(abs(z_d))) if not np.isnan(z_d) else np.nan
        else:
            delta_p_asy_vals[j] = np.nan
    vr_p_asy = pd.Series(vr_p_asy_vals, name='vr_p_asy')
    delta_p_asy = pd.Series(delta_p_asy_vals, name='delta_p_asy')

    # Bootstrap simulation (white-noise null via i.i.d. resampling)
    if bootstrap and lags:
        rng = np.random.default_rng(seed)
        num_lags = len(lags)
        # 2D array to store boot autocorr slopes (re-using fast covariance formula)
        boot_rhos = np.empty((reps, num_lags), dtype=float)
        boot_rhos.fill(np.nan)
        # Precompute index map for position of lag h within lags list
        lag_pos = {h: i for i, h in enumerate(lags)}

        # For joint stats across j we also store VR, delta, Q per replication for all j up to L
        boot_Q = np.empty((reps, L), dtype=float); boot_Q.fill(np.nan)
        boot_VR = np.empty((reps, L), dtype=float); boot_VR.fill(np.nan)
        boot_D = np.empty((reps, L), dtype=float); boot_D.fill(np.nan)

        for b in range(reps):
            sample = r[rng.integers(0, n, size=n)]  # i.i.d. resample
            # Compute slopes for each computed lag only (lags may be contiguous starting at 1)
            for h in lags:
                if n - h < 5:
                    continue
                xb = sample[:-h]
                yb = sample[h:]
                xb_mean = xb.mean(); yb_mean = yb.mean()
                num = ((xb - xb_mean) * (yb - yb_mean)).sum()
                den = ((xb - xb_mean) ** 2).sum()
                if den == 0:
                    beta_b = np.nan
                else:
                    beta_b = num / den
                boot_rhos[b, lag_pos[h]] = beta_b
            # Derive joint stats using available initial lags sequentially
            # Build a rho vector length L with NaN if missing
            rho_b = np.full(L, np.nan)
            for h in lags:
                rho_b[h-1] = boot_rhos[b, lag_pos[h]]
            # Running Q
            running_q = 0.0
            for j in range(1, L + 1):
                rj = rho_b[j-1]
                if np.isnan(rj):
                    break
                running_q += rj * rj
                boot_Q[b, j-1] = n * running_q
                # VR
                if j == 1:
                    boot_VR[b, j-1] = 0.0
                else:
                    boot_VR[b, j-1] = 2.0 * np.nansum([((j - k) / j) * rho_b[k-1] for k in range(1, j)])
                # delta scaling
                if j == 1:
                    factor_b = 1.0
                else:
                    csum_b = np.concatenate([[0.0], np.cumsum(sample)])
                    S_b = csum_b[j:] - csum_b[:-j]
                    var_Sb = np.var(S_b, ddof=1) if len(S_b) > 1 else np.nan
                    var_rb = np.var(sample, ddof=1)
                    if var_Sb == 0 or np.isnan(var_Sb):
                        factor_b = np.nan
                    else:
                        factor_b = var_rb / ((1.0 / j) * var_Sb)
                boot_D[b, j-1] = factor_b * np.nanmean(rho_b[:j])

        # Bias & SE for individual lags
        bias_sim = {h: float(np.nanmean(boot_rhos[:, lag_pos[h]])) for h in lags}
        se_sim = {h: float(np.nanstd(boot_rhos[:, lag_pos[h]], ddof=1)) for h in lags}
        bias_sim_series = pd.Series(bias_sim, name='bias_sim')
        se_sim_series = pd.Series(se_sim, name='se_sim')

        # Bootstrap p-values (two-sided) for joint stats (exclude j where stat is NaN)
        def two_sided_p(obs: float, arr: np.ndarray) -> float:
            if np.isnan(obs):
                return np.nan
            valid = arr[~np.isnan(arr)]
            if valid.size == 0:
                return np.nan
            return float(np.mean(np.abs(valid) >= abs(obs)))
        def right_tail_p(obs: float, arr: np.ndarray) -> float:
            if np.isnan(obs):
                return np.nan
            valid = arr[~np.isnan(arr)]
            if valid.size == 0:
                return np.nan
            return float(np.mean(valid >= obs))

        q_p_sim = pd.Series({j: right_tail_p(q_stat.loc[j], boot_Q[:, j-1]) for j in range(1, L + 1)}, name='q_p_sim')
        vr_p_sim = pd.Series({j: two_sided_p(vr.loc[j], boot_VR[:, j-1]) for j in range(1, L + 1)}, name='vr_p_sim')
        delta_p_sim = pd.Series({j: two_sided_p(delta.loc[j], boot_D[:, j-1]) for j in range(1, L + 1)}, name='delta_p_sim')
    else:
        bias_sim_series = pd.Series(dtype=float, name='bias_sim')
        se_sim_series = pd.Series(dtype=float, name='se_sim')
        q_p_sim = pd.Series(dtype=float, name='q_p_sim')
        vr_p_sim = pd.Series(dtype=float, name='vr_p_sim')
        delta_p_sim = pd.Series(dtype=float, name='delta_p_sim')

    return AutoCorrResult(
        autocorr=ac_series,
        se_asy=se_series,
        bias_asy=bias_asy_series,
        bias_sim=bias_sim_series,
        se_sim=se_sim_series,
        q_stat=q_stat,
        q_p_asy=q_p_asy,
        q_p_sim=q_p_sim,
        vr=vr,
        vr_p_asy=vr_p_asy,
        vr_p_sim=vr_p_sim,
        delta=delta,
        delta_p_asy=delta_p_asy,
        delta_p_sim=delta_p_sim,
        n=n,
        mean_return=float(returns.mean()),
        reps=reps if bootstrap else None,
    )

# ----------------------------------------------------------------------------------
# Simulation of classic sample autocorr bias (legacy helper retained)
# ----------------------------------------------------------------------------------

def simulate_autocorr_bias(n: int, reps: int = 10_000, max_lag: int = 5, seed: int | None = 42) -> pd.DataFrame:
    """Simulate white-noise returns to illustrate small-sample bias of sample autocorrelation (legacy).

    Returns DataFrame with columns: true_rho, sample_mean, bias, se.
    """
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(reps):
        x = rng.standard_normal(n)
        for lag in range(1, max_lag + 1):
            rhat = pd.Series(x).autocorr(lag=lag)
            acc.append((lag, rhat))
    df = pd.DataFrame(acc, columns=["lag", "rhat"])
    out = df.groupby("lag").agg(sample_mean=("rhat", "mean"), se=("rhat", "std"))
    out.insert(0, "true_rho", 0.0)
    out["bias"] = out["sample_mean"] - out["true_rho"]
    return out.reset_index()

# ----------------------------------------------------------------------------------
# Simple bias correction placeholder
# ----------------------------------------------------------------------------------

def bias_correction(sample_r: float, n: int, lag: int) -> float:  # noqa: ARG001 (lag unused placeholder)
    return sample_r - (1.0 / n)

__all__ = [
    "AutoCorrResult", "compute_autocorrelation", "simulate_autocorr_bias", "bias_correction"
]

# -----------------------------
# CLI / Script Execution Support
# -----------------------------

def _load_returns(kind: str, length: int) -> pd.Series:
    rng = np.random.default_rng(0 if kind == 'value' else 1)
    eps = rng.standard_normal(length) * 0.01
    r = np.zeros(length)
    phi = 0.05 if kind == 'value' else 0.03
    for t in range(1, length):
        r[t] = phi * r[t-1] + eps[t]
    idx = pd.RangeIndex(start=0, stop=length, step=1)
    return pd.Series(r, index=idx, name=f"{kind}_weighted_return")

def write_autocorr_csv(returns: pd.Series, max_lag: int, output_path: str, bootstrap: bool, reps: int) -> None:
    res = compute_autocorrelation(returns, max_lag=max_lag, bootstrap=bootstrap, reps=reps)
    # Build DataFrame with per-lag (univariate) and per-j (joint) stats (align j with lag index where possible)
    lag_index = res.autocorr.index
    joint_index = res.q_stat.index
    # Align joint stats to lag index where overlapping values exist
    max_idx = max(lag_index.max(), joint_index.max())
    all_index = pd.Index(range(1, max_idx + 1), name='lag')
    df = pd.concat([
        res.autocorr.reindex(all_index),
        res.se_asy.reindex(all_index),
        res.bias_asy.reindex(all_index),
        res.bias_sim.reindex(all_index),
        res.se_sim.reindex(all_index),
        res.q_stat.reindex(all_index),
        res.q_p_asy.reindex(all_index),
        res.q_p_sim.reindex(all_index),
        res.vr.reindex(all_index),
        res.vr_p_asy.reindex(all_index),
        res.vr_p_sim.reindex(all_index),
        res.delta.reindex(all_index),
        res.delta_p_asy.reindex(all_index),
        res.delta_p_sim.reindex(all_index),
    ], axis=1)
    df.reset_index().to_csv(output_path, index=False)

def main():  # pragma: no cover
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Compute regression-based autocorrelations and joint stats; output CSV with SE/bias/p-values.")
    parser.add_argument("--lags", type=int, default=10, help="Number of lags to compute (positive integer).")
    parser.add_argument("--kind", type=str, default="value", choices=["value", "equal"], help="Return series type (synthetic placeholder).")
    parser.add_argument("--length", type=int, default=80, help="Length of synthetic return series.")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: autocorr_<kind>_L<lags>.csv).")
    parser.add_argument("--no-bootstrap", action="store_true", help="Disable bootstrap for simulated bias/SE and joint p-values.")
    parser.add_argument("--reps", type=int, default=1000, help="Bootstrap repetitions (default 1000).")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for bootstrap resampling.")

    args = parser.parse_args()

    if args.lags < 1:
        print("--lags must be >= 1", file=sys.stderr)
        sys.exit(2)
    if args.length <= args.lags + 5:
        print("--length should exceed --lags by at least 5 for stable regression windows", file=sys.stderr)
        sys.exit(2)
    if args.reps < 1:
        print("--reps must be >=1", file=sys.stderr)
        sys.exit(2)

    returns = _load_returns(args.kind, args.length)
    out_path = args.output or f"autocorr_{args.kind}_L{args.lags}.csv"
    write_autocorr_csv(returns, args.lags, out_path, bootstrap=not args.no_bootstrap, reps=args.reps)

    p = Path(out_path).resolve()
    print(f"Wrote autocorrelation & joint stats (lags 1..{args.lags}) for {args.kind} series (n={len(returns)}) -> {p}")

if __name__ == "__main__":  # pragma: no cover
    main()
