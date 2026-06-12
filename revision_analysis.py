"""Revision analyses using the exported trajectories in Codes/output.

This script leaves the original Julia training code unchanged. It fits:
1. a no-information baseline with constant beta and nu;
2. a time-spline beta baseline;
3. a semi-mechanistic information model;
and then runs M=0/C=0 counterfactuals, parameter sensitivity, and a
residual bootstrap for the semi-mechanistic information link.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).parent / "revision_output" / ".matplotlib")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares, lsq_linear


N = 38_250_000.0
DEFAULT_FIXED = {"sigma": 0.19, "gamma": 0.1, "epsilon": 0.8}
WAVES = {
    "delta": {
        "epi": "neuralepi12.csv",
        "inter": "neuralintervention.csv",
        "vac": "neuralvaccine.csv",
    },
    "omicron": {
        "epi": "neuralepi12_second.csv",
        "inter": "neuralintervention_second.csv",
        "vac": "neuralvaccine_second.csv",
    },
}


def softplus(x: np.ndarray | float) -> np.ndarray | float:
    return np.logaddexp(0.0, x)


def load_wave(output_dir: Path, wave: str) -> pd.DataFrame:
    spec = WAVES[wave]
    epi = pd.read_csv(output_dir / spec["epi"])
    inter = pd.read_csv(output_dir / spec["inter"])
    vac = pd.read_csv(output_dir / spec["vac"])
    raw = pd.read_csv(output_dir / "datasmoothing.csv")
    for frame in (epi, inter, vac, raw):
        frame["date"] = pd.to_datetime(frame["date"])
    cols = [
        "date",
        "dailycase",
        "dailyvaccine",
        "dailyvaccine3",
    ]
    data = epi.merge(inter, on="date").merge(vac, on="date").merge(raw[cols], on="date")
    return data.sort_values("date").reset_index(drop=True)


def fit_information_dynamics(m: np.ndarray, c: np.ndarray) -> dict:
    """Fit dM=a0-aM*M+aC*C and dC=b0+bM*M-bC*C."""
    observed = np.column_stack([m, c])
    scale = observed.std(axis=0)
    scale[scale == 0] = 1.0
    dm = np.gradient(m)
    dc = np.gradient(c)
    xm = np.column_stack([np.ones_like(m), -m, c])
    xc = np.column_stack([np.ones_like(c), m, -c])
    initial = np.r_[
        lsq_linear(xm, dm, bounds=(0.0, np.inf)).x,
        lsq_linear(xc, dc, bounds=(0.0, np.inf)).x,
    ]
    times = np.arange(len(m), dtype=float)

    def run(p):
        def rhs(_t, z):
            return [
                p[0] - p[1] * z[0] + p[2] * z[1],
                p[3] + p[4] * z[0] - p[5] * z[1],
            ]

        sol = solve_ivp(
            rhs,
            (times[0], times[-1]),
            observed[0],
            t_eval=times,
            method="DOP853",
            rtol=1e-8,
            atol=1e-7,
        )
        if not sol.success:
            raise RuntimeError(sol.message)
        return sol.y.T

    result = least_squares(
        lambda p: ((run(p) - observed) / scale).ravel(),
        initial,
        bounds=(0.0, np.inf),
        max_nfev=1000,
    )
    fitted = run(result.x)
    pm, pc = result.x[:3], result.x[3:]
    return {
        "a0": pm[0],
        "decay_m": pm[1],
        "coupling_c_to_m": pm[2],
        "b0": pc[0],
        "coupling_m_to_c": pc[1],
        "decay_c": pc[2],
        "trajectory_rmse": float(np.sqrt(np.mean((fitted - observed) ** 2))),
        "trajectory": fitted,
    }


def initial_state(data: pd.DataFrame, fixed: dict) -> np.ndarray:
    daily_case = max(float(data.loc[0, "dailycase"]), 1.0)
    v0 = max(float(data.loc[0, "Vaccine"]), 1.0)
    return np.array(
        [
            N,
            v0,
            daily_case / fixed["sigma"],
            daily_case / fixed["gamma"],
            max(float(data.loc[0, "Case"]), 1.0),
            v0,
        ]
    )


def simulate(
    data: pd.DataFrame,
    beta_fn,
    nu_fn,
    fixed: dict,
) -> np.ndarray:
    times = np.arange(len(data), dtype=float)
    u0 = initial_state(data, fixed)

    def rhs(t, u):
        s, v, e, i, hi, hv = u
        beta = max(float(beta_fn(t)), 1e-10)
        nu = max(float(nu_fn(t)), 1e-10)
        inf_s = beta * s * i / N
        inf_v = (1.0 - fixed["epsilon"]) * beta * v * i / N
        return [
            -inf_s - nu * s,
            nu * s - inf_v,
            inf_s + inf_v - fixed["sigma"] * e,
            fixed["sigma"] * e - fixed["gamma"] * i,
            inf_s + inf_v,
            nu * s,
        ]

    sol = solve_ivp(
        rhs,
        (times[0], times[-1]),
        u0,
        t_eval=times,
        method="DOP853",
        rtol=1e-7,
        atol=1e-5,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def observation_residual(sim: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = np.maximum(sim[:, [4, 5]], 1.0)
    return (np.log(pred) - np.log(np.maximum(target, 1.0))).ravel()


def metrics(sim: np.ndarray, target: np.ndarray) -> dict:
    pred = np.maximum(sim[:, [4, 5]], 1.0)
    rel = (pred - target) / np.maximum(target, 1.0)
    return {
        "log_rmse": float(np.sqrt(np.mean(observation_residual(sim, target) ** 2))),
        "case_mape": float(np.mean(np.abs(rel[:, 0]))),
        "vaccine_mape": float(np.mean(np.abs(rel[:, 1]))),
    }


def standardized_information(
    data: pd.DataFrame, raw_override: np.ndarray | None = None
) -> tuple[np.ndarray, dict]:
    raw = (
        data[["MScoreInter", "CScoreInter", "MScoreVac", "CScoreVac"]].to_numpy(float)
        if raw_override is None
        else raw_override
    )
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0)
    scale[scale == 0] = 1.0
    return (raw - mean) / scale, {"mean": mean, "scale": scale}


def interpolate_columns(values: np.ndarray):
    x = np.arange(len(values), dtype=float)
    splines = [CubicSpline(x, values[:, j], extrapolate=True) for j in range(values.shape[1])]
    return lambda t: np.array([s(t) for s in splines], dtype=float)


def fit_no_information(data: pd.DataFrame, fixed: dict) -> dict:
    target = data[["Case", "Vaccine"]].to_numpy(float)

    def run(p):
        beta, nu = np.exp(p)
        return simulate(data, lambda _t: beta, lambda _t: nu, fixed)

    result = least_squares(
        lambda p: observation_residual(run(p), target),
        np.log([0.12, 0.005]),
        bounds=(np.log([1e-5, 1e-8]), np.log([3.0, 0.1])),
        max_nfev=500,
    )
    sim = run(result.x)
    return {
        "params": {"beta": float(np.exp(result.x[0])), "nu": float(np.exp(result.x[1]))},
        "simulation": sim,
        "metrics": metrics(sim, target),
    }


def fit_spline_beta(data: pd.DataFrame, fixed: dict, knots: int = 5) -> dict:
    target = data[["Case", "Vaccine"]].to_numpy(float)
    knot_t = np.linspace(0.0, len(data) - 1.0, knots)

    def run(p):
        beta_spline = CubicSpline(knot_t, p[:knots], extrapolate=True)
        beta_fn = lambda t: np.exp(np.clip(beta_spline(t), -12.0, 2.0))
        nu = np.exp(p[-1])
        return simulate(data, beta_fn, lambda _t: nu, fixed)

    initial = np.r_[np.full(knots, np.log(0.12)), np.log(0.005)]
    result = least_squares(
        lambda p: observation_residual(run(p), target),
        initial,
        bounds=(np.r_[np.full(knots, -12.0), -18.0], np.r_[np.full(knots, 2.0), -2.0]),
        max_nfev=1000,
    )
    sim = run(result.x)
    return {
        "params": {
            "knot_times": knot_t.tolist(),
            "log_beta_knots": result.x[:knots].tolist(),
            "nu": float(np.exp(result.x[-1])),
        },
        "simulation": sim,
        "metrics": metrics(sim, target),
    }


def fit_semi_mechanistic_link(
    data: pd.DataFrame,
    fixed: dict,
    information_raw: np.ndarray,
    target_override: np.ndarray | None = None,
    initial: np.ndarray | None = None,
) -> dict:
    target = (
        data[["Case", "Vaccine"]].to_numpy(float)
        if target_override is None
        else target_override
    )
    info, stats = standardized_information(data, information_raw)
    info_fn = interpolate_columns(info)

    def functions(p, intervention=None):
        def transformed(t):
            z = info_fn(t)
            if intervention == "M_zero":
                z[[0, 2]] = (0.0 - stats["mean"][[0, 2]]) / stats["scale"][[0, 2]]
            elif intervention == "C_zero":
                z[[1, 3]] = (0.0 - stats["mean"][[1, 3]]) / stats["scale"][[1, 3]]
            return z

        beta_fn = lambda t: min(
            5.0, softplus(p[0] + p[1] * transformed(t)[0] + p[2] * transformed(t)[1])
        )
        nu_fn = lambda t: softplus(p[3] + p[4] * transformed(t)[2] + p[5] * transformed(t)[3])
        return beta_fn, nu_fn

    def run(p, intervention=None):
        beta_fn, nu_fn = functions(p, intervention)
        return simulate(data, beta_fn, nu_fn, fixed)

    if initial is None:
        initial = np.array([-2.0, 0.0, 0.0, -5.5, 0.0, 0.0])
    result = least_squares(
        lambda p: observation_residual(run(p), target),
        initial,
        bounds=(-15.0, 15.0),
        max_nfev=1500,
    )
    sim = run(result.x)
    beta_fn, nu_fn = functions(result.x)
    times = np.arange(len(data), dtype=float)
    return {
        "params_array": result.x,
        "information_raw": information_raw,
        "params": {
            "beta_intercept": result.x[0],
            "beta_m": result.x[1],
            "beta_c": result.x[2],
            "nu_intercept": result.x[3],
            "nu_m": result.x[4],
            "nu_c": result.x[5],
        },
        "simulation": sim,
        "beta": np.array([beta_fn(t) for t in times]),
        "nu": np.array([nu_fn(t) for t in times]),
        "metrics": metrics(sim, target),
        "counterfactuals": {
            "M_zero": run(result.x, "M_zero"),
            "C_zero": run(result.x, "C_zero"),
        },
    }


def endpoint_effect(reference: np.ndarray, alternative: np.ndarray) -> dict:
    return {
        "case_endpoint_relative_change": float(
            alternative[-1, 4] / reference[-1, 4] - 1.0
        ),
        "vaccine_endpoint_relative_change": float(
            alternative[-1, 5] / reference[-1, 5] - 1.0
        ),
    }


def sensitivity_analysis(data: pd.DataFrame, semi: dict) -> pd.DataFrame:
    rows = []
    base = semi["simulation"]
    p = semi["params_array"]
    for name, base_value in DEFAULT_FIXED.items():
        for multiplier in (0.8, 1.2):
            fixed = DEFAULT_FIXED.copy()
            fixed[name] = base_value * multiplier
            perturbed = fit_semi_mechanistic_link(
                data, fixed, semi["information_raw"], initial=p
            )
            row = {
                "parameter": name,
                "multiplier": multiplier,
                "value": fixed[name],
                **endpoint_effect(base, perturbed["simulation"]),
                **perturbed["metrics"],
            }
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap(data: pd.DataFrame, semi: dict, replicates: int, seed: int) -> pd.DataFrame:
    if replicates <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    observed = data[["Case", "Vaccine"]].to_numpy(float)
    fitted = np.maximum(semi["simulation"][:, [4, 5]], 1.0)
    residuals = np.log(np.maximum(observed, 1.0)) - np.log(fitted)
    rows = []
    for b in range(replicates):
        sampled = residuals[rng.integers(0, len(residuals), len(residuals))]
        target = fitted * np.exp(sampled)
        try:
            fit = fit_semi_mechanistic_link(
                data,
                DEFAULT_FIXED,
                semi["information_raw"],
                target_override=target,
                initial=semi["params_array"],
            )
            rows.append(
                {
                    "replicate": b,
                    **fit["params"],
                    **endpoint_effect(fit["simulation"], fit["counterfactuals"]["M_zero"]),
                    "c_zero_case_endpoint_relative_change": endpoint_effect(
                        fit["simulation"], fit["counterfactuals"]["C_zero"]
                    )["case_endpoint_relative_change"],
                    "c_zero_vaccine_endpoint_relative_change": endpoint_effect(
                        fit["simulation"], fit["counterfactuals"]["C_zero"]
                    )["vaccine_endpoint_relative_change"],
                }
            )
        except (RuntimeError, ValueError):
            continue
    return pd.DataFrame(rows)


def cross_correlation_lag(
    x: np.ndarray, y: np.ndarray, max_lag: int = 30
) -> dict:
    """Correlate x(t) with y(t+lag); positive lag means y follows x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xx, yy = x[: len(x) - lag or None], y[lag:]
        else:
            xx, yy = x[-lag:], y[: len(y) + lag]
        if len(xx) < 8 or np.std(xx) == 0 or np.std(yy) == 0:
            corr = np.nan
        else:
            corr = float(np.corrcoef(xx, yy)[0, 1])
        rows.append({"lag_days": lag, "correlation": corr})
    valid = [row for row in rows if np.isfinite(row["correlation"])]
    strongest = max(valid, key=lambda row: abs(row["correlation"]))
    strongest_positive_lag = max(
        (row for row in valid if row["lag_days"] >= 0),
        key=lambda row: abs(row["correlation"]),
    )
    return {
        "strongest": strongest,
        "strongest_nonnegative_lag": strongest_positive_lag,
        "curve": rows,
    }


def lag_analysis(data: pd.DataFrame, semi: dict) -> dict:
    daily_case = np.diff(data["Case"].to_numpy(float), prepend=data.loc[0, "Case"])
    daily_vaccine = np.diff(
        data["Vaccine"].to_numpy(float), prepend=data.loc[0, "Vaccine"]
    )
    # Rate trajectories are the primary behavioral outcomes; daily increments
    # are included as observational checks.
    pairs = {
        "intervention_m_vs_beta": (data["MScoreInter"], data["beta"]),
        "intervention_c_vs_beta": (data["CScoreInter"], data["beta"]),
        "vaccine_m_vs_nu": (data["MScoreVac"], data["nu"]),
        "vaccine_c_vs_nu": (data["CScoreVac"], data["nu"]),
        "intervention_m_vs_daily_cases": (data["MScoreInter"], daily_case),
        "vaccine_m_vs_daily_vaccinations": (data["MScoreVac"], daily_vaccine),
    }
    return {
        name: cross_correlation_lag(np.asarray(x), np.asarray(y))
        for name, (x, y) in pairs.items()
    }


def save_trajectory(
    data: pd.DataFrame,
    models: dict,
    path: Path,
) -> None:
    out = pd.DataFrame({"date": data["date"]})
    for name, sim in models.items():
        out[f"{name}_case"] = sim[:, 4]
        out[f"{name}_vaccine"] = sim[:, 5]
    out.to_csv(path, index=False)


def make_revision_figure(result_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    colors = {
        "Observed": "black",
        "No information": "#999999",
        "Spline beta": "#e69f00",
        "Semi-mechanistic": "#0072b2",
    }
    for row, wave in enumerate(("delta", "omicron")):
        traj = pd.read_csv(result_dir / f"{wave}_trajectories.csv")
        lag = pd.read_csv(result_dir / f"{wave}_lag_correlations.csv")
        x = np.arange(len(traj))

        ax = axes[row, 0]
        ax.plot(x, traj["observed_case"], color=colors["Observed"], lw=2, label="Observed")
        ax.plot(
            x,
            traj["no_information_case"],
            color=colors["No information"],
            lw=1.6,
            ls="--",
            label="No information",
        )
        ax.plot(
            x,
            traj["spline_beta_case"],
            color=colors["Spline beta"],
            lw=1.6,
            ls="-.",
            label=r"Spline $\beta(t)$",
        )
        ax.plot(
            x,
            traj["semi_mechanistic_case"],
            color=colors["Semi-mechanistic"],
            lw=1.8,
            label="Semi-mechanistic",
        )
        ax.set_title(f"{wave.title()}: cumulative cases")
        ax.set_xlabel("Day")
        ax.set_ylabel("Cumulative cases")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.25)

        ax = axes[row, 1]
        for pair, label, color in (
            ("intervention_m_vs_beta", "Misinformation vs. beta", "#d55e00"),
            ("intervention_c_vs_beta", "Correct information vs. beta", "#009e73"),
        ):
            part = lag[lag["pair"] == pair]
            ax.plot(part["lag_days"], part["correlation"], lw=1.8, label=label, color=color)
            nonnegative = part[part["lag_days"] >= 0]
            peak = nonnegative.iloc[np.argmax(np.abs(nonnegative["correlation"]))]
            ax.scatter([peak["lag_days"]], [peak["correlation"]], color=color, s=35, zorder=3)
            ax.annotate(
                f'{int(peak["lag_days"])} d',
                (peak["lag_days"], peak["correlation"]),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=8,
            )
        ax.axhline(0, color="black", lw=0.8)
        ax.axvline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{wave.title()}: intervention-information lag")
        ax.set_xlabel("Lag (days; positive means beta follows information)")
        ax.set_ylabel("Cross-correlation")
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(result_dir / "revision_validation.pdf", bbox_inches="tight")
    fig.savefig(result_dir / "revision_validation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def json_ready(value):
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def analyze_wave(output_dir: Path, result_dir: Path, wave: str, replicates: int) -> None:
    data = load_wave(output_dir, wave)
    fixed = DEFAULT_FIXED.copy()
    no_info = fit_no_information(data, fixed)
    spline = fit_spline_beta(data, fixed)
    information = {
        "intervention": fit_information_dynamics(
            data["MScoreInter"].to_numpy(float), data["CScoreInter"].to_numpy(float)
        ),
        "vaccination": fit_information_dynamics(
            data["MScoreVac"].to_numpy(float), data["CScoreVac"].to_numpy(float)
        ),
    }
    information_raw = np.column_stack(
        [
            information["intervention"]["trajectory"],
            information["vaccination"]["trajectory"],
        ]
    )
    semi = fit_semi_mechanistic_link(data, fixed, information_raw)
    counterfactual = {
        name: endpoint_effect(semi["simulation"], sim)
        for name, sim in semi["counterfactuals"].items()
    }
    lags = lag_analysis(data, semi)
    sensitivity = sensitivity_analysis(data, semi)
    boot = bootstrap(data, semi, replicates, seed=20260612 + (wave == "omicron"))

    result_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory(
        data,
        {
            "observed": np.column_stack(
                [
                    np.zeros((len(data), 4)),
                    data["Case"].to_numpy(float),
                    data["Vaccine"].to_numpy(float),
                ]
            ),
            "no_information": no_info["simulation"],
            "spline_beta": spline["simulation"],
            "semi_mechanistic": semi["simulation"],
            "counterfactual_M_zero": semi["counterfactuals"]["M_zero"],
            "counterfactual_C_zero": semi["counterfactuals"]["C_zero"],
        },
        result_dir / f"{wave}_trajectories.csv",
    )
    sensitivity.to_csv(result_dir / f"{wave}_sensitivity.csv", index=False)
    boot.to_csv(result_dir / f"{wave}_bootstrap.csv", index=False)
    if not boot.empty:
        boot.quantile([0.025, 0.5, 0.975], numeric_only=True).to_csv(
            result_dir / f"{wave}_bootstrap_summary.csv"
        )
    lag_rows = []
    for pair, result in lags.items():
        for row in result["curve"]:
            lag_rows.append({"pair": pair, **row})
    pd.DataFrame(lag_rows).to_csv(result_dir / f"{wave}_lag_correlations.csv", index=False)

    summary = {
        "wave": wave,
        "rows": len(data),
        "no_information": {"params": no_info["params"], "metrics": no_info["metrics"]},
        "spline_beta": {"params": spline["params"], "metrics": spline["metrics"]},
        "semi_mechanistic": {"params": semi["params"], "metrics": semi["metrics"]},
        "semi_mechanistic_information_dynamics": {
            group: {k: v for k, v in values.items() if k != "trajectory"}
            for group, values in information.items()
        },
        "counterfactual_endpoint_effects": counterfactual,
        "cross_correlation_lags": {
            pair: {
                "strongest": result["strongest"],
                "strongest_nonnegative_lag": result["strongest_nonnegative_lag"],
            }
            for pair, result in lags.items()
        },
        "counterfactual_warning": (
            "M(t)=0 and C(t)=0 are outside the observed information-score range; "
            "their effects are model-based extrapolations and should not be interpreted "
            "as identified causal effects."
        ),
        "bootstrap_requested": replicates,
        "bootstrap_completed": len(boot),
    }
    (result_dir / f"{wave}_summary.json").write_text(
        json.dumps(json_ready(summary), indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    parser.add_argument(
        "--result-dir", type=Path, default=Path(__file__).parent / "revision_output"
    )
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--waves", nargs="+", choices=WAVES, default=list(WAVES))
    args = parser.parse_args()
    for wave in args.waves:
        print(f"Analyzing {wave}...")
        analyze_wave(args.output_dir, args.result_dir, wave, args.bootstrap)
    if set(args.waves) == set(WAVES):
        make_revision_figure(args.result_dir)
    print(f"Results written to {args.result_dir}")


if __name__ == "__main__":
    main()
