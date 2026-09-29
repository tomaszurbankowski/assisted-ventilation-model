#!/usr/bin/env python3
"""PEEP-inclusive pressure-reference sensitivity for the assisted-ventilation model.

This is NOT an application of Gattinoni et al.'s constant-flow closed-form
Eq. (6) to pressure-support breaths. It uses direct signed pressure-flow
integration with PEEP included in airway pressure, while retaining the model's
original inspiratory mask and all 180 primary input combinations.

The original model is imported without modification. PEEP is included
consistently in ventilator, total, and compartmental energy; muscle power is
unchanged. Primary above-PEEP outputs are retained in separate columns.

Usage (beside assisted_ventilation_model_final.py in the repository root):
    python run_R3C1_peep_reference.py
Or specify locations explicitly:
    python run_R3C1_peep_reference.py --model path/to/model.py --output results
Requires the dependencies of the original model (NumPy with np.trapezoid,
Pandas). No figures are generated; no primary files are overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd

PHENOTYPES = ("compliance_dominant", "resistance_dominant", "mixed", "severe_mixed")
PRESSURE_SUPPORT = (5.0, 10.0, 15.0)
EFFORT_AMPLITUDES = (0.0, 3.0, 6.0, 9.0, 12.0)
EFFORT_DURATIONS = (0.6, 1.0, 1.4)
TOLERANCE = 0.05


def load_model(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}. Use --model to specify it.")
    name = "_r3c1_original_model"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import model: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for attr in ("simulate_breath", "compute_metrics", "matched_pairs", "inspiratory_mask"):
        if not callable(getattr(module, attr, None)):
            raise AttributeError(f"Original model is missing {attr}().")
    return module


def run_grid(model: Any, dt: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for number, (ph, ps, amplitude, duration) in enumerate(itertools.product(
        PHENOTYPES, PRESSURE_SUPPORT, EFFORT_AMPLITUDES, EFFORT_DURATIONS
    )):
        compartments = model.phenotype_definition(ph)
        c1, c2 = compartments["comp1"], compartments["comp2"]
        vent = model.VentilatorSettings(
            peep=5.0, pressure_support=ps, respiratory_rate=18.0,
            inspiratory_time=1.0, dt=dt,
        )
        effort = model.EffortSettings(pmus_peak=amplitude, effort_duration=duration, onset=0.0)
        sim = model.simulate_breath(c1, c2, vent, effort)
        row = model.compute_metrics(sim, c1, c2, vent)
        mask = model.inspiratory_mask(sim["t"], vent.inspiratory_time)
        time = sim["t"][mask]

        def integrate(signal: np.ndarray) -> float:
            return model.trapz_integral(signal[mask], time)

        conversion = model.CMH2O_L_TO_J
        rr = vent.respiratory_rate
        # Integrate flow over EXACTLY the same samples as the energy integrals.
        # Do not replace this with VT_total_L, which the original model obtains
        # using endpoint interpolation after explicit-Euler volume integration.
        volume_integral = integrate(sim["f_tot"])
        peep_power = conversion * rr * vent.peep * volume_integral
        e_vent = conversion * integrate(sim["paw"] * sim["f_tot"])
        total_pressure = sim["paw"] + sim["pmus"]
        e_total = conversion * integrate(total_pressure * sim["f_tot"])
        e1 = conversion * integrate(total_pressure * sim["f1"])
        e2 = conversion * integrate(total_pressure * sim["f2"])
        if not (e_vent > 0 and e_total > 0 and e1 > 0 and e2 > 0):
            raise ValueError(f"Unexpected non-positive energy in scenario {number}.")
        row.update({
            "scenario_id": f"S{number:03d}", "phenotype": ph,
            "Pmus_peak_cmH2O": amplitude, "Tmus_s": duration,
            "Pmus_onset_s": 0.0, "dt_s": dt,
            "V_integral_I_L": volume_integral,
            "last_inspiratory_sample_s": float(time[-1]),
            "min_total_flow_in_I_Ls": float(np.min(sim["f_tot"][mask])),
            "MP_PEEP_term_Jmin": peep_power,
            "E_vent_PEEP_J": e_vent, "E_tot_PEEP_J": e_total,
            "E1_tot_PEEP_J": e1, "E2_tot_PEEP_J": e2,
            "MP_vent_PEEP_Jmin": e_vent * rr,
            "MP_tot_PEEP_Jmin": e_total * rr,
            "EF1_PEEP": e1 / e_total, "EF2_PEEP": e2 / e_total,
            "EII_PEEP": abs(e1 - e2) / e_total,
            "ECF_PEEP": row["MP_mus_Jmin"] / (e_total * rr),
            "ratio_PEEP": e_total / e_vent,
        })
        records.append(row)
    frame = pd.DataFrame(records)
    if len(frame) != 180:
        raise AssertionError("The primary grid must contain exactly 180 scenarios.")
    # These algebraic checks verify consistent accounting, not clinical validity.
    np.testing.assert_allclose(frame.MP_vent_PEEP_Jmin,
        frame.MP_vent_Jmin + frame.MP_PEEP_term_Jmin, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(frame.MP_tot_PEEP_Jmin,
        frame.MP_tot_Jmin + frame.MP_PEEP_term_Jmin, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(frame.MP_tot_PEEP_Jmin,
        frame.MP_vent_PEEP_Jmin + frame.MP_mus_Jmin, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(frame.EF1_PEEP + frame.EF2_PEEP, 1.0, atol=1e-12)
    np.testing.assert_allclose(frame.ratio_PEEP, 1.0 / (1.0-frame.ECF_PEEP), rtol=1e-12)
    np.testing.assert_allclose(frame.MP_vent_PEEP_Jmin,
        frame.MP_vent_Jmin * (1.0 + frame.PEEP_cmH2O/frame.PS_cmH2O), rtol=1e-12)
    return frame


def make_pairs(model: Any, frame: pd.DataFrame, matching: str) -> pd.DataFrame:
    pairs = model.matched_pairs(frame, match_variable=matching, tolerance=TOLERANCE)
    if pairs.empty:
        raise ValueError(f"No matched pairs for {matching}.")
    left = pairs.idx_i.to_numpy(dtype=int)
    right = pairs.idx_j.to_numpy(dtype=int)
    for field in ("scenario_id", "Pmus_peak_cmH2O", "Tmus_s", "PS_cmH2O",
                  "MP_vent_PEEP_Jmin", "MP_tot_PEEP_Jmin", "MP_mus_Jmin",
                  "EF1_PEEP", "EII_PEEP", "ECF_PEEP", "ratio_PEEP"):
        a = frame.iloc[left][field].to_numpy()
        b = frame.iloc[right][field].to_numpy()
        pairs[field+"_i"], pairs[field+"_j"] = a, b
        if field != "scenario_id":
            pairs["abs_delta_"+field] = np.abs(a-b)
    return pairs


def summarize_pairs(pairs: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {"n_pairs": len(pairs)}
    for field in ("MP_tot_PEEP_Jmin", "MP_mus_Jmin", "EF1_PEEP", "EII_PEEP",
                  "ECF_PEEP", "ratio_PEEP"):
        values = pairs["abs_delta_"+field].to_numpy()
        output["abs_delta_"+field] = {
            "min": float(np.min(values)), "median": float(np.median(values)),
            "max": float(np.max(values)),
        }
    return output


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=here / "assisted_ventilation_model_final.py")
    parser.add_argument("--output", type=Path, default=here / "supplementary" / "peep_inclusive" / "data")
    args = parser.parse_args()
    model_path = args.model.resolve()
    model = load_model(model_path)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "analysis": "PEEP-inclusive pressure-reference sensitivity; not direct Gattinoni Eq. (6)",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "model_filename": model_path.name,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "conversion_cmh2o_l_to_j": float(model.CMH2O_L_TO_J),
        "peep_cmH2O": 5.0, "rr_per_min": 18.0, "Ti_s": 1.0,
        "integration": "signed trapezoidal integration on original inspiratory mask t < Ti",
        "pairing": "original non-exclusive i<j ordering; abs(x_j-x_i)/x_i <= 0.05",
        "passive_duplicates": "retained exactly as in the original 180-scenario grid",
        "timesteps": {},
    }
    frames = []
    for dt in (0.001, 0.0005):
        tag = "dt" + str(dt).replace(".", "p")
        frame = run_grid(model, dt)
        frames.append(frame)
        frame.to_csv(output / f"R3C1_scenarios_{tag}.csv", index=False)
        item: dict[str, Any] = {"n_scenarios": len(frame), "ranges": {}, "pairs": {}}
        for field in ("MP_vent_Jmin", "MP_tot_Jmin", "MP_mus_Jmin", "HBR",
                      "MP_vent_PEEP_Jmin", "MP_tot_PEEP_Jmin", "ratio_PEEP", "ECF_PEEP",
                      "EF1_PEEP", "EII_PEEP"):
            item["ranges"][field] = [float(frame[field].min()), float(frame[field].max())]
        item["ratio_PEEP_max_by_phenotype"] = {
            key: float(value) for key,value in frame.groupby("phenotype").ratio_PEEP.max().items()
        }
        for matching in ("MP_vent_Jmin", "MP_tot_Jmin", "MP_vent_PEEP_Jmin", "MP_tot_PEEP_Jmin"):
            pairs = make_pairs(model, frame, matching)
            if dt == 0.001 and matching in ("MP_vent_Jmin", "MP_tot_Jmin"):
                expected = 505 if matching == "MP_vent_Jmin" else 534
                if len(pairs) != expected:
                    raise AssertionError(f"Original {matching} pair count differs: {len(pairs)} vs {expected}.")
            if "_PEEP_" in matching:
                pairs.to_csv(output / f"R3C1_pairs_{matching}_{tag}.csv", index=False)
            item["pairs"][matching] = summarize_pairs(pairs)
        item["max_balance_error_Jmin"] = float(np.max(np.abs(
            frame.MP_tot_PEEP_Jmin - frame.MP_vent_PEEP_Jmin - frame.MP_mus_Jmin)))
        metadata["timesteps"][str(dt)] = item
    baseline, finer = frames
    if baseline.scenario_id.tolist() != finer.scenario_id.tolist():
        raise AssertionError("Scenario ordering changed between time steps.")
    checks: dict[str, Any] = {}
    for field in ("MP_vent_PEEP_Jmin", "MP_tot_PEEP_Jmin", "MP_mus_Jmin", "ratio_PEEP",
                  "ECF_PEEP", "EF1_PEEP", "EII_PEEP"):
        a, b = baseline[field].to_numpy(), finer[field].to_numpy()
        absolute = np.abs(b-a)
        nonzero = np.abs(a) > 1e-12
        checks[field] = {"max_absolute_change": float(absolute.max()),
            "max_relative_change_percent_nonzero_baseline": float(np.max(100*absolute[nonzero]/np.abs(a[nonzero])))}
    metadata["time_step_refinement"] = checks
    (output/"R3C1_summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Results saved to {output}")
    print(json.dumps(metadata["timesteps"]["0.001"], indent=2, allow_nan=False))
    print("Time-step refinement:", json.dumps(checks, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
