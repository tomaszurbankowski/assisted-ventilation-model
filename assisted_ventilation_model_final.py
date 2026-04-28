"""
Two-compartment model of assisted ventilation with patient inspiratory effort.

This implementation is intended for physiology-oriented in-silico analyses of how
ventilator assistance and patient effort jointly shape total inspiratory mechanical
loading and its distribution between parallel lung compartments.

Modeling conventions
--------------------
- The respiratory system is represented by two parallel linear RC compartments.
- PEEP is treated as baseline pressure; energies are computed above PEEP during inspiration.
- Patient inspiratory effort is represented by a positive-valued waveform denoting the
  magnitude of inspiratory muscle pressure contribution. With this sign convention,
  ventilator assistance and patient effort add to the net distending pressure.
- The model is deliberately simplified and does not include chest wall mechanics,
  nonlinear recruitment/derecruitment, hysteresis, triggering/cycling dynamics,
  auto-PEEP, or gas redistribution beyond that implied by parallel RC mechanics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Literal

import numpy as np
import pandas as pd

CMH2O_L_TO_J = 0.0980665


@dataclass(frozen=True)
class Compartment:
    compliance: float  # L/cmH2O
    resistance: float  # cmH2O*s/L

    def __post_init__(self) -> None:
        if self.compliance <= 0:
            raise ValueError("Compliance must be > 0.")
        if self.resistance <= 0:
            raise ValueError("Resistance must be > 0.")


@dataclass(frozen=True)
class VentilatorSettings:
    peep: float                  # cmH2O
    pressure_support: float      # cmH2O above PEEP
    respiratory_rate: float      # breaths/min
    inspiratory_time: float      # s
    dt: float = 0.001            # s

    def __post_init__(self) -> None:
        if self.peep < 0:
            raise ValueError("PEEP must be >= 0.")
        if self.pressure_support < 0:
            raise ValueError("Pressure support must be >= 0.")
        if self.respiratory_rate <= 0:
            raise ValueError("Respiratory rate must be > 0.")
        if self.inspiratory_time <= 0:
            raise ValueError("Inspiratory time must be > 0.")
        if self.dt <= 0:
            raise ValueError("dt must be > 0.")
        cycle_time = 60.0 / self.respiratory_rate
        if self.inspiratory_time >= cycle_time:
            raise ValueError("Inspiratory time must be shorter than total cycle time.")


@dataclass(frozen=True)
class EffortSettings:
    pmus_peak: float             # cmH2O; positive magnitude of inspiratory effort
    effort_duration: float       # s
    onset: float = 0.0           # s relative to inspiratory onset

    def __post_init__(self) -> None:
        if self.pmus_peak < 0:
            raise ValueError("pmus_peak must be >= 0.")
        if self.effort_duration < 0:
            raise ValueError("effort_duration must be >= 0.")
        if self.onset < 0:
            raise ValueError("onset must be >= 0.")


PhenotypeName = Literal[
    "symmetric",
    "compliance_dominant",
    "resistance_dominant",
    "mixed",
    "severe_mixed",
]


def square_pressure_support_waveform(
    t: np.ndarray,
    peep: float,
    pressure_support: float,
    inspiratory_time: float,
) -> np.ndarray:
    paw = np.full_like(t, peep, dtype=float)
    insp_mask = (t >= 0.0) & (t < inspiratory_time)
    paw[insp_mask] = peep + pressure_support
    return paw


def half_sine_pmus_waveform(
    t: np.ndarray,
    pmus_peak: float,
    effort_duration: float,
    onset: float = 0.0,
) -> np.ndarray:
    pmus = np.zeros_like(t, dtype=float)
    if pmus_peak == 0 or effort_duration == 0:
        return pmus

    start = onset
    end = onset + effort_duration
    mask = (t >= start) & (t < end)
    tau = (t[mask] - start) / effort_duration
    pmus[mask] = pmus_peak * np.sin(np.pi * tau)
    return pmus


def inspiratory_mask(t: np.ndarray, inspiratory_time: float) -> np.ndarray:
    return (t >= 0.0) & (t < inspiratory_time)


def trapz_integral(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def phenotype_definition(name: PhenotypeName) -> Dict[str, Compartment]:
    base = Compartment(compliance=0.04, resistance=8.0)

    definitions = {
        "symmetric": Compartment(compliance=0.04, resistance=8.0),
        "compliance_dominant": Compartment(compliance=0.08, resistance=8.0),
        "resistance_dominant": Compartment(compliance=0.04, resistance=16.0),
        "mixed": Compartment(compliance=0.08, resistance=4.0),
        "severe_mixed": Compartment(compliance=0.10, resistance=4.0),
    }
    if name not in definitions:
        raise ValueError(f"Unknown phenotype: {name}")
    return {"comp1": base, "comp2": definitions[name]}


def simulate_breath(
    comp1: Compartment,
    comp2: Compartment,
    vent: VentilatorSettings,
    effort: EffortSettings,
) -> Dict[str, np.ndarray]:
    cycle_time = 60.0 / vent.respiratory_rate
    t = np.arange(0.0, cycle_time + vent.dt, vent.dt)

    paw = square_pressure_support_waveform(
        t=t,
        peep=vent.peep,
        pressure_support=vent.pressure_support,
        inspiratory_time=vent.inspiratory_time,
    )
    pmus = half_sine_pmus_waveform(
        t=t,
        pmus_peak=effort.pmus_peak,
        effort_duration=effort.effort_duration,
        onset=effort.onset,
    )

    paw_above_peep = paw - vent.peep
    pdist = paw_above_peep + pmus

    v1 = np.zeros_like(t, dtype=float)
    v2 = np.zeros_like(t, dtype=float)
    f1 = np.zeros_like(t, dtype=float)
    f2 = np.zeros_like(t, dtype=float)
    f_tot = np.zeros_like(t, dtype=float)

    for k in range(len(t) - 1):
        elastic1 = v1[k] / comp1.compliance
        elastic2 = v2[k] / comp2.compliance
        f1[k] = (pdist[k] - elastic1) / comp1.resistance
        f2[k] = (pdist[k] - elastic2) / comp2.resistance
        f_tot[k] = f1[k] + f2[k]
        v1[k + 1] = v1[k] + f1[k] * vent.dt
        v2[k + 1] = v2[k] + f2[k] * vent.dt

    elastic1_last = v1[-1] / comp1.compliance
    elastic2_last = v2[-1] / comp2.compliance
    f1[-1] = (pdist[-1] - elastic1_last) / comp1.resistance
    f2[-1] = (pdist[-1] - elastic2_last) / comp2.resistance
    f_tot[-1] = f1[-1] + f2[-1]

    return {
        "t": t,
        "paw": paw,
        "paw_above_peep": paw_above_peep,
        "pmus": pmus,
        "pdist": pdist,
        "v1": v1,
        "v2": v2,
        "vtot": v1 + v2,
        "f1": f1,
        "f2": f2,
        "f_tot": f_tot,
    }


def compute_metrics(
    sim: Dict[str, np.ndarray],
    comp1: Compartment,
    comp2: Compartment,
    vent: VentilatorSettings,
) -> Dict[str, float]:
    t = sim["t"]
    paw_above_peep = sim["paw_above_peep"]
    pmus = sim["pmus"]
    total_distending = sim["pdist"]
    f1 = sim["f1"]
    f2 = sim["f2"]
    f_tot = sim["f_tot"]
    v1 = sim["v1"]
    v2 = sim["v2"]
    vtot = sim["vtot"]

    insp = inspiratory_mask(t, vent.inspiratory_time)

    e_vent_raw = trapz_integral(paw_above_peep[insp] * f_tot[insp], t[insp])
    e_mus_raw = trapz_integral(pmus[insp] * f_tot[insp], t[insp])
    e_tot_raw = trapz_integral(total_distending[insp] * f_tot[insp], t[insp])

    e1_tot_raw = trapz_integral(total_distending[insp] * f1[insp], t[insp])
    e2_tot_raw = trapz_integral(total_distending[insp] * f2[insp], t[insp])

    e_vent = e_vent_raw * CMH2O_L_TO_J
    e_mus = e_mus_raw * CMH2O_L_TO_J
    e_tot = e_tot_raw * CMH2O_L_TO_J
    e1_tot = e1_tot_raw * CMH2O_L_TO_J
    e2_tot = e2_tot_raw * CMH2O_L_TO_J

    rr = vent.respiratory_rate
    mp_vent = e_vent * rr
    mp_mus = e_mus * rr
    mp_tot = e_tot * rr

    ref1 = e1_tot / e_tot if e_tot > 0 else np.nan
    ref2 = e2_tot / e_tot if e_tot > 0 else np.nan
    eii = abs(e1_tot - e2_tot) / e_tot if e_tot > 0 else np.nan
    ecf = mp_mus / mp_tot if mp_tot > 0 else np.nan
    hbr = mp_tot / mp_vent if mp_vent > 0 else np.nan

    vt1_end_insp = float(np.interp(vent.inspiratory_time, t, v1))
    vt2_end_insp = float(np.interp(vent.inspiratory_time, t, v2))
    vt_end_insp = float(np.interp(vent.inspiratory_time, t, vtot))

    flow1_peak = float(np.max(f1[insp]))
    flow2_peak = float(np.max(f2[insp]))
    flow_total_peak = float(np.max(f_tot[insp]))
    pdist_peak = float(np.max(total_distending[insp]))

    return {
        "E_vent_J": e_vent,
        "E_mus_J": e_mus,
        "E_tot_J": e_tot,
        "E1_tot_J": e1_tot,
        "E2_tot_J": e2_tot,
        "MP_vent_Jmin": mp_vent,
        "MP_mus_Jmin": mp_mus,
        "MP_tot_Jmin": mp_tot,
        "REF1": ref1,
        "REF2": ref2,
        "EII": eii,
        "ECF": ecf,
        "HBR": hbr,
        "VT1_L": vt1_end_insp,
        "VT2_L": vt2_end_insp,
        "VT_total_L": vt_end_insp,
        "Flow1_peak_Ls": flow1_peak,
        "Flow2_peak_Ls": flow2_peak,
        "Flow_total_peak_Ls": flow_total_peak,
        "Pdist_peak_cmH2O": pdist_peak,
        "C1_L_cmH2O": comp1.compliance,
        "R1_cmH2O_s_L": comp1.resistance,
        "C2_L_cmH2O": comp2.compliance,
        "R2_cmH2O_s_L": comp2.resistance,
        "PEEP_cmH2O": vent.peep,
        "PS_cmH2O": vent.pressure_support,
        "RR_bpm": vent.respiratory_rate,
        "Ti_s": vent.inspiratory_time,
        "Energy_balance_error_J": abs(e_tot - (e_vent + e_mus)),
        "Partition_balance_error": abs((ref1 + ref2) - 1.0) if not np.isnan(ref1 + ref2) else np.nan,
    }


def run_single_scenario(
    phenotype: PhenotypeName,
    pressure_support: float,
    pmus_peak: float,
    effort_duration: float,
    peep: float = 5.0,
    rr: float = 18.0,
    ti: float = 1.0,
    dt: float = 0.001,
    onset: float = 0.0,
) -> Dict[str, float]:
    ph = phenotype_definition(phenotype)
    comp1 = ph["comp1"]
    comp2 = ph["comp2"]

    vent = VentilatorSettings(
        peep=peep,
        pressure_support=pressure_support,
        respiratory_rate=rr,
        inspiratory_time=ti,
        dt=dt,
    )
    effort = EffortSettings(
        pmus_peak=pmus_peak,
        effort_duration=effort_duration,
        onset=onset,
    )

    sim = simulate_breath(comp1, comp2, vent, effort)
    metrics = compute_metrics(sim, comp1, comp2, vent)
    metrics.update(
        {
            "phenotype": phenotype,
            "Pmus_peak_cmH2O": pmus_peak,
            "Tmus_s": effort_duration,
            "Pmus_onset_s": onset,
        }
    )
    return metrics


def run_parameter_sweep() -> pd.DataFrame:
    phenotypes: List[PhenotypeName] = [
        "compliance_dominant",
        "resistance_dominant",
        "mixed",
        "severe_mixed",
    ]
    pressure_support_levels = [5.0, 10.0, 15.0]
    pmus_peaks = [0.0, 3.0, 6.0, 9.0, 12.0]
    effort_durations = [0.6, 1.0, 1.4]

    rows: List[Dict[str, float]] = []
    for phenotype in phenotypes:
        for ps in pressure_support_levels:
            for pmus_peak in pmus_peaks:
                for tmus in effort_durations:
                    rows.append(
                        run_single_scenario(
                            phenotype=phenotype,
                            pressure_support=ps,
                            pmus_peak=pmus_peak,
                            effort_duration=tmus,
                            peep=5.0,
                            rr=18.0,
                            ti=1.0,
                            dt=0.001,
                        )
                    )
    return pd.DataFrame(rows)


def matched_pairs(
    df: pd.DataFrame,
    match_variable: str = "MP_vent_Jmin",
    tolerance: float = 0.05,
) -> pd.DataFrame:
    if match_variable not in df.columns:
        raise ValueError(f"Unknown match variable: {match_variable}")

    rows: List[Dict[str, float]] = []
    df = df.reset_index(drop=True)

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            x_i = float(df.loc[i, match_variable])
            x_j = float(df.loc[j, match_variable])
            if x_i <= 0:
                continue

            rel_diff = abs(x_j - x_i) / x_i
            if rel_diff <= tolerance:
                rows.append(
                    {
                        "idx_i": i,
                        "idx_j": j,
                        "match_variable": match_variable,
                        "phenotype_i": df.loc[i, "phenotype"],
                        "phenotype_j": df.loc[j, "phenotype"],
                        f"{match_variable}_i": x_i,
                        f"{match_variable}_j": x_j,
                        "relative_difference": rel_diff,
                        "delta_MP_vent": abs(df.loc[j, "MP_vent_Jmin"] - df.loc[i, "MP_vent_Jmin"]),
                        "delta_MP_tot": abs(df.loc[j, "MP_tot_Jmin"] - df.loc[i, "MP_tot_Jmin"]),
                        "delta_EII": abs(df.loc[j, "EII"] - df.loc[i, "EII"]),
                        "delta_REF1": abs(df.loc[j, "REF1"] - df.loc[i, "REF1"]),
                        "delta_ECF": abs(df.loc[j, "ECF"] - df.loc[i, "ECF"]),
                        "delta_HBR": abs(df.loc[j, "HBR"] - df.loc[i, "HBR"]),
                    }
                )
    return pd.DataFrame(rows)


def internal_validation(dt_reference: float = 0.001, dt_test: float = 0.0005) -> pd.DataFrame:
    checks: List[Dict[str, float | str | bool]] = []

    symmetric = run_single_scenario(
        phenotype="symmetric",
        pressure_support=10.0,
        pmus_peak=0.0,
        effort_duration=0.0,
        dt=dt_reference,
    )
    checks.append(
        {
            "test": "Symmetric phenotype gives equal energy partition",
            "passed": math.isclose(symmetric["REF1"], 0.5, rel_tol=1e-3, abs_tol=1e-3)
            and math.isclose(symmetric["REF2"], 0.5, rel_tol=1e-3, abs_tol=1e-3),
            "value": symmetric["REF1"],
        }
    )

    passive = run_single_scenario(
        phenotype="mixed",
        pressure_support=10.0,
        pmus_peak=0.0,
        effort_duration=0.0,
        dt=dt_reference,
    )
    checks.append(
        {
            "test": "No effort gives zero muscular power",
            "passed": math.isclose(passive["MP_mus_Jmin"], 0.0, abs_tol=1e-10),
            "value": passive["MP_mus_Jmin"],
        }
    )
    checks.append(
        {
            "test": "Energy balance E_tot = E_vent + E_mus",
            "passed": passive["Energy_balance_error_J"] < 1e-8,
            "value": passive["Energy_balance_error_J"],
        }
    )
    checks.append(
        {
            "test": "Partition fractions sum to 1",
            "passed": passive["Partition_balance_error"] < 1e-8,
            "value": passive["Partition_balance_error"],
        }
    )

    coarse = run_single_scenario(
        phenotype="mixed",
        pressure_support=10.0,
        pmus_peak=6.0,
        effort_duration=1.0,
        dt=dt_reference,
    )
    fine = run_single_scenario(
        phenotype="mixed",
        pressure_support=10.0,
        pmus_peak=6.0,
        effort_duration=1.0,
        dt=dt_test,
    )
    rel_mp_diff = abs(fine["MP_tot_Jmin"] - coarse["MP_tot_Jmin"]) / coarse["MP_tot_Jmin"]
    checks.append(
        {
            "test": "Numerical stability across time step refinement",
            "passed": rel_mp_diff < 0.01,
            "value": rel_mp_diff,
        }
    )

    return pd.DataFrame(checks)


if __name__ == "__main__":
    single = run_single_scenario(
        phenotype="compliance_dominant",
        pressure_support=10.0,
        pmus_peak=6.0,
        effort_duration=1.0,
    )
    print("Single scenario metrics:")
    for k, v in single.items():
        print(f"{k}: {v}")

    df = run_parameter_sweep()
    print("\nSweep summary:")
    print(df.head())
    print(f"\nNumber of scenarios: {len(df)}")

    matched_vent = matched_pairs(df, match_variable="MP_vent_Jmin", tolerance=0.05)
    matched_total = matched_pairs(df, match_variable="MP_tot_Jmin", tolerance=0.05)
    print(f"\nMatched pairs by MP_vent: {len(matched_vent)}")
    print(f"Matched pairs by MP_tot: {len(matched_total)}")

    validation = internal_validation()
    print("\nInternal validation:")
    print(validation)

    df.to_csv("assisted_ventilation_effort_sweep_final.csv", index=False)
    matched_vent.to_csv("assisted_ventilation_mpvent_matched_pairs_final.csv", index=False)
    matched_total.to_csv("assisted_ventilation_mptot_matched_pairs_final.csv", index=False)
    validation.to_csv("assisted_ventilation_internal_validation.csv", index=False)
