"""Targeted mechanical-heterogeneity sensitivity analysis for Reviewer 1, comment 6.

The original model implementation is imported without modification.  This script
changes only compartment-2 compliance and resistance, uses the original 45
ventilator/effort input combinations, and compares two integration time steps.
Outputs describe a supplementary analysis, NOT a replacement of the 180-scenario
main analysis. Signed flow and the original inspiratory integration mask are
preserved exactly.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import assisted_ventilation_model_final as model

C2_VALUES = [round(0.02 + 0.005 * i, 6) for i in range(17)]
R2_VALUES = [float(i) for i in range(4, 17)]
PS_VALUES = [5.0, 10.0, 15.0]
AMPLITUDES = [0.0, 3.0, 6.0, 9.0, 12.0]
DURATIONS = [0.6, 1.0, 1.4]


def run_configuration(task: tuple[float, float, float]) -> list[dict]:
    c2, r2, dt = task
    c1 = model.Compartment(0.04, 8.0)
    comp2 = model.Compartment(c2, r2)
    rows = []
    for ps, amp, duration in itertools.product(PS_VALUES, AMPLITUDES, DURATIONS):
        vent = model.VentilatorSettings(5.0, ps, 18.0, 1.0, dt)
        effort = model.EffortSettings(amp, duration, 0.0)
        sim = model.simulate_breath(c1, comp2, vent, effort)
        row = model.compute_metrics(sim, c1, comp2, vent)
        row.update(Pmus_peak_cmH2O=amp, Tmus_s=duration, Pmus_onset_s=0.0,
                   dt_s=dt, tau1_s=0.32, tau2_s=c2 * r2)
        # These volume values describe the original initialized single breath;
        # they do not imply a multi-breath steady-state/auto-PEEP simulation.
        rows.append(row)
    return rows


def make_sweep(dt: float, workers: int) -> pd.DataFrame:
    tasks = [(c, r, dt) for c, r in itertools.product(C2_VALUES, R2_VALUES)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, batch in enumerate(pool.map(run_configuration, tasks)):
            rows.extend(batch)
            if (i + 1) % 26 == 0 or i + 1 == len(tasks):
                print(f'dt={dt}: {i + 1}/{len(tasks)} mechanical configurations', flush=True)
    df = pd.DataFrame(rows)
    if len(df) != 9945 or not np.isfinite(df.select_dtypes(include='number')).all().all():
        raise RuntimeError('Unexpected row count or non-finite numerical output.')
    if df['E_tot_J'].min() <= 0 or df['E1_tot_J'].min() <= 0 or df['E2_tot_J'].min() <= 0:
        raise RuntimeError('Nonpositive net inspiratory energy; re-evaluate index interpretations.')
    if df['Energy_balance_error_J'].max() > 1e-10 or df['Partition_balance_error'].max() > 1e-10:
        raise RuntimeError('Energy or partition identity check failed.')
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('R1C6_outputs'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    baseline = make_sweep(0.001, args.workers)
    baseline.to_csv(out/'sensitivity_grid_dt_0p001.csv', index=False)
    refined = make_sweep(0.0005, args.workers)
    refined.to_csv(out/'sensitivity_grid_dt_0p0005.csv', index=False)
    keys = ['C2_L_cmH2O','R2_cmH2O_s_L','PS_cmH2O','Pmus_peak_cmH2O','Tmus_s']
    if not baseline[keys].equals(refined[keys]):
        raise RuntimeError('The two sweeps are not aligned.')
    comp_keys = ['C2_L_cmH2O','R2_cmH2O_s_L']
    summary = baseline.groupby(comp_keys).agg(
        MPtot_min=('MP_tot_Jmin','min'), MPtot_max=('MP_tot_Jmin','max'),
        EF1_min=('REF1','min'), EF1_max=('REF1','max'),
        EII_min=('EII','min'), EII_max=('EII','max'),
    ).reset_index()
    summary['EII_range'] = summary.EII_max - summary.EII_min
    summary['EF1_range'] = summary.EF1_max - summary.EF1_min
    summary['preference_reversal_across_loading_inputs'] = (summary.EF1_min < 0.5 - 1e-10) & (summary.EF1_max > 0.5 + 1e-10)
    summary.to_csv(out/'per_mechanical_configuration_ranges.csv',index=False)
    amp_ranges = baseline.groupby(comp_keys + ['PS_cmH2O','Tmus_s']).agg(
        EII_min=('EII','min'), EII_max=('EII','max'),
        EF1_min=('REF1','min'), EF1_max=('REF1','max'),
    ).reset_index()
    amp_ranges['EII_range'] = amp_ranges.EII_max - amp_ranges.EII_min
    amp_ranges['EF1_range'] = amp_ranges.EF1_max - amp_ranges.EF1_min
    amp_ranges.to_csv(out/'effort_amplitude_ranges_at_fixed_PS_and_duration.csv',index=False)
    fixed = baseline[(baseline.PS_cmH2O==10) & (baseline.Pmus_peak_cmH2O==6) & (baseline.Tmus_s==1)].copy()
    fixed.to_csv(out/'fixed_input_surface_PS10_Pmus6_Tmus1.csv',index=False)
    examples = [(0.04,8),(0.08,8),(0.04,16),(0.08,4),(0.10,4),(0.08,16),(0.10,16),(0.02,4),(0.06,16)]
    example_rows=[]
    for c,r in examples:
        e=fixed[np.isclose(fixed.C2_L_cmH2O,c)&np.isclose(fixed.R2_cmH2O_s_L,r)].iloc[0]
        s=summary[np.isclose(summary.C2_L_cmH2O,c)&np.isclose(summary.R2_cmH2O_s_L,r)].iloc[0]
        example_rows.append(dict(C2=c,R2=r,tau2_s=c*r,MPtot_fixed=e.MP_tot_Jmin,EF1_fixed=e.REF1,EII_fixed=e.EII,
                                 EF1_min=s.EF1_min,EF1_max=s.EF1_max,EII_min=s.EII_min,EII_max=s.EII_max))
    pd.DataFrame(example_rows).to_csv(out/'Supplementary_Table_S1_selected_configurations.csv',index=False)
    delta_mp=(refined.MP_tot_Jmin-baseline.MP_tot_Jmin).abs()/baseline.MP_tot_Jmin*100
    delta_eii=(refined.EII-baseline.EII).abs()
    delta_ef1=(refined.REF1-baseline.REF1).abs()
    results = {
        'n_C2_levels':len(C2_VALUES),'n_R2_levels':len(R2_VALUES),
        'n_mechanical_configurations':len(summary),'n_input_combinations_per_configuration':45,
        'n_supplementary_conditions_per_dt':len(baseline),
        'n_distinct_pressure_inputs_per_configuration':39,
        'C2_range_L_per_cmH2O':[min(C2_VALUES),max(C2_VALUES)],'C2_step':0.005,
        'R2_range_cmH2O_s_per_L':[min(R2_VALUES),max(R2_VALUES)],'R2_step':1.0,
        'tau2_range_s':[float(baseline.tau2_s.min()),float(baseline.tau2_s.max())],
        'MPtot_range_Jmin':[float(baseline.MP_tot_Jmin.min()),float(baseline.MP_tot_Jmin.max())],
        'EF1_range':[float(baseline.REF1.min()),float(baseline.REF1.max())],
        'EII_range':[float(baseline.EII.min()),float(baseline.EII.max())],
        'max_EII_range_across_loading_inputs':summary.loc[summary.EII_range.idxmax()].to_dict(),
        'max_EF1_range_across_loading_inputs':summary.loc[summary.EF1_range.idxmax()].to_dict(),
        'max_EII_range_across_amplitude_at_fixed_PS_Tmus':amp_ranges.loc[amp_ranges.EII_range.idxmax()].to_dict(),
        'max_EF1_range_across_amplitude_at_fixed_PS_Tmus':amp_ranges.loc[amp_ranges.EF1_range.idxmax()].to_dict(),
        'n_mechanical_configurations_with_EF1_crossing_0p5_across_loading_inputs':int(summary.preference_reversal_across_loading_inputs.sum()),
        'refinement_max_relative_MPtot_difference_percent':float(delta_mp.max()),
        'refinement_median_relative_MPtot_difference_percent':float(delta_mp.median()),
        'refinement_max_absolute_EII_difference':float(delta_eii.max()),
        'refinement_max_absolute_EF1_difference':float(delta_ef1.max()),
        'max_energy_balance_error_J':float(baseline.Energy_balance_error_J.max()),
        'max_partition_balance_error':float(baseline.Partition_balance_error.max()),
        'source_model_sha256':hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
        'python_version':platform.python_version(),'numpy_version':np.__version__,'pandas_version':pd.__version__,
        'elapsed_seconds':round(time.time()-start,2),
    }
    (out/'analysis_summary.json').write_text(json.dumps(results,indent=2,default=lambda x: x.item() if isinstance(x,np.generic) else str(x)))
    print(json.dumps(results,indent=2),flush=True)

if __name__=='__main__':
    main()
