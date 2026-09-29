# PEEP-inclusive pressure-reference sensitivity (Reviewer 3, Comment 1)

This directory contains the supplementary analysis added during peer review to test
whether the study's qualitative findings depend on the primary above-PEEP pressure
reference used for mechanical-power accounting.

## Important scope statement

This analysis is **not** a direct application of the closed-form Eq. (6) from
Gattinoni et al. (Intensive Care Med. 2016;42:1567-1575) to pressure-support
breaths. That expression assumes, among other simplifications, constant flow during
inflation and contains no explicit inspiratory-muscle-pressure term. The present
pressure-support simulations have time-varying, effort-dependent flow.

Instead, the analysis retains the original simulated pressure and flow waveforms and
recalculates work by direct signed pressure-flow integration with PEEP included in the
airway-pressure reference, consistent with the PEEP-inclusive pressure-volume work
convention used as the measured reference by Gattinoni et al.

## Reproduction

Run from the repository root:

```bash
python -m pip install -r supplementary/peep_inclusive/requirements.txt
python run_R3C1_peep_reference.py
```

The script imports the unchanged `assisted_ventilation_model_final.py`, reproduces the
180 primary scenarios at `dt = 0.001 s` and `dt = 0.0005 s`, recalculates PEEP-inclusive
ventilator, total, and compartmental energies, and repeats the original non-exclusive
5% matched-pair procedure using PEEP-inclusive ventilator and total power.

By default, outputs are written to `supplementary/peep_inclusive/data/`.

## Main baseline results (dt = 0.001 s)

- PEEP-inclusive ventilator power: 6.16-116.10 J/min.
- PEEP-inclusive total modeled power: 6.16-159.23 J/min.
- PEEP-inclusive total-to-ventilator power ratio: 1.00-2.50.
- Matching by PEEP-inclusive ventilator power: 603 pairs; median absolute difference
  in corresponding PEEP-inclusive total power = 8.83 J/min.
- Matching by PEEP-inclusive total power: 569 pairs; median absolute differences in
  EF1 and EII = 0.0735 and 0.1470, respectively.
- Time-step refinement changed PEEP-inclusive ventilator and total power by at most
  0.0742% across the primary scenarios.

These numerical values demonstrate robustness of the qualitative non-uniqueness
finding to the pressure-reference convention; they do not validate a bedside equation
for assisted ventilation and are not clinical thresholds.

## Files

- `data/R3C1_scenarios_dt0p001.csv` - all 180 scenarios at the primary time step.
- `data/R3C1_scenarios_dt0p0005.csv` - time-step refinement grid.
- `data/R3C1_pairs_MP_vent_PEEP_Jmin_dt0p001.csv` - pairs matched by PEEP-inclusive ventilator power.
- `data/R3C1_pairs_MP_tot_PEEP_Jmin_dt0p001.csv` - pairs matched by PEEP-inclusive total power.
- Corresponding `dt0p0005` pair files - time-step refinement outputs.
- `data/R3C1_summary.json` - machine-readable ranges, pair summaries, environment metadata,
  model SHA-256, and numerical checks.
- `Supplementary_addendum_R3C1.md` - manuscript/supplement-ready methodological summary.

## Reference

Gattinoni L, Tonetti T, Cressoni M, et al. Ventilator-related causes of lung injury:
the mechanical power. Intensive Care Med. 2016;42:1567-1575.
DOI: 10.1007/s00134-016-4505-2.
