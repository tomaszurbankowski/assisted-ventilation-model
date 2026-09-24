# Supplementary sensitivity analysis of mechanical heterogeneity

This folder supports Additional file 1 and the supplementary compliance-resistance
analysis. It supplements, rather than replaces, the primary 180-scenario analysis.
The driver is `run_R1C6_sensitivity.py` at the repository root. It imports the existing
`assisted_ventilation_model_final.py` without modifying the model equations,
integration, or metric calculations.

## Design and scope

- C2: 0.020 to 0.100 L/cmH2O, in 0.005 increments (17 levels).
- R2: 4 to 16 cmH2O*s/L, in increments of 1 (13 levels).
- Reference compartment: C1 = 0.04 L/cmH2O; R1 = 8 cmH2O*s/L.
- Pressure support: 5, 10, 15 cmH2O; peak Pmus: 0, 3, 6, 9, 12 cmH2O;
  effort duration: 0.6, 1.0, 1.4 s.
- Fixed PEEP: 5 cmH2O; respiratory rate: 18/min; ventilator inspiration: 1.0 s.
- 221 mechanical configurations x 45 input combinations = 9,945 rows per time step.
- Both dt = 0.001 s and dt = 0.0005 s are included.

There are 39 distinct pressure-input histories per mechanical configuration because
zero-effort conditions repeat across the three duration labels. These deterministic
combinations are not independent patient observations. The illustrative grid includes
the four primary configurations; it is not 9,945 new configurations in addition to
the original 180 scenarios. Matched-pair comparisons were not repeated on the grid.

## Reproduce from the repository root

```bash
python -m pip install -r supplementary/sensitivity/requirements.txt
python run_R1C6_sensitivity.py --output-dir reproduced_sensitivity --workers 4
```

The recorded execution environment was Python 3.13.5, NumPy 2.3.5, and pandas 2.2.3.
Use a suitable Python environment for these recorded package versions. The core model
uses `numpy.trapezoid`. The numerical driver does not require matplotlib; the existing
figure-generation scripts do. Reproduction writes to a new output directory, leaving
the deposited data and the primary outputs unchanged. The primary model source used
for the deposited outputs has SHA-256:

`505c9d2b9ea651c67ba24655fbb8dd55d82c69ea0075114ef3780f550e6b4f70`

Check a local model copy from the repository root with:

```bash
python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('assisted_ventilation_model_final.py').read_bytes()).hexdigest())"
```

This source hash identifies the model used for these results. If your checkout has a
different hash, inspect the source differences before claiming exact reproduction.
The driver records the executed source hash in its newly generated analysis summary.

## Files

- `Additional_file_1.docx`: supplementary methods, selected results, Table S1,
  mathematical explanation, and time-step checks.
- `data/sensitivity_grid_dt_0p001.csv`: all 9,945 baseline-step combinations.
- `data/sensitivity_grid_dt_0p0005.csv`: the same combinations at the refined step.
- `data/per_mechanical_configuration_ranges.csv`: results over all loading inputs
  for each of the 221 mechanical configurations.
- `data/fixed_input_surface_PS10_Pmus6_Tmus1.csv`: 221 configurations at fixed inputs.
- `data/effort_amplitude_ranges_at_fixed_PS_and_duration.csv`: effort-amplitude ranges
  with mechanics, pressure support, and effort duration held fixed.
- `data/Supplementary_Table_S1_selected_configurations.csv`: selected configurations.
- `data/baseline_180_reproduced.csv`: separately reproduced primary-sweep results;
  generated using the original model's `run_parameter_sweep`, not by this driver.
- `data/analysis_summary.json`: result ranges, numerical checks, source hash, and
  recorded execution environment. Runtime is execution-specific.
- `requirements.txt`: recorded numerical package versions.
- `MANIFEST_SHA256.txt`: checksums relative to the repository root.

## Column naming and interpretation

The existing code's `REF1` and `REF2` columns correspond to EF1 and EF2 in the manuscript.
The legacy `HBR` column is MPtot/MPvent, not a separately validated clinical index.
Energies are in joules, power in J/min, and energy fractions, EII, ECF, and the
power ratio are dimensionless. Other units are included in the CSV column names.

All energies retain signed flow, zero initial volume increments, and the original
`t < Ti` integration mask. Negative pressure-flow products are not clipped.
The analysis concerns net above-PEEP energy transfer in an initialized single breath,
not positive-only energy, tissue dissipation, a steady-state repeated-breath simulation,
or a clinically validated injury metric.

Equal time constants in the primary mixed configuration impose EF1 = EII = 1/3.
The broader grid shows that limited effort-related variation in EII does not apply to
all configurations. Opposing compliance/resistance changes can partly offset each
other and can change the compartment with the larger energy fraction across inputs.
Time-step agreement assesses numerical precision, not physiological validity.
