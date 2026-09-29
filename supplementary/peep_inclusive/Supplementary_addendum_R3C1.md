# Sensitivity to inclusion of PEEP in power accounting

## Rationale and relation to Gattinoni et al. (2016)

Gattinoni et al. derived their closed-form mechanical-power equation from a linear equation of motion without an explicit muscle-pressure term and assumed constant flow during inflation. They compared the equation with the work obtained by integrating the inspiratory airway pressure-volume relationship, including PEEP in the pressure reference. The current pressure-support model instead has time-varying, effort-dependent flow and calculates primary energies above PEEP. Therefore, the present analysis uses the PEEP-inclusive work integral rather than applying their constant-flow closed-form expression directly to assisted breaths.

For a passive linear single-compartment system at constant flow, the integral reduces to

MP = k RR [Elrs VT^2/2 + Raw VT^2/Ti + PEEP VT],

which is Eq. (6) of Gattinoni et al. after substituting 1/Ti = RR(1+I:E)/(60 I:E), apart from the rounding of the unit-conversion factor. This reduction establishes the relation between the work integral and the original closed-form equation; it is not a validation of that shortcut under assisted ventilation.

## Methods

All 180 primary scenarios were evaluated without changing the original equations, prescribed inputs, or initial conditions. PEEP remained 5 cmH2O, RR 18/min, and Ti 1.0 s. The original signed trapezoidal pressure-flow integration and inspiratory mask (t < Ti) were retained. The calculation was repeated at dt = 0.001 s and 0.0005 s. Let I denote the ventilator-defined inspiratory integration window and k = 0.0980665 J/(cmH2O L).

MPvent,PEEP = k RR integral_I Paw(t) Qtotal(t) dt.

MPtot,PEEP = k RR integral_I [Paw(t) + Pmus(t)] Qtotal(t) dt.

Ei,PEEP = k integral_I [Paw(t) + Pmus(t)] Qi(t) dt.

The muscle-generated component was unchanged. The additional PEEP term, k RR PEEP integral_I Qtotal(t) dt, was added consistently to ventilator and total power. Compartmental energy fractions, EII, ECF, and the total-to-ventilator ratio were recalculated using the PEEP-inclusive energies. These quantities were stored separately from the primary above-PEEP outputs.

For the square pressure-support waveform, MPvent,PEEP = (1 + PEEP/PS) MPvent. This is a scenario-level identity, not an independent simulation result. Because PS varied, the rescaling factor was not common to all scenarios, and the matched-pair analyses were repeated using the PEEP-inclusive power values.

Pairs were matched separately by MPvent,PEEP and MPtot,PEEP using the original non-exclusive comparison procedure, scenario ordering, and 5% relative tolerance: abs(xj-xi)/xi <= 0.05 for i < j. The repeated passive entries at the three prescribed effort durations were retained, consistently with the primary analysis.

## Results at dt = 0.001 s

PEEP-inclusive ventilator power ranged from 6.16 to 116.10 J/min, and PEEP-inclusive total power from 6.16 to 159.23 J/min. Muscle-generated power ranged from 0 to 43.14 J/min. The PEEP-inclusive total-to-ventilator ratio ranged from 1.00 to 2.50; phenotype-specific maxima were 2.23, 2.30, 2.50, and 2.39 for the compliance-dominant, resistance-dominant, mixed, and severe mixed configurations, respectively.

| Matching quantity | Number of pairs | Absolute difference in PEEP-inclusive total power, median (maximum), J/min | Absolute difference in muscle power, median (maximum), J/min | Absolute difference in PEEP-inclusive EF1, median (maximum) | Absolute difference in PEEP-inclusive EII, median (maximum) |
|---|---:|---:|---:|---:|---:|
| PEEP-inclusive ventilator power | 603 | 8.83 (37.13) | 8.46 (36.03) | 0.0418 (0.2580) | 0.0835 (0.3499) |
| PEEP-inclusive total power | 569 | 1.05 (5.26) | 7.11 (29.95) | 0.0735 (0.2574) | 0.1470 (0.3514) |

For a concrete example, in the severe mixed configuration, PS 10 cmH2O with peak Pmus 12 cmH2O and Tmus 1.4 s yielded PEEP-inclusive ventilator power of 69.88 J/min, muscle power of 36.03 J/min, and total power of 105.91 J/min. A passive scenario at PS 15 cmH2O yielded ventilator and total power of 68.77 J/min. Thus, similar PEEP-inclusive ventilator power coexisted with different total loading within the same mechanical configuration. This is an illustrative model comparison, not a clinical example.

## Numerical checks

The original above-PEEP pairing counts (505 and 534) were reproduced at the baseline time step. PEEP-inclusive total power equaled ventilator plus muscle power to numerical precision, with a maximum absolute discrepancy below 4e-14 J/min. The compartmental fractions summed to one and the ratio equaled 1/(1-ECF), within numerical precision.

Refining the time step changed PEEP-inclusive ventilator and total power by at most 0.0742% across the primary scenarios. The maximum absolute change in PEEP-inclusive EII was 0.0001603. At dt = 0.0005 s, the PEEP-inclusive matching yielded 604 ventilator-power-matched pairs and 569 total-power-matched pairs. The one-pair change for ventilator matching reflects sensitivity near the 5% boundary. Detailed descriptive summaries at both time steps are provided in R3C1_summary.json; invariance of pair counts under time-step refinement is not claimed.

## Interpretation and limits

The numerical magnitude of power and its relative source-contribution ratio depends on the pressure-reference convention. Nevertheless, residual differences in total modeled loading and compartmental energy allocation remained after matching on the PEEP-inclusive power measures. Adding the same PEEP term to ventilator and total power necessarily leaves their difference equal to muscle power; that identity is not interpreted as an independent finding.

This analysis tests pressure-reference sensitivity at fixed prescribed mechanics and PEEP. It is not an assessment of physiological responses to PEEP titration, validation against patient data, or a direct application of the original constant-flow closed-form equation to pressure-support ventilation. The original primary calculations, figures, and analyses remain unchanged.

Reference: Gattinoni L, Tonetti T, Cressoni M, et al. Ventilator-related causes of lung injury: the mechanical power. Intensive Care Med. 2016;42:1567-1575. DOI: 10.1007/s00134-016-4505-2.
