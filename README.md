# Two-Compartment Assisted Ventilation Model

This repository contains the Python code used for an in-silico two-compartment model of assisted mechanical ventilation with patient inspiratory effort. The model was developed to analyze how ventilator assistance and patient-generated effort jointly shape total inspiratory mechanical loading and how this load is distributed between two parallel lung compartments.

## Repository contents
- `assisted_ventilation_model_final.py` – main simulation model and metric calculations.  
- `figure_utils.py` – shared plotting and figure-saving utilities.  
- `generate_figure1_conceptual_model.py` – generates the conceptual model figure.  
- `generate_figure2_representative_waveforms.py` – generates representative pressure, flow, and volume waveforms.  
- `generate_figure3_parameter_sweep_heatmaps.py` – generates parameter-sweep heatmaps.  
- `generate_figure4_hidden_burden_scatter.py` – generates scatter plots comparing ventilator and total mechanical power.  
- `generate_figure5_matched_pair_summary.py` – generates matched-pair summary plots.  

## Model overview
The respiratory system is represented as two parallel linear resistance–compliance compartments. Airway pressure above PEEP and patient inspiratory effort are combined into a net distending pressure:  
Pdist(t) = Paw(t) − PEEP + Pmus(t)  

Patient effort is represented as a positive half-sine waveform. With this sign convention, ventilator assistance and patient effort add to the net distending pressure.

The model computes:
- ventilator-delivered inspiratory energy and mechanical power,  
- patient-generated inspiratory energy and mechanical power,  
- total inspiratory energy and mechanical power,  
- regional energy fractions for each compartment,  
- indices describing energy imbalance and hidden patient-effort burden.  

## Main assumptions
- Two parallel linear RC compartments.  
- PEEP is treated as baseline pressure.  
- Energies are calculated above PEEP during inspiration.  
- Patient effort is represented by a simplified half-sine inspiratory muscle pressure waveform.  
- The model does not include chest wall mechanics, nonlinear recruitment/derecruitment, hysteresis, triggering/cycling dynamics, auto-PEEP, or gas redistribution beyond parallel RC mechanics.  

## Requirements
The code requires Python 3 and the following packages:  
pip install numpy pandas matplotlib  

## Basic usage
Run the main model script:  
python assisted_ventilation_model_final.py  

This generates example scenario outputs, parameter-sweep results, matched-pair analyses, and internal validation results.

To generate the figures:  
python generate_figure1_conceptual_model.py  
python generate_figure2_representative_waveforms.py  
python generate_figure3_parameter_sweep_heatmaps.py  
python generate_figure4_hidden_burden_scatter.py  
python generate_figure5_matched_pair_summary.py  

Generated figures are saved in the `outputs/` directory.

## Output files
Running the main model script produces CSV files including:
- assisted_ventilation_effort_sweep_final.csv  
- assisted_ventilation_mpvent_matched_pairs_final.csv  
- assisted_ventilation_mptot_matched_pairs_final.csv  
- assisted_ventilation_internal_validation.csv  

Figure scripts export publication-style figures (PNG/SVG/PDF depending on script).

## Interpretation
The simulations illustrate that ventilator-derived mechanical power may underestimate total mechanical loading during assisted ventilation when patient effort is present. They also show that similar global mechanical power values may be associated with different regional energy distributions in heterogeneous lungs. Ventilator mechanical power, total mechanical power, patient effort contribution, and regional energy partitioning should therefore be interpreted as complementary but non-equivalent descriptors of mechanical load.

## Limitations
This is a simplified physiological model intended for conceptual and mechanistic analysis and should not be interpreted as a patient-specific clinical simulator. Limitations include:
- no chest wall mechanics,  
- no nonlinear pressure–volume behavior,  
- no recruitment or derecruitment,  
- no hysteresis,  
- no trigger or cycling algorithm,  
- no auto-PEEP,  
- no anatomical regional modeling beyond two parallel compartments.  
