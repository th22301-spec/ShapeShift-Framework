
# ShapeShift Framework

ShapeShift is a physics-informed machine learning framework designed to accelerate mechanical simulation of MEX (Material Extrusion) 3D-printed parts. It replaces high-fidelity filament-level FEA models with fast, accurate solid surrogate models, enabling efficient early-stage design evaluation.

This repository includes all modules necessary for generating training data, building models, and running surrogate simulations.

## Repository Structure

ShapeShift-Framework/
├── SurrogateSimulationTool.py           # Abaqus script: keyed shaft solid surrogate model
├── FilamentLevelSimulationTool.py       # Abaqus script: filament-level simulation from G-code
├── shapeshift_train.py                  # ML model training: MLP + GNN (ShapeShift Brain)
├── shapeshift_hyperopt.py               # Hyperparameter tuning module
├── MEX-FEA-dataset.xlsx                 # MEX process input data (e.g., infill, pattern)
├── SOLID-FEA-dataset.xlsx               # Surrogate geometry and torsional stiffness data
└── README.md                            # Project documentation

## Requirements

### Abaqus Scripts
- Abaqus/CAE (2018 or later)
- Python (included with Abaqus)

### Python Scripts
- Python
- Dependencies: torch, torch-geometric, pandas, scikit-learn, matplotlib, numpy

Install dependencies with:

pip install torch torchvision torchaudio torch-geometric scikit-learn pandas matplotlib

## How to Use

1. **Generate Surrogate Model in Abaqus**  
   Run `SurrogateSimulationTool.py` in Abaqus scripting interface. Input shaft parameters and mesh size to generate a keyed shaft geometry for simulation.

2. **Generate Filament-Level Model (Optional)**  
   Run `FilamentLevelSimulationTool.py` to build a detailed filament-level model directly from a G-code file.

3. **Train the ShapeShift Model**  
   Run `shapeshift_train.py` to train the forward (MLP) and inverse (GNN) models on paired MEX and geometry datasets.

4. **Hyperparameter Tuning**  
   Run `shapeshift_hyperopt.py` to explore model configurations using grid search and rank them by validation loss.

## Sample Prediction

Input:
- Part = KSD10-KW4-KD1.8
- Infill = 66%
- Pattern = line
- Layer height = 0.3 mm

Output:
- Predicted diameter = 10.02 mm
- Key width = 3.89 mm
- Key depth = 1.76 mm

## License

This framework is intended for academic research and educational use only. Contact the author for commercial licensing or collaboration.

## Citation

@misc{shapeshift2025,  
  title = {ShapeShift: A Physics-Informed Transformation Framework for computational efficient Finite Element Analysis (FEA) in Material       Extrusion (MEX)},  
  author = {Chanawee Promaue},  
  year = {2025},  
  note = {https://github.com/your-username/ShapeShift-Framework}  
}

## Contact

th22301@bristol.ac.uk
