# ADMET Predictor

A streamlined tool for predicting Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET) properties of small molecules based on their SMILES representation.

## Features

- Predict key ADMET properties from SMILES strings
- Assess drug-likeness using multiple rule sets (Lipinski, Veber, etc.)
- Analyze single molecules or batch process multiple compounds
- Generate downloadable reports for further analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/admet-predictor.git
cd admet-predictor

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# You can install them with
pip install rdkit mordred scikit-learn matplotlib pandas seaborn

If you face any issues installing these dependancies with pip then try to use conda for the same.

# scikit-learn
conda install -c conda-forge scikit-learn

# Also try to use anaconda prompt as it comes with many dependencies already pre-installed.
