# ADMET Predictor - Hybrid Approach (Local System Version)
# ========================================================

# This script provides ADMET prediction functionality using a hybrid approach
# of rule-based methods and cheminformatics

# 1. Dependencies and Imports
# --------------------------
# These libraries need to be installed in your environment
# You can install them with: pip install rdkit mordred scikit-learn matplotlib pandas seaborn

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw, PandasTools, AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from mordred import Calculator, descriptors
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# 2. Molecular Representation and Processing
# -----------------------------------------

class MoleculeProcessor:
    """Class for handling molecular input, validation, and standardization"""
    
    @staticmethod
    def smiles_to_mol(smiles):
        """Convert SMILES string to RDKit molecule"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        return mol
    
    @staticmethod
    def standardize_mol(mol):
        """Standardize molecule (remove salts, normalize structure)"""
        # Remove salts and keep largest fragment
        frags = Chem.GetMolFrags(mol, asMols=True)
        largest_mol = max(frags, default=mol, key=lambda x: x.GetNumAtoms())
        
        # Set aromaticity, sanitize, and assign stereochemistry
        Chem.SetAromaticity(largest_mol)
        return largest_mol
    
    @staticmethod
    def generate_3d_coordinates(mol, optimize=True):
        """Generate 3D coordinates for molecule"""
        mol_with_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
        
        if optimize:
            AllChem.MMFFOptimizeMolecule(mol_with_h)
        
        return mol_with_h
    
    @staticmethod
    def validate_molecule(mol):
        """Validate whether molecule is suitable for ADMET prediction"""
        if mol is None:
            return False, "Molecule is None"
        
        # Check if molecule has atoms
        if mol.GetNumAtoms() == 0:
            return False, "Molecule has no atoms"
        
        # Check for unusual elements (focus on organic drug-like compounds)
        allowed_elements = {6, 7, 8, 9, 15, 16, 17, 35, 53}  # C, N, O, F, P, S, Cl, Br, I
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() not in allowed_elements:
                return False, f"Contains uncommon element: {atom.GetSymbol()}"
        
        return True, "Molecule is valid"

# 3. Descriptor Calculation
# ------------------------

class DescriptorCalculator:
    """Calculate molecular descriptors and properties"""
    
    @staticmethod
    def calc_basic_descriptors(mol):
        """Calculate basic RDKit descriptors"""
        descriptors = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'RotBonds': Descriptors.NumRotatableBonds(mol),
            'AromaticRings': Chem.Lipinski.NumAromaticRings(mol),
            'HeavyAtoms': mol.GetNumHeavyAtoms(),
            'Complexity': Descriptors.BertzCT(mol),
            'QED': Descriptors.qed(mol)
        }
        return descriptors
    
    @staticmethod
    def calc_mordred_descriptors(mol, selected_only=True):
        """Calculate Mordred descriptors"""
        calc = Calculator(descriptors)
        
        # For faster calculation, only use a subset of useful descriptors if selected_only is True
        if selected_only:
            # Create a subset of descriptors (important for ADMET)
            selected_descriptors = [
                'nAcid', 'nBase', 'nBonds', 'TopoPSA', 'PEOE_VSA1',
                'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
                'SMR_VSA1', 'SMR_VSA2', 'SMR_VSA3', 'SlogP_VSA1',
                'SlogP_VSA2', 'SlogP_VSA3', 'EState_VSA1', 'EState_VSA2'
            ]
            
            # Calculate all and filter
            result = calc(mol)
            descriptors_dict = {str(d): result[d] for d in result.descriptors if str(d) in selected_descriptors}
        else:
            # Calculate all descriptors
            result = calc(mol)
            descriptors_dict = {str(d): result[d] for d in result.descriptors}
        
        # Convert any NaN to None
        descriptors_dict = {k: None if pd.isna(v) else v for k, v in descriptors_dict.items()}
        
        return descriptors_dict
    
    @staticmethod
    def get_fingerprints(mol, fp_type='morgan', radius=2, n_bits=2048):
        """Generate molecular fingerprints"""
        if fp_type == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        elif fp_type == 'maccs':
            fp = AllChem.GetMACCSKeysFingerprint(mol)
        elif fp_type == 'rdkit':
            fp = Chem.RDKFingerprint(mol)
        else:
            raise ValueError(f"Fingerprint type {fp_type} not supported")
        
        # Convert to binary list
        return np.array(list(fp))
    
    @staticmethod
    def combine_descriptors(basic_desc, mordred_desc=None):
        """Combine different descriptor sets"""
        combined = basic_desc.copy()
        if mordred_desc:
            combined.update(mordred_desc)
        return combined

# 4. Rule-based ADMET Predictions
# ------------------------------

class RuleBasedADMET:
    """Rule-based ADMET predictions"""
    
    @staticmethod
    def lipinski_ro5(mol):
        """Check Lipinski's Rule of Five"""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        violations = 0
        violation_details = []
        
        if mw > 500:
            violations += 1
            violation_details.append("MW > 500")
        if logp > 5:
            violations += 1
            violation_details.append("LogP > 5")
        if hbd > 5:
            violations += 1
            violation_details.append("HBD > 5")
        if hba > 10:
            violations += 1
            violation_details.append("HBA > 10")
        
        return {
            'ro5_pass': violations <= 1,
            'ro5_violations': violations,
            'ro5_details': '; '.join(violation_details) if violation_details else "No violations"
        }
    
    @staticmethod
    def veber_rules(mol):
        """Check Veber rules for oral bioavailability"""
        rotb = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        
        violations = 0
        violation_details = []
        
        if rotb > 10:
            violations += 1
            violation_details.append("RotB > 10")
        if tpsa > 140:
            violations += 1
            violation_details.append("TPSA > 140")
        
        return {
            'veber_pass': violations == 0,
            'veber_violations': violations,
            'veber_details': '; '.join(violation_details) if violation_details else "No violations"
        }
    
    @staticmethod
    def ghose_filter(mol):
        """Ghose filter for drug-likeness"""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        atoms = mol.GetNumHeavyAtoms()
        molar_refractivity = Descriptors.MolMR(mol)
        
        violations = 0
        violation_details = []
        
        if not (160 <= mw <= 480):
            violations += 1
            violation_details.append("MW outside [160, 480]")
        if not (-0.4 <= logp <= 5.6):
            violations += 1
            violation_details.append("LogP outside [-0.4, 5.6]")
        if not (20 <= atoms <= 70):
            violations += 1
            violation_details.append("Atoms outside [20, 70]")
        if not (40 <= molar_refractivity <= 130):
            violations += 1
            violation_details.append("MR outside [40, 130]")
        
        return {
            'ghose_pass': violations == 0,
            'ghose_violations': violations,
            'ghose_details': '; '.join(violation_details) if violation_details else "No violations"
        }
    
    @staticmethod
    def muegge_filter(mol):
        """Muegge filter for drug-likeness"""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hba = Descriptors.NumHAcceptors(mol)
        hbd = Descriptors.NumHDonors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotb = Descriptors.NumRotatableBonds(mol)
        # Use correct function to count rings
        rings = Chem.rdMolDescriptors.CalcNumRings(mol)
        
        violations = 0
        violation_details = []
        
        if not (200 <= mw <= 600):
            violations += 1
            violation_details.append("MW outside [200, 600]")
        if not (-2 <= logp <= 5):
            violations += 1
            violation_details.append("LogP outside [-2, 5]")
        if hba > 10:
            violations += 1
            violation_details.append("HBA > 10")
        if hbd > 5:
            violations += 1
            violation_details.append("HBD > 5")
        if tpsa > 150:
            violations += 1
            violation_details.append("TPSA > 150")
        if rotb > 15:
            violations += 1
            violation_details.append("RotB > 15")
        if rings > 7:
            violations += 1
            violation_details.append("Rings > 7")
        
        return {
            'muegge_pass': violations <= 1,
            'muegge_violations': violations,
            'muegge_details': '; '.join(violation_details) if violation_details else "No violations"
        }
    
    @staticmethod
    def predict_absorption(mol):
        """Predict absorption properties"""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        tpsa = Descriptors.TPSA(mol)
        
        # Estimate Caco-2 permeability (rule of thumb)
        # LogP > 0, MW < 400, HBD < 3, TPSA < 120
        caco2_features = (logp > 0, mw < 400, hbd < 3, tpsa < 120)
        caco2_score = sum(caco2_features) / len(caco2_features)
        
        if caco2_score > 0.75:
            caco2_class = "High"
        elif caco2_score > 0.5:
            caco2_class = "Medium"
        else:
            caco2_class = "Low"
        
        # Human Intestinal Absorption (HIA) rules
        # Rule of thumb: If compound follows Lipinski rules and TPSA < 140
        lipinski_check = RuleBasedADMET.lipinski_ro5(mol)
        hia_pass = lipinski_check['ro5_pass'] and tpsa < 140
        
        if hia_pass:
            hia_class = "High (>80%)"
        else:
            hia_class = "Low-Medium (<80%)"
        
        # Pgp substrate prediction (simplified rule-based approach)
        # LogP > 3, MW > 400 are often Pgp substrates
        pgp_substrate = logp > 3 and mw > 400
        
        return {
            'Caco2_Permeability_Class': caco2_class,
            'Caco2_Score': round(caco2_score, 2),
            'HIA_Class': hia_class,
            'Pgp_Substrate': "Likely" if pgp_substrate else "Unlikely"
        }
    
    @staticmethod
    def predict_distribution(mol):
        """Predict distribution properties"""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        # Blood-Brain Barrier (BBB) penetration
        # Rule of thumb: LogP - (N+O) > 0, MW < 450 for good BBB penetration
        # Count N and O atoms
        num_n_o = 0
        for atom in mol.GetAtoms():
            if atom.GetSymbol() in ['N', 'O']:
                num_n_o += 1
        
        bbb_score = logp - (0.1 * num_n_o)
        
        if bbb_score > 0 and mw < 450:
            bbb_class = "High"
        elif bbb_score > -1:
            bbb_class = "Medium"
        else:
            bbb_class = "Low"
        
        # Plasma Protein Binding (PPB) prediction
        # Rule of thumb: LogP > 3 often has high PPB
        if logp > 4:
            ppb_class = "Very High (>95%)"
        elif logp > 3:
            ppb_class = "High (>90%)"
        elif logp > 2:
            ppb_class = "Medium (>85%)"
        else:
            ppb_class = "Low (<85%)"
        
        # Volume of distribution (Vd) estimation
        # Basic rule of thumb based on logP and ionization
        has_basic_n = False
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.GetTotalDegree() < 4:
                has_basic_n = True
                break
        
        if has_basic_n and logp > 1:
            vd_class = "High"
        elif logp > 3:
            vd_class = "Medium-High"
        else:
            vd_class = "Low-Medium"
        
        return {
            'BBB_Penetration': bbb_class,
            'BBB_Score': round(bbb_score, 2),
            'Plasma_Protein_Binding': ppb_class,
            'Volume_Distribution': vd_class
        }
    
    @staticmethod
    def predict_metabolism(mol):
        """Predict metabolism properties"""
        # SMARTS patterns for common metabolic sites
        cyp_substrate_smarts = {
            'CYP1A2': ['[c;H]1[c]2[c]([c;H][c;H][c]1)[n][c;H][c;H][n]2', 'c1ccc2[n]ccc2c1'],  # Planar aromatic compounds
            'CYP2C9': ['[c]1[c][c]([c;H])[c]([C;H])[c]1', 'c1cc([O;H])ccc1'],  # Aromatic with acidic groups
            'CYP2C19': ['S', '[N;H]', '[N;H][C;H]=O'],  # Sulfur containing, amides
            'CYP2D6': ['[c][N;H][C;H]', '[c]1[c][n][c]([c;H])[c]1'],  # Basic nitrogen
            'CYP3A4': ['[C;H]=[C;H]', '[N;H][C;H]=O', 'C(=O)[O;H]']  # Diverse substrates
        }
        
        metabolism_predictions = {}
        
        # Check for each CYP substrate pattern
        for cyp, smarts_list in cyp_substrate_smarts.items():
            is_substrate = False
            for smarts in smarts_list:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    is_substrate = True
                    break
            metabolism_predictions[f'{cyp}_Substrate'] = "Likely" if is_substrate else "Unlikely"
        
        # Simple metabolic stability prediction based on structural features
        # Count groups susceptible to metabolism
        labile_groups = 0
        labile_patterns = [
            '[C;H][O;H]',  # Alcohols
            '[C][N;H][C;H]',  # Secondary amines
            '[C;H]=[C;H]',  # Alkenes
            'c1[c;H][c;H][c;H][c;H][c;H]1',  # Benzene
            '[C;H][S]',  # Sulfides
            '[C;H][Cl,Br,I,F]'  # Halogens
        ]
        
        for smarts in labile_patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                labile_groups += len(mol.GetSubstructMatches(pattern))
        
        if labile_groups > 5:
            metabolism_predictions['Metabolic_Stability'] = "Low"
        elif labile_groups > 2:
            metabolism_predictions['Metabolic_Stability'] = "Medium"
        else:
            metabolism_predictions['Metabolic_Stability'] = "High"
        
        return metabolism_predictions
    
    @staticmethod
    def predict_excretion(mol):
        """Predict excretion properties"""
        # Factors influencing excretion: molecular weight, LogP, charge state
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        # Check for ionizable groups (basic/acidic)
        basic_pattern = Chem.MolFromSmarts('[N;H,H2,H3,H4+]')
        
        # FIX: Separate SMARTS patterns for different acidic groups
        carboxylic_acid = Chem.MolFromSmarts('[O;H]C(=O)')
        phosphoric_acid = Chem.MolFromSmarts('[O;H]P(=O)O')
        sulfonic_acid = Chem.MolFromSmarts('[O;H]S(=O)(=O)')
        
        has_basic = basic_pattern and mol.HasSubstructMatch(basic_pattern)
        # Check each acidic pattern individually
        has_acidic = ((carboxylic_acid and mol.HasSubstructMatch(carboxylic_acid)) or
                     (phosphoric_acid and mol.HasSubstructMatch(phosphoric_acid)) or
                     (sulfonic_acid and mol.HasSubstructMatch(sulfonic_acid)))
        
        # Predict clearance pathway based on properties
        if mw > 400 and logp > 5:
            clearance_pathway = "Biliary/Fecal"
        elif has_acidic and not has_basic:
            clearance_pathway = "Renal (active secretion likely)"
        elif has_basic and mw < 400 and logp < 3:
            clearance_pathway = "Renal (filtration and active secretion)"
        elif logp < 0:
            clearance_pathway = "Renal (filtration)"
        else:
            clearance_pathway = "Mixed (renal/hepatic)"
        
        # Estimate half-life
        if mw > 500 and logp > 5:
            half_life = "Long (>24h)"
        elif (mw > 400 and 3 < logp < 5) or has_basic:
            half_life = "Moderate (12-24h)"
        else:
            half_life = "Short (<12h)"
        
        return {
            'Clearance_Pathway': clearance_pathway,
            'Half_Life': half_life
        }
    
    @staticmethod
    def predict_toxicity(mol):
        """Predict toxicity properties using structural alerts"""
        # SMARTS patterns for common toxicophores
        toxicophores = {
            'Alkylating_Agent': ['[C;H]=[C;H][C;H]=[O]', '[N,O,S][C;H]=[C;H][C;H]=[O]', '[C;H][Cl,Br,I]'],
            'Michael_Acceptor': ['[C;H]=[C;H][C](=O)', '[C;H]=[C;H][N](=O)'],
            'Epoxide': ['[C]1[O][C]1'],
            'Aziridine': ['[C]1[N][C]1'],
            'Aromatic_Amine': ['c[N;H2]'],
            'Aromatic_Nitro': ['c[N+](=O)[O-]'],
            'Quinone': ['[O]=[C]1[C]=[C][C](=[O])[C]=[C]1'],
            'Polyhalogenated': ['[C]([Cl,Br,I])([Cl,Br,I])[Cl,Br,I]'],
            'Acyl_Halide': ['[C](=O)[Cl,Br,I]']
        }
        
        alerts_found = []
        for alert_name, smarts_list in toxicophores.items():
            for smarts in smarts_list:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    alerts_found.append(alert_name)
                    break
        
        # AMES mutagenicity prediction (simplified rule-based)
        # Common mutagenic features
        ames_patterns = [
            'c1c([N+](=O)[O-])cccc1',  # Nitroaromatic
            'c1c([N;H2])cccc1',  # Aromatic amine
            '[C;H][N]=[N][C;H]',  # Azo compound
            'c1cc(Cl)c(Cl)cc1',  # Polychlorinated aromatic
            '[C;H]=[C;H][C](=O)[O;H]'  # Alpha,beta-unsaturated carbonyl
        ]
        
        ames_positive = False
        for smarts in ames_patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                ames_positive = True
                break
        
        # hERG inhibition prediction (simplified rule-based)
        # Basic nitrogen and high LogP are common features of hERG inhibitors
        logp = Descriptors.MolLogP(mol)
        basic_nitrogen = mol.HasSubstructMatch(Chem.MolFromSmarts('[N;!$(N(=O)=O)]'))
        
        herg_risk = "Low"
        if basic_nitrogen and logp > 3:
            herg_risk = "High"
        elif basic_nitrogen or logp > 4:
            herg_risk = "Medium"
        
        # Hepatotoxicity prediction (simplified rule-based)
        # Based on common hepatotoxic features
        hepatotox_patterns = [
            'c1ccccc1[C](=O)[C;H]',  # Acetophenone derivatives
            'c1c([Cl,Br,I])c([Cl,Br,I])ccc1',  # Polyhalogenated aromatics
            '[S](=O)(=O)[N;H]',  # Sulfonamides
            'c1c([N+](=O)[O-])cccc1'  # Nitroaromatics
        ]
        
        hepatotox_risk = "Low"
        for smarts in hepatotox_patterns:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                hepatotox_risk = "Medium-High"
                break
        
        # Estimate lethal dose (LD50) based on structural properties
        # A very rough estimate based on Lipinski parameters and toxic groups
        lipinski = RuleBasedADMET.lipinski_ro5(mol)
        
        if len(alerts_found) > 2 or ames_positive:
            ld50_estimate = "Potentially low LD50 (<50 mg/kg)"
        elif lipinski['ro5_violations'] > 1 or herg_risk == "High" or hepatotox_risk == "Medium-High":
            ld50_estimate = "Moderate LD50 (50-500 mg/kg)"
        else:
            ld50_estimate = "Likely high LD50 (>500 mg/kg)"
        
        return {
            'Toxicity_Alerts': '; '.join(alerts_found) if alerts_found else "No alerts found",
            'AMES_Mutagenicity': "Positive" if ames_positive else "Negative",
            'hERG_Inhibition_Risk': herg_risk,
            'Hepatotoxicity_Risk': hepatotox_risk,
            'LD50_Estimate': ld50_estimate
        }
    
    @staticmethod
    def get_all_predictions(mol):
        """Get all rule-based ADMET predictions for a molecule"""
        results = {}
        
        # Get all drug-likeness rules results
        results.update(RuleBasedADMET.lipinski_ro5(mol))
        results.update(RuleBasedADMET.veber_rules(mol))
        results.update(RuleBasedADMET.ghose_filter(mol))
        results.update(RuleBasedADMET.muegge_filter(mol))
        
        # Get ADMET predictions
        results.update(RuleBasedADMET.predict_absorption(mol))
        results.update(RuleBasedADMET.predict_distribution(mol))
        results.update(RuleBasedADMET.predict_metabolism(mol))
        results.update(RuleBasedADMET.predict_excretion(mol))
        results.update(RuleBasedADMET.predict_toxicity(mol))
        
        return results

# 5. Advanced ADMET Predictions (Optional: Place for integration with pre-trained models)
# ------------------------------------------------------------------------------------

class AdvancedADMET:
    """Class for advanced ADMET predictions using pre-trained models"""
    
    def __init__(self, models_dir='models'):
        """Initialize with pre-trained models if available"""
        self.models = {}
        self.models_dir = models_dir
        
        # Check if models directory exists
        if os.path.exists(models_dir):
            self._load_models()
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        try:
            for model_file in os.listdir(self.models_dir):
                if model_file.endswith('.pkl'):
                    model_name = os.path.splitext(model_file)[0]
                    model_path = os.path.join(self.models_dir, model_file)
                    self.models[model_name] = pickle.load(open(model_path, 'rb'))
                    print(f"Loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def _get_features_for_model(self, mol, model_name):
        """Extract appropriate features for a specific model"""
        # Example implementation - would depend on how models were trained
        desc_calc = DescriptorCalculator()
        basic_desc = desc_calc.calc_basic_descriptors(mol)
        
        if model_name in ['LogS', 'BBB']:
            # Use fingerprints for these models
            fp = desc_calc.get_fingerprints(mol, 'morgan', 2, 1024)
            return fp
        else:
            # Use descriptors for other models
            mordred_desc = desc_calc.calc_mordred_descriptors(mol, selected_only=True)
            combined = desc_calc.combine_descriptors(basic_desc, mordred_desc)
            
            # Convert to numpy array in correct order
            # This would depend on how your model was trained
            feature_names = sorted(combined.keys())
            return np.array([combined[f] for f in feature_names])
    
    def predict(self, mol, property_name):
        """Make prediction for a specific ADMET property"""
        if property_name not in self.models:
            raise ValueError(f"No model available for {property_name}")
        
        # Get appropriate features
        features = self._get_features_for_model(mol, property_name)
        
        # Make prediction
        model = self.models[property_name]
        prediction = model.predict([features])[0]
        
        return prediction
    
    def predict_all(self, mol):
        """Make predictions for all available models"""
        results = {}
        for model_name in self.models:
            results[model_name] = self.predict(mol, model_name)
        return results

# 6. ADMET Prediction Manager
# -------------------------

class ADMETPredictor:
    """Main class for ADMET prediction"""
    
    def __init__(self, use_advanced=False, models_dir='models'):
        """Initialize the ADMET predictor"""
        self.mol_processor = MoleculeProcessor()
        self.desc_calculator = DescriptorCalculator()
        self.rule_based = RuleBasedADMET()
        self.use_advanced = use_advanced
        
        # Only initialize advanced models if requested
        if use_advanced:
            self.advanced = AdvancedADMET(models_dir)
    
    def predict_from_smiles(self, smiles):
        """Predict ADMET properties from SMILES string"""
        try:
            # Process molecule
            mol = self.mol_processor.smiles_to_mol(smiles)
            standardized_mol = self.mol_processor.standardize_mol(mol)
            
            # Validate molecule
            valid, message = self.mol_processor.validate_molecule(standardized_mol)
            if not valid:
                return {'error': message}
            
            # Calculate descriptors
            descriptors = self.desc_calculator.calc_basic_descriptors(standardized_mol)
            
            # Get rule-based predictions
            predictions = self.rule_based.get_all_predictions(standardized_mol)
            
            # Add advanced predictions if available
            if self.use_advanced and hasattr(self, 'advanced') and self.advanced.models:
                advanced_predictions = self.advanced.predict_all(standardized_mol)
                predictions.update(advanced_predictions)
            
            # Combine results
            results = {
                'smiles': smiles,
                'descriptors': descriptors,
                'predictions': predictions,
                'mol': standardized_mol  # Include molecule for visualization
            }
            
            return results
        
        except Exception as e:
            return {'error': str(e)}
    
    def predict_batch(self, smiles_list):
        """Predict ADMET properties for a list of SMILES strings"""
        results = []
        for smiles in smiles_list:
            result = self.predict_from_smiles(smiles)
            results.append(result)
        return results
    
    def predict_from_file(self, file_path, smiles_col=0, delimiter=',', has_header=True):
        """Predict ADMET properties for molecules in a file"""
        try:
            # Read file
            df = pd.read_csv(file_path, delimiter=delimiter)
            
            # Get SMILES column
            if isinstance(smiles_col, int):
                smiles_col = df.columns[smiles_col]
            
            # Extract SMILES
            smiles_list = df[smiles_col].tolist()
            
            # Predict
            results = self.predict_batch(smiles_list)
            
            return results
        
        except Exception as e:
            return {'error': str(e)}

# 7. Visualization and Reporting
# ----------------------------

class ResultVisualizer:
    """Visualization and reporting of ADMET predictions"""
    
    @staticmethod
    def visualize_molecule(mol, width=400, height=200, filename=None):
        """Display molecule as image file"""
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()
        
        if filename:
            with open(filename, 'wb') as f:
                f.write(png_data)
            return filename
        else:
            # Return the PNG data
            return png_data
    
    @staticmethod
    def create_radar_chart(predictions, filename=None, figsize=(8, 8)):
        """Create radar chart of key ADMET properties"""
        # Extract key properties and normalize them to 0-1 scale
        props = {
            'Absorption': 1 if predictions['Caco2_Permeability_Class'] == 'High' else (
                0.5 if predictions['Caco2_Permeability_Class'] == 'Medium' else 0.1),
            'Distribution': 1 if predictions['BBB_Penetration'] == 'High' else (
                0.5 if predictions['BBB_Penetration'] == 'Medium' else 0.1),
            'Metabolism': 1 if predictions['Metabolic_Stability'] == 'High' else (
                0.5 if predictions['Metabolic_Stability'] == 'Medium' else 0.1),
            'Drug-likeness': 1 if (predictions['ro5_pass'] and predictions['veber_pass']) else (
                0.5 if predictions['ro5_pass'] or predictions['veber_pass'] else 0.1)
        }
        
        # Check toxicity - invert scale (lower is better)
        toxic_alerts = len(predictions['Toxicity_Alerts'].split('; ')) if predictions['Toxicity_Alerts'] != 'No alerts found' else 0
        props['Safety'] = 1 if toxic_alerts == 0 else (0.5 if toxic_alerts <= 2 else 0.1)
        
        # Create radar chart
        categories = list(props.keys())
        values = [props[cat] for cat in categories]
        
        # Add the first value to close the circular plot
        values += values[:1]
        categories += categories[:1]
        
        # Create angles for each category
        angles = [n / float(len(categories)-1) * 2 * np.pi for n in range(len(categories))]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        
        # Draw chart
        ax.plot(angles, values, linewidth=2, linestyle='solid')
        ax.fill(angles, values, alpha=0.4)
        
        # Set category labels
        plt.xticks(angles[:-1], categories[:-1])
        
        # Add title
        ax.set_title("ADMET Profile", size=15, pad=20)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return filename
        
        return fig
    
    @staticmethod
    def format_results_as_html(results):
        """Format results as HTML for display"""
        if 'error' in results:
            return f"<h3>Error</h3><p>{results['error']}</p>"
        
        html = "<h2>ADMET Prediction Results</h2>"
        
        # Add molecule info
        html += f"<h3>Molecule: {results['smiles']}</h3>"
        
        # Format descriptors
        html += "<h3>Physicochemical Properties</h3>"
        html += "<table border='1'><tr><th>Property</th><th>Value</th></tr>"
        for prop, value in results['descriptors'].items():
            html += f"<tr><td>{prop}</td><td>{value:.2f if isinstance(value, float) else value}</td></tr>"
        html += "</table>"
        
        # Drug-likeness rules
        html += "<h3>Drug-likeness</h3>"
        html += "<table border='1'><tr><th>Rule</th><th>Status</th><th>Details</th></tr>"
        
        rule_results = [
            ('Lipinski Rule of 5', results['predictions']['ro5_pass'], results['predictions']['ro5_details']),
            ('Veber Rules', results['predictions']['veber_pass'], results['predictions']['veber_details']),
            ('Ghose Filter', results['predictions']['ghose_pass'], results['predictions']['ghose_details']),
            ('Muegge Filter', results['predictions']['muegge_pass'], results['predictions']['muegge_details'])
        ]
        
        for rule, status, details in rule_results:
            status_text = "✅ Pass" if status else "❌ Fail"
            html += f"<tr><td>{rule}</td><td>{status_text}</td><td>{details}</td></tr>"
        html += "</table>"
        
        # ADMET predictions
        admet_sections = [
            ('Absorption', ['Caco2_Permeability_Class', 'HIA_Class', 'Pgp_Substrate']),
            ('Distribution', ['BBB_Penetration', 'Plasma_Protein_Binding', 'Volume_Distribution']),
            ('Metabolism', ['CYP1A2_Substrate', 'CYP2C9_Substrate', 'CYP2C19_Substrate', 
                           'CYP2D6_Substrate', 'CYP3A4_Substrate', 'Metabolic_Stability']),
            ('Excretion', ['Clearance_Pathway', 'Half_Life']),
            ('Toxicity', ['Toxicity_Alerts', 'AMES_Mutagenicity', 'hERG_Inhibition_Risk', 
                         'Hepatotoxicity_Risk', 'LD50_Estimate'])
        ]
        
        for section, properties in admet_sections:
            html += f"<h3>{section}</h3>"
            html += "<table border='1'><tr><th>Property</th><th>Prediction</th></tr>"
            for prop in properties:
                if prop in results['predictions']:
                    html += f"<tr><td>{prop.replace('_', ' ')}</td><td>{results['predictions'][prop]}</td></tr>"
            html += "</table>"
        
        return html
    
    @staticmethod
    def export_to_csv(results, filename="admet_results.csv"):
        """Export results to CSV file"""
        # Flatten the nested structure
        flat_results = {}
        
        if isinstance(results, list):
            # Batch results
            rows = []
            for res in results:
                if 'error' in res:
                    row = {'smiles': res.get('smiles', 'Unknown'), 'error': res['error']}
                else:
                    row = {'smiles': res['smiles']}
                    # Add descriptors
                    for k, v in res['descriptors'].items():
                        row[f"desc_{k}"] = v
                    # Add predictions
                    for k, v in res['predictions'].items():
                        row[f"pred_{k}"] = v
                rows.append(row)
            
            # Create dataframe and save
            df = pd.DataFrame(rows)
            df.to_csv(filename, index=False)
            return f"Results saved to {filename}"
        else:
            # Single result
            if 'error' in results:
                return f"Error: {results['error']}"
            
            row = {'smiles': results['smiles']}
            # Add descriptors
            for k, v in results['descriptors'].items():
                row[f"desc_{k}"] = v
            # Add predictions
            for k, v in results['predictions'].items():
                row[f"pred_{k}"] = v
            
            # Create dataframe and save
            df = pd.DataFrame([row])
            df.to_csv(filename, index=False)
            return f"Results saved to {filename}"
    
    @staticmethod
    def generate_pdf_report(results, filename="admet_report.pdf"):
        """Generate PDF report for results - placeholder - requires additional libraries"""
        # This is a placeholder for PDF generation
        # You would typically use a library like reportlab or WeasyPrint
        return "PDF generation requires additional libraries (e.g., reportlab)"

# 8. Command Line Interface
# -----------------------

def main():
    """Command-line interface for ADMET prediction"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ADMET Predictor')
    parser.add_argument('--smiles', '-s', type=str, help='SMILES string of molecule')
    parser.add_argument('--file', '-f', type=str, help='File containing SMILES strings (CSV/TSV)')
    parser.add_argument('--output', '-o', type=str, default='admet_results.csv', help='Output file name')
    parser.add_argument('--visualize', '-v', action='store_true', help='Generate visualizations')
    parser.add_argument('--advanced', '-a', action='store_true', help='Use advanced models (if available)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ADMETPredictor(use_advanced=args.advanced)
    visualizer = ResultVisualizer()
    
    if args.smiles:
        # Single molecule prediction
        print(f"Processing molecule: {args.smiles}")
        result = predictor.predict_from_smiles(args.smiles)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        # Print key results to console
        print("\nKey Properties:")
        for key, value in result['descriptors'].items():
            if key in ['MW', 'LogP', 'HBA', 'HBD', 'TPSA']:
                print(f"  {key}: {value:.2f}")
        
        print("\nDrug-likeness:")
        print(f"  Lipinski Rule of 5: {'Pass' if result['predictions']['ro5_pass'] else 'Fail'}")
        print(f"  Veber Rules: {'Pass' if result['predictions']['veber_pass'] else 'Fail'}")
        
        print("\nADMET Highlights:")
        print(f"  Caco-2 Permeability: {result['predictions']['Caco2_Permeability_Class']}")
        print(f"  BBB Penetration: {result['predictions']['BBB_Penetration']}")
        print(f"  CYP3A4 Substrate: {result['predictions']['CYP3A4_Substrate']}")
        print(f"  Toxicity Alerts: {result['predictions']['Toxicity_Alerts']}")
        
        # Export to CSV
        visualizer.export_to_csv(result, args.output)
        print(f"\nDetailed results saved to {args.output}")
        
        # Generate visualizations if requested
        if args.visualize:
            # Ensure output directory exists
            output_dir = os.path.dirname(args.output) or '.'
            base_name = os.path.splitext(os.path.basename(args.output))[0]
            
            # Save molecule image
            mol_img_path = os.path.join(output_dir, f"{base_name}_molecule.png")
            visualizer.visualize_molecule(result['mol'], filename=mol_img_path)
            print(f"Molecule image saved to {mol_img_path}")
            
            # Save radar chart
            radar_path = os.path.join(output_dir, f"{base_name}_radar.png")
            visualizer.create_radar_chart(result['predictions'], filename=radar_path)
            print(f"ADMET radar chart saved to {radar_path}")
    
    elif args.file:
        # Batch processing
        print(f"Processing molecules from file: {args.file}")
        results = predictor.predict_from_file(args.file)
        
        if isinstance(results, dict) and 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        # Print summary
        successful = sum(1 for r in results if 'error' not in r)
        print(f"Successfully processed {successful} out of {len(results)} molecules")
        
        # Export to CSV
        visualizer.export_to_csv(results, args.output)
        print(f"Results saved to {args.output}")
    
    else:
        # Interactive mode - example
        print("No input provided. Running example...")
        examples = [
            ('Aspirin', 'CC(=O)OC1=CC=CC=C1C(=O)O'),
            ('Caffeine', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'),
            ('Ibuprofen', 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O')
        ]
        
        for name, smiles in examples:
            print(f"\nProcessing {name} ({smiles})...")
            result = predictor.predict_from_smiles(smiles)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
                continue
            
            # Print key results
            print("Key Properties:")
            for key, value in result['descriptors'].items():
                if key in ['MW', 'LogP', 'HBA', 'HBD', 'TPSA']:
                    print(f"  {key}: {value:.2f}")
            
            print("\nDrug-likeness:")
            print(f"  Lipinski Rule of 5: {'Pass' if result['predictions']['ro5_pass'] else 'Fail'}")
            print(f"  Veber Rules: {'Pass' if result['predictions']['veber_pass'] else 'Fail'}")
            
            print("\nADMET Highlights:")
            print(f"  Caco-2 Permeability: {result['predictions']['Caco2_Permeability_Class']}")
            print(f"  BBB Penetration: {result['predictions']['BBB_Penetration']}")
            print(f"  CYP3A4 Substrate: {result['predictions']['CYP3A4_Substrate']}")
            print(f"  Toxicity Alerts: {result['predictions']['Toxicity_Alerts']}")
            
            print("\n" + "-"*50)

# 9. Example Usage
# --------------

# Example code
def run_example():
    """Run an example prediction"""
    predictor = ADMETPredictor()
    
    # Example molecules
    examples = {
        'Aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'Ibuprofen': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'
    }
    
    for name, smiles in examples.items():
        print(f"\nProcessing {name} ({smiles})...")
        results = predictor.predict_from_smiles(smiles)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            continue
        
        # Display key properties
        print("Key Properties:")
        for key, value in results['descriptors'].items():
            if key in ['MW', 'LogP', 'HBA', 'HBD', 'TPSA']:
                print(f"  {key}: {value:.2f}")
        
        # Display drug-likeness rules
        print("\nDrug-likeness:")
        print(f"  Lipinski Rule of 5: {'Pass' if results['predictions']['ro5_pass'] else 'Fail'}")
        print(f"  Veber Rules: {'Pass' if results['predictions']['veber_pass'] else 'Fail'}")
        
        # Display ADMET highlights
        print("\nADMET Highlights:")
        print(f"  Caco-2 Permeability: {results['predictions']['Caco2_Permeability_Class']}")
        print(f"  BBB Penetration: {results['predictions']['BBB_Penetration']}")
        print(f"  CYP3A4 Substrate: {results['predictions']['CYP3A4_Substrate']}")
        print(f"  Toxicity Alerts: {results['predictions']['Toxicity_Alerts']}")
        
        print("\n" + "-"*50)

# Run example if script is executed directly
if __name__ == "__main__":
    main()