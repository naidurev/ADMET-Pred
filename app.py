import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from admet_predictor import ADMETPredictor, ResultVisualizer

# Page configuration
st.set_page_config(
    page_title="ADMET Predictor",
    page_icon="💊",
    layout="wide"
)

# Custom CSS - only basic styling that works well with Streamlit
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0;
        color: #2563eb;
    }
    .subtitle {
        color: #64748b;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .success {
        color: #10b981;
        font-weight: bold;
    }
    .failure {
        color: #ef4444;
        font-weight: bold;
    }
    .neutral {
        color: #3b82f6;
        font-weight: bold;
    }
    footer {
        text-align: center;
        color: #94a3b8;
        padding: 1rem 0;
        margin-top: 2rem;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-title">ADMET Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict Absorption, Distribution, Metabolism, Excretion, and Toxicity properties</p>', 
            unsafe_allow_html=True)

# Initialize predictor
predictor = ADMETPredictor()
visualizer = ResultVisualizer()

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select option:", ["Single Molecule", "Batch Processing", "About"])

# Single Molecule Page
if page == "Single Molecule":
    st.header("Single Molecule Analysis")
    
    # SMILES input and example molecules 
    col1, col2 = st.columns([3, 1])
    
    with col1:
        smiles = st.text_input("Enter SMILES string:", placeholder="Example: CC(=O)OC1=CC=CC=C1C(=O)O (Aspirin)")
    
    with col2:
        # Example molecules dropdown
        examples = {
            "Select example": "",
            "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        }
        example = st.selectbox("Or choose example:", list(examples.keys()))
        if example != "Select example":
            smiles = examples[example]
    
    # Predict button
    predict_btn = st.button("Predict ADMET Properties", type="primary", use_container_width=True)
    
    if predict_btn and smiles:
        with st.spinner("Analyzing molecule..."):
            result = predictor.predict_from_smiles(smiles)
            
            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                # Results layout
                st.markdown("---")
                
                # Basic info and structure
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Molecule image
                    st.subheader("Molecular Structure")
                    mol_img = visualizer.visualize_molecule(result['mol'])
                    st.image(mol_img, use_column_width=True)
                    
                    # Basic properties table
                    st.subheader("Physicochemical Properties")
                    
                    props_df = pd.DataFrame(
                        {"Value": [f"{v:.2f}" if isinstance(v, float) else v 
                                 for k, v in result['descriptors'].items() 
                                 if k in ['MW', 'LogP', 'HBA', 'HBD', 'TPSA', 'QED']]},
                        index=['MW', 'LogP', 'HBA', 'HBD', 'TPSA', 'QED']
                    )
                    st.dataframe(props_df)
                
                with col2:
                    # ADMET Profile
                    st.subheader("ADMET Profile")
                    
                    # Radar chart
                    fig = visualizer.create_radar_chart(result['predictions'])
                    st.pyplot(fig)
                    
                    # Drug-likeness rules summary
                    st.subheader("Drug-likeness Rules")
                    
                    # Create a dataframe for the rules display
                    rules_data = pd.DataFrame({
                        "Rule": ["Lipinski Rule of 5", "Veber Rules", "Ghose Filter", "Muegge Filter"],
                        "Status": [
                            "PASS" if result['predictions']['ro5_pass'] else "FAIL",
                            "PASS" if result['predictions']['veber_pass'] else "FAIL",
                            "PASS" if result['predictions']['ghose_pass'] else "FAIL",
                            "PASS" if result['predictions']['muegge_pass'] else "FAIL"
                        ],
                        "Details": [
                            result['predictions']['ro5_details'],
                            result['predictions']['veber_details'],
                            result['predictions']['ghose_details'],
                            result['predictions']['muegge_details']
                        ]
                    })
                    
                    # Create a custom dataframe display with better visibility for the status column
                    # This approach uses a combination of background color and high-contrast text
                    def color_status(val):
                        if val == 'PASS':
                            # Light green background with dark text
                            return 'background-color: #d1fae5; color: #064e3b; font-weight: bold'
                        else:
                            # Light red background with dark text
                            return 'background-color: #fee2e2; color: #7f1d1d; font-weight: bold'
                    
                    # Apply the styling with better text contrast
                    st.dataframe(rules_data.style.applymap(color_status, subset=['Status']))
                
                # ADMET Properties in tabs
                st.subheader("Detailed ADMET Properties")
                tabs = st.tabs(["Absorption", "Distribution", "Metabolism", "Excretion", "Toxicity"])
                
                # Absorption tab
                with tabs[0]:
                    st.write("**Absorption Properties**")
                    
                    absorption_data = pd.DataFrame({
                        "Property": [
                            "Caco-2 Permeability", 
                            "Caco-2 Score", 
                            "Human Intestinal Absorption",
                            "P-glycoprotein Substrate"
                        ],
                        "Value": [
                            result['predictions']['Caco2_Permeability_Class'],
                            result['predictions']['Caco2_Score'],
                            result['predictions']['HIA_Class'],
                            result['predictions']['Pgp_Substrate']
                        ]
                    })
                    
                    st.table(absorption_data)
                
                # Distribution tab
                with tabs[1]:
                    st.write("**Distribution Properties**")
                    
                    distribution_data = pd.DataFrame({
                        "Property": [
                            "Blood-Brain Barrier Penetration", 
                            "BBB Score", 
                            "Plasma Protein Binding",
                            "Volume of Distribution"
                        ],
                        "Value": [
                            result['predictions']['BBB_Penetration'],
                            result['predictions']['BBB_Score'],
                            result['predictions']['Plasma_Protein_Binding'],
                            result['predictions']['Volume_Distribution']
                        ]
                    })
                    
                    st.table(distribution_data)
                
                # Metabolism tab
                with tabs[2]:
                    st.write("**Metabolism Properties**")
                    
                    # Get all CYP substrates plus metabolic stability
                    metabolism_props = []
                    metabolism_values = []
                    
                    for key in result['predictions']:
                        if key.endswith('_Substrate') or key == 'Metabolic_Stability':
                            metabolism_props.append(key.replace('_', ' '))
                            metabolism_values.append(result['predictions'][key])
                    
                    metabolism_data = pd.DataFrame({
                        "Property": metabolism_props,
                        "Value": metabolism_values
                    })
                    
                    st.table(metabolism_data)
                
                # Excretion tab
                with tabs[3]:
                    st.write("**Excretion Properties**")
                    
                    excretion_data = pd.DataFrame({
                        "Property": ["Clearance Pathway", "Half Life"],
                        "Value": [
                            result['predictions']['Clearance_Pathway'],
                            result['predictions']['Half_Life']
                        ]
                    })
                    
                    st.table(excretion_data)
                
                # Toxicity tab
                with tabs[4]:
                    st.write("**Toxicity Properties**")
                    
                    toxicity_data = pd.DataFrame({
                        "Property": [
                            "Toxicity Alerts",
                            "AMES Mutagenicity",
                            "hERG Inhibition Risk",
                            "Hepatotoxicity Risk",
                            "LD50 Estimate"
                        ],
                        "Value": [
                            result['predictions']['Toxicity_Alerts'],
                            result['predictions']['AMES_Mutagenicity'],
                            result['predictions']['hERG_Inhibition_Risk'],
                            result['predictions']['Hepatotoxicity_Risk'],
                            result['predictions']['LD50_Estimate']
                        ]
                    })
                    
                    st.table(toxicity_data)
                
                # Download options
                st.download_button(
                    label="Download Results as CSV",
                    data=pd.DataFrame([{**{'SMILES': smiles}, 
                                      **{f"desc_{k}": v for k, v in result['descriptors'].items()},
                                      **{f"pred_{k}": v for k, v in result['predictions'].items()}}]
                                    ).to_csv(index=False),
                    file_name="admet_result.csv",
                    mime="text/csv"
                )

# Batch Processing Page
elif page == "Batch Processing":
    st.header("Batch ADMET Prediction")
    
    st.info("Upload a file containing SMILES strings. The file should have a column named 'SMILES' or similar.")
    
    uploaded_file = st.file_uploader("Upload file (CSV, TXT, SMI)", type=['csv', 'txt', 'smi'])
    
    if uploaded_file is not None:
        # Process the file
        try:
            # Detect file type and load accordingly
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:  # txt or smi file
                content = uploaded_file.getvalue().decode('utf-8')
                smiles_list = [line.strip().split()[0] for line in content.splitlines() if line.strip()]
                df = pd.DataFrame({'SMILES': smiles_list})
            
            # Find SMILES column
            smiles_col = None
            for col in ['SMILES', 'smiles', 'SMILE', 'smile', 'Structure']:
                if col in df.columns:
                    smiles_col = col
                    break
            
            if smiles_col is None:
                if len(df.columns) > 0:
                    smiles_col = st.selectbox("Select column containing SMILES:", df.columns)
                else:
                    st.error("Could not identify any columns in the file.")
            
            if smiles_col:
                smiles_list = df[smiles_col].tolist()
                
                if st.button("Run Batch Prediction", type="primary"):
                    with st.spinner(f"Processing {len(smiles_list)} molecules..."):
                        results = predictor.predict_batch(smiles_list)
                        
                        # Process results
                        result_rows = []
                        for i, res in enumerate(results):
                            row = {'SMILES': smiles_list[i]}
                            
                            if 'error' in res:
                                row['Status'] = 'Error'
                                row['Error'] = res['error']
                            else:
                                row['Status'] = 'Success'
                                # Add key properties and predictions
                                row['MW'] = res['descriptors'].get('MW', 'N/A')
                                row['LogP'] = res['descriptors'].get('LogP', 'N/A')
                                row['TPSA'] = res['descriptors'].get('TPSA', 'N/A')
                                row['Lipinski'] = res['predictions'].get('ro5_pass', 'N/A')
                                row['Veber'] = res['predictions'].get('veber_pass', 'N/A')
                                row['Caco2'] = res['predictions'].get('Caco2_Permeability_Class', 'N/A')
                                row['BBB'] = res['predictions'].get('BBB_Penetration', 'N/A')
                                row['CYP3A4'] = res['predictions'].get('CYP3A4_Substrate', 'N/A')
                                row['Toxicity'] = res['predictions'].get('Toxicity_Alerts', 'N/A')
                            
                            result_rows.append(row)
                        
                        # Show results
                        results_df = pd.DataFrame(result_rows)
                        st.dataframe(results_df)
                        
                        # Prepare full detailed results for download
                        detailed_rows = []
                        for i, res in enumerate(results):
                            if 'error' in res:
                                continue
                                
                            row = {'SMILES': smiles_list[i]}
                            # Add descriptors
                            for k, v in res['descriptors'].items():
                                row[f"desc_{k}"] = v
                            # Add predictions
                            for k, v in res['predictions'].items():
                                row[f"pred_{k}"] = v
                                
                            detailed_rows.append(row)
                            
                        detailed_df = pd.DataFrame(detailed_rows)
                        
                        # Download options
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                label="Download Results Summary",
                                data=results_df.to_csv(index=False),
                                file_name="admet_summary.csv",
                                mime="text/csv"
                            )
                        with col2:
                            st.download_button(
                                label="Download Full Results",
                                data=detailed_df.to_csv(index=False),
                                file_name="admet_detailed.csv",
                                mime="text/csv"
                            )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# About Page
else:
    st.header("About ADMET Predictor")
    
    st.markdown("""
    ### Overview

    This tool uses cheminformatics methods to predict ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) 
    properties of small molecules from their SMILES representations. It is designed to support drug discovery efforts by 
    providing rapid assessment of drug-like properties and potential issues.

    ### Features

    - **Drug-likeness Assessment**:
      - Lipinski's Rule of Five
      - Veber Rules
      - Ghose Filter
      - Muegge Filter

    - **ADMET Predictions**:
      - Absorption: Caco-2 permeability, intestinal absorption, Pgp substrate
      - Distribution: Blood-brain barrier penetration, plasma protein binding
      - Metabolism: CYP450 substrate predictions, metabolic stability
      - Excretion: Clearance pathways, half-life estimation
      - Toxicity: Structural alerts, AMES mutagenicity, hERG inhibition risk

    ### Implementation

    The tool employs a hybrid approach:
    - Rule-based calculations using established medicinal chemistry guidelines
    - Structural pattern recognition for metabolism and toxicity prediction
    - Molecular descriptor-based predictions
    
    The primary dependencies are:
    - RDKit for molecular handling and cheminformatics
    - Mordred for molecular descriptor calculation
    - Streamlit for the web interface
    
    ### Limitations

    - Predictions are based on general rules and patterns, not experimental data
    - Results should be considered as initial guidance rather than definitive assessments
    - More advanced predictions require machine learning models trained on specific datasets
    """)

# Footer
st.markdown("""
<footer>
    ADMET Predictor Tool v1.0 | Developed for drug discovery acceleration
</footer>
""", unsafe_allow_html=True)