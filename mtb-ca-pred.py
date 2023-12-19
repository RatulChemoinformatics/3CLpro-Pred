# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 11:10:36 2023

@author: RATUL BHOWMIK
"""


import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
from streamlit_option_menu import option_menu

# The App
st.title('💊 Mtb-CA-pred app')
st.info('Mtb-CA-pred allows users to predict bioactivity of a query molecule against the Mycobacterium tuberculosis carbonic anhydrase target protein.')



# loading the saved models
bioactivity_first_model = pickle.load(open('pubchem.pkl', 'rb'))
bioactivity_second_model = pickle.load(open('substructure.pkl', 'rb'))
bioactivity_third_model = pickle.load(open('descriptors.pkl', 'rb'))

# Define the tabs
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs(['Main', 'About', 'What is Mycobacterium tuberculosis carbonic anhydrase (Mtb-CA)?', 'Dataset', 'Model performance', 'Python libraries', 'Citing us', 'Application Developers'])

with tab1:
    st.title('Application Description')
    st.success(
        " This module of [**Mtb-CA-pred**](https://github.com/RatulChemoinformatics/Mtb-CA-pred) has been built to predict bioactivity and identify potent inhibitors against Mycobacterium tuberculosis carbonic anhydrase using robust machine learning algorithms."
    )

# Define a sidebar for navigation
with st.sidebar:
    selected = st.selectbox(
        'Choose a prediction model',
        [
            'Mtb-CA prediction model using pubchemfingerprints',
            'Mtb-CA prediction model using substructurefingerprints',
            'Mtb-CA prediction model using 1D and 2D molecular descriptors',
        ],
    )

# Mtb-CA prediction model using pubchemfingerprints
if selected == 'Mtb-CA prediction model using pubchemfingerprints':
    # page title
    st.title('Predict bioactivity of molecules against Mtb-CA using pubchemfingerprints')

    # Molecular descriptor calculator
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_first_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pKi')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
        st.sidebar.markdown("""
        [Example input file](https://github.com/RatulChemoinformatics/QSAR/blob/main/predict.txt)
        """)

    if st.sidebar.button('Predict'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('pubchem.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.')

# Mtb-CA prediction model using substructurefingerprints
elif selected == 'Mtb-CA prediction model using substructurefingerprints':
    # page title
    st.title('Predict bioactivity of molecules against Mtb-CA using substructurefingerprints')

    # Molecular descriptor calculator
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/SubstructureFingerprinter.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_second_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pKi')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
        st.sidebar.markdown("""
        [Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
        """)

    if st.sidebar.button('Predict'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('substructure.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.')
            
            
# Mtb-CA prediction model using 1D and 2D molecular descriptors
if selected == 'Mtb-CA prediction model using 1D and 2D molecular descriptors':
    # page title
    st.title('Predict bioactivity of molecules against Mtb-CA using 1D and 2D molecular descriptors')

    # Molecular descriptor calculator
    def desc_calc():
        # Performs the descriptor calculation
        bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -2d -descriptortypes ./PaDEL-Descriptor/descriptors.xml -dir ./ -file descriptors_output.csv"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        os.remove('molecule.smi')

    # File download
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
        href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
        return href

    # Model building
    def build_model(input_data):
        # Apply model to make predictions
        prediction = bioactivity_third_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pKi')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)

    # Sidebar
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
        st.sidebar.markdown("""
        [Example input file](https://raw.githubusercontent.com/dataprofessor/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
        """)

    if st.sidebar.button('Predict'):
        if uploaded_file is not None:
            load_data = pd.read_table(uploaded_file, sep=' ', header=None)
            load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

            st.header('**Original input data**')
            st.write(load_data)

            with st.spinner("Calculating descriptors..."):
                desc_calc()

            # Read in calculated descriptors and display the dataframe
            st.header('**Calculated molecular descriptors**')
            desc = pd.read_csv('descriptors_output.csv')
            st.write(desc)
            st.write(desc.shape)

            # Read descriptor list used in previously built model
            st.header('**Subset of descriptors from previously built models**')
            Xlist = list(pd.read_csv('descriptors.csv').columns)
            desc_subset = desc[Xlist]
            st.write(desc_subset)
            st.write(desc_subset.shape)

            # Apply trained model to make prediction on query compounds
            build_model(desc_subset)
        else:
            st.warning('Please upload an input file.')
            
            
with tab2:
  coverimage = Image.open('Logo.png')
  st.image(coverimage)
with tab3:
  st.header('What is Mtb-CA?')
  st.write('The genome of Mycobacterium tuberculosis (Mtb) encodes three β-carbonic anhydrases (CAs, EC 4.2.1.1) that are crucial for the life cycle of the bacterium. Studies conducted on mycobacterial β-CAs have shown that they are involved in the invasion and survival of pathogens in the host environment.')
with tab4:
  st.header('Dataset')
  st.write('''
    In our work, we retrieved a Mycobacterium tubaerculosis carbonic anhydrase (Mtb-CA) dataset from the ChEMBL database. The data was curated and resulted in a non-redundant set of 267 Mtb-CA inhibitors, which can be divided into:
    - 147 active compounds
    - 46 inactive compounds
    - 74 intermediate compounds
    ''')
with tab5:
  st.header('Model performance')
  st.write('We selected a total of 3 different molecular signatures namely pubchem fingerprints, substructure fingerprints, and 1D 2D molecular descriptors to build the web application. The correlation coefficient, RMSE, and MAE values for the pubchem fingerprint model was found to be 0.931, 0.3554, and 0.2847. The correlation coefficient, RMSE, and MAE values for the substructure fingerprint model was found to be 0.9227, 0.373, and 0.382. The correlation coefficient, RMSE, and MAE values for the 1D and 2D molecular descriptor model was found to be 0.9447, 0.3278, and 0.2588')
with tab6:
  st.header('Python libraries')
  st.markdown('''
    This app is based on the following Python libraries:
    - `streamlit`
    - `pandas`
    - `rdkit`
    - `padelpy`
  ''')
with tab7:
  st.markdown('Aspatwar A, Kairys V, Rala S, Parikka M, Bozdag M, Carta F, Supuran CT, Parkkila S. Mycobacterium tuberculosis β-carbonic anhydrases: Novel targets for developing antituberculosis drugs. International Journal of Molecular Sciences. DOI: https://doi.org/10.3390/ijms20205153.')
with tab8:
  st.markdown('Ratul Bhowmik, Ajay Manaithiya, Sunil Kumar, Ranajit Nath, Dr. Ashok Aspatwar, Dr. Bijo Mathew. [***Infectious Disease Lab, Faculty of Medicine and Health Technology, Tampere University, Finland***] [***CADD LAB, Department of Pharmaceutical Chemistry, Amrita School of Pharmacy, Amrita Vishwa Vidyapeetham, Kochi, India***] ')