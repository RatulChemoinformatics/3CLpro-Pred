# -*- coding: utf-8 -*-
"""

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
st.title('💊 3CLpro-Pred app')
st.info('3CLpro-Pred allows users to predict bioactivity of a query molecule against the SARS CoV-2 3CL Protease target protein.')



# loading the saved models
bioactivity_first_model = pickle.load(open('substructure.pkl', 'rb'))
bioactivity_second_model = pickle.load(open('descriptors.pkl', 'rb'))

# Define the tabs
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs(['Main', 'About', 'What is SARS CoV-2 3CL Protease?', 'Dataset', 'Model performance', 'Python libraries', 'Citing us', 'Application Developers'])

with tab1:
    st.title('Application Description')
    st.success(
        " This module of [**3CLpro-pred**](https://github.com/RatulChemoinformatics/3CLpro-Pred) has been built to predict bioactivity and identify potent inhibitors against SARS CoV-2 3CL protease using robust machine learning algorithms."
    )

# Define a sidebar for navigation
with st.sidebar:
    selected = st.selectbox(
        'Choose a prediction model',
        [
            
            '3CLpro prediction model using substructurefingerprints',
            '3CLpro prediction model using 1D and 2D molecular descriptors',
        ],
    )


# 3Clpro prediction model using substructurefingerprints
if selected == '3CLpro prediction model using substructurefingerprints':
    # page title
    st.title('Predict bioactivity of molecules against 3CLpro using substructurefingerprints')

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
        prediction = bioactivity_first_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pIC50')
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
            
            
# 3CLpro prediction model using 1D and 2D molecular descriptors
if selected == '3CLpro prediction model using 1D and 2D molecular descriptors':
    # page title
    st.title('Predict bioactivity of molecules against 3CLpro using 1D and 2D molecular descriptors')

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
        prediction = bioactivity_second_model.predict(input_data)
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pIC50')
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
  st.header('What is SARS CoV-2 3CL protease?')
  st.write('The 3CLpro type is regarded to be the most important of the four types, since it is the one most closely associated with viral replication. It was identified in a research that the major protease 3CLpro of COVID19 has a 96 percent structural similarities to the SARS-CoV protease. The 3CLpro enzyme is the primary enzyme essential for the process of proteolysis. It degrades the viral polyprotein and separates it into functional components that may be used separately. Because of the critical role that 3CLpro plays in the virus life cycle, it is an excellent target for the development of efficient antiviral medicines against a variety of Coronaviruses .')
with tab4:
  st.header('Dataset')
  st.write('''
    In our work, we retrieved a SARS CoV-2 3CL protease (3CLpro) dataset from the ChEMBL database. The data was curated and resulted in a non-redundant set of 919 3CLpro inhibitors, which can be divided into:
    - 473 active compounds
    - 189 inactive compounds
    - 257 intermediate compounds
    ''')
with tab5:
  st.header('Model performance')
  st.write('We selected a total of 2 different molecular signatures namely substructure fingerprints, and 1D 2D molecular descriptors to build the web application. The correlation coefficient, RMSE, and MAE values for the substructure fingerprint model was found to be 0.9233, 0.4453, and 0.3348. The correlation coefficient, RMSE, and MAE values for the 1D and 2D molecular descriptor model was found to be 0.9736, 0.3016, and 0.2178')
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
  st.markdown('Bhowmik R, Manaithiya A, Vyas B, Nath R, Rehman S, Roy S, Roy R. Identification of potential inhibitor against Ebola virus VP35: insight into virtual screening, pharmacoinformatics profiling, and molecular dynamic studies. Structural Chemistry. DOI: https://doi.org/10.1007/s11224-022-01899-y.')
with tab8:
  st.markdown('Ratul Bhowmik, Ajay Manaithiya. [***Department of Pharmaceutical Chemistry, SPER, Jamia Hamdard, New Delhi, India***] ')
