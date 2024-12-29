This repository contains the experimental data and source code for the paper:
**"Guided Learning for Photovoltaic Power Regression in the Absence of Key Information."**

## Contents
- 'dataset/': Datasets used for the experiments.
- 'PhotoVoltaic/': Source code for data loader, model training, and evaluation.
- 'ReadMe,md': Description of implementation

## Getting Started
1. Clone the repository:
   ```bash
   git clone https://github.com/tkddnr7671/GL_PVR.git

2. Install dependencies:
   ```bash
   conda install ./PhotoVoltaic/conda_create_environmental.sh

3. Run the main script:
   Training PV power regression model with solar irradiation
   ```bash
   python main_w_sr.py --mode train --save_dir SAVE_PATH --modelType MODELTYPE --loc_ID 105 --nLayers 3 --seqLeng SEQUENCE_LENGTH
   ```

   Training PV power regression model without solar irradiation
   ```bash
   python main.py --mode train --save_dir SAVE_PATH --modelType MODELTYPE --loc_ID 105 --nLayers 3 --seqLeng SEQUENCE_LENGTH
   ```

   Guided Training PV power regression model without solar irradiation [proposed] 
   ```bash
   python main_wo_sr.py --mode train --save_dir SAVE_PATH --modelType MODELTYPE --loc_ID 105 --nLayers 3 --seqLeng SEQUENCE_LENGTH
   ```

## Data Repository 
1. 'dataset/AWOS/' : Weather recordings for the entire year, from Jan. 1, 2022 to Dec. 31, 2022
2. 'dataset/AWS/'  : Weather recordings for the entire year, from Jan. 1, 2022 to Dec. 31, 2022
3. 'dataset/SR'    : Recordings of Solar irradiance for the entire year, from Jan. 1, 2022 to Dec. 31, 2022
4. 'dataset/photovoltaic/':  Recordings of PV power generation for the entire year, from Jan. 1, 2022 to Dec. 31, 2022

## Citation
To be updated

   
   
