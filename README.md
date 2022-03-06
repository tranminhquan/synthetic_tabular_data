# synthetic_tabular_data_research
Syntheic Tabular Data using Variational Auto Encoder

Status
- [x] Implement automated data cleaning
- [x] Implement frequency encoder
  - [x] Add normalized option  
- [x] Implement data transform for tabular
- [x] Implement baseline VAE
- [x] Signature representation
  - [x] Column name embedding (extracted from BERT)
  - [ ] Quantiles
- [x] Test training pipeline
- [ ] Verify the training and generating data results

Dataset (Suong)
Download samples data from this [link](https://drive.google.com/drive/folders/1C_-Pn4uxs1PF42i0Ve9FfN9p6nGZA1oy?usp=sharing)


## Run
Note: The result has not been verified yet. Below is to ensure the pipeline can run successfully. Open a terminal and:

### Create virtual environment and install packages

0. `cd` to the root path of project

1. Create the virtual environment named *env*
`python3 -m venv env`

2. Activate the virtual environment
`source env/bin/activate`

3. Upgrade to the latest pip
`pip install --upgrade pip`

4. Install packages from *requirements.txt*
`pip install -r requirements.txt`

### Config settings
Go the the `configs/local` if you wish to change the default configuration setting

### Training and Generating

1. Go to file `train.py` and run 
`python train.py`

1. To generate synthetic data after training, go to file `generate_data.py` to change the config or run with the defaults by command
`python generate_data.py`


