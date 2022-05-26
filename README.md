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

## 1. Single tabular running
Note: The result has not been verified yet. Below is to ensure the pipeline can run successfully. Open a terminal and:

### Config settings
Go the the `configs/local` if you wish to change the default configuration setting

### Training and Generating

1. Go to file `train.py` and run 
`python train.py`

2. To generate synthetic data after training, go to file `generate_data.py` to change the config or run with the defaults by command
`python generate_data.py`


## 2. Transfer Learning (multi-training)

### Config settings:
Go the the `configs/multi_training.yaml`
In the configs file
* `source_domain_dir`: folder contain sub-folders, each sub-folder must have a csv file and a metadata json file (optional).
* `target_domain_dir`: if target domain is not indicated, the KFold is appled on the `source_domain_dir` to get the target directories for each fold

### Run multi-training
Run `python3 multitrain.py`

The sdgym dataset can be downloaded here
https://drive.google.com/drive/folders/1z6WmY057FfAqFf8DRma0ici7p_jCogJ9?usp=sharing
