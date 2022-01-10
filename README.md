# synthetic_tabular_data_research
Syntheic Tabular Data using Variational Auto Encoder

Status
- [x] Implement automated data cleaning
- [x] Implement data transform for tabular
- [x] Implement baseline VAE
- [x] Signature representation
- [x] Test training pipeline
- [ ] Verify the training and generating data results

Dataset
Download samples data from this [link](https://drive.google.com/drive/folders/1C_-Pn4uxs1PF42i0Ve9FfN9p6nGZA1oy?usp=sharing)


## Run
Note: The result has not been verified yet. Below is to ensure the pipeline can run successfully

0. `cd` to the root path of project

1. Activate the virtual environment
`source env2/bin/activate`

2. Go to file `train.py` to change the config or run with the defaults by command
`python train.py`
    After the training completes, model weights and meta data will be stored in folder `results`

3. To generate synthetic data after training, go to file `generate_data.py` to change the config or run with the defaults by command
`python generate_data.py`


