from processing.signatures import TransformerEncoder
import os
import glob
import pandas as pd
import pickle
import shutil


transformers = TransformerEncoder("bert-base-multilingual-uncased", max_seq_length=10, pooling_size=4)
transformers.eval()

home = './data/zip/dataset'
dataset_folders = os.listdir(home)

for dataset in dataset_folders:
    directory = f"{home}/{dataset}"
    sub_dir = [x[0] for x in os.walk(directory)]
    containt_csv_files = glob.glob(f"{directory}/*.csv")
    if len(sub_dir) == 1 and len(containt_csv_files) == 1: # only containt 1 csv file
        try:
            df = pd.read_csv(containt_csv_files[0])
            signature_dict = transformers.encode_list(df.columns.to_list())
            with open(f"{directory}/signature.pkl", "wb") as fout:
                pickle.dump(signature_dict, fout)
        except:
            print(f"Error while convert {directory}")
            shutil.move(directory, f"./data/zip/fails/{dataset}")
