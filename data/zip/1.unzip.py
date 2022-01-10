import zipfile
import glob
import tqdm
import shutil
import os


for file in tqdm.tqdm(glob.glob("./raw/*.zip")):
    directory = "./dataset/"+file.split("/")[-1].split(".")[0]
    try:
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(directory)
        sub_dir = [x[0] for x in os.walk(directory)]
        containt_csv_files = glob.glob(f"{directory}/*.csv")
        if len(sub_dir) != 1 or len(containt_csv_files) != 1:
            shutil.rmtree(directory)
            draft_dir = "./draft/"+file.split("/")[-1].split(".")[0]
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(draft_dir)
    except:
        # broken zip files
        print(file)
