# Imports.
import os
import json
import subprocess
import argparse

username = "vincenttu"

os.environ["KAGGLE_USERNAME"] = username
os.environ["KAGGLE_KEY"] = "d8cb10c6ebfb6529afda64a2a04745f7"

import kaggle

# Custom imports.
# from utils import clean_model_folder

# Arguments.
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--title", default="test", type=str)
parser.add_argument("--save_p", default="../models/0297_baseline/", type=str)
args = parser.parse_args()
print(args)
print()

n_folds = 5

if __name__ == "__main__":
    folders = [i for i in os.listdir(args.save_p) if "fold" in i]
    print(folders)
    
    # assert len(args.title) >= 6 and len(args.title) <= 50
    # for i, f in enumerate(folders): assert f in [f"fold{i}" for i in range(n_folds)]
    
    # for f in folders: clean_model_folder(os.path.join(args.save_p, f), by="epoch")

    title_id = args.title.lower().replace("_", "-")
    data = {
      "title": f"{args.title}", 
      "id": f"{username}/{title_id}", 
      "licenses": [{"name": "CC0-1.0"}]
    }

    with open(os.path.join(args.save_p, 'dataset-metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    subprocess.run(["kaggle", "datasets", "create", "-p", f"{args.save_p}", "-u", "--dir-mode", "zip"])
    
    print("Kaggle Dataset created!")