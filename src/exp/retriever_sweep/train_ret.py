import os
import pandas as pd
import argparse
import numpy as np
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from datasets import Dataset
from torch.utils.data import DataLoader

import wandb
wandb.login()

# Arguments.
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train_p", default="../../../input/train.csv", type=str)
parser.add_argument("--project_run_root", default=".", type=str)
parser.add_argument("--save_root", default="../../../models/retriever_sweep/", type=str)
parser.add_argument("--project", default="lecr_retriever_sweep", type=str)
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--epochs", default=5, type=int)
parser.add_argument("--backbone_type", default="sentence-transformers/all-MiniLM-L6-v2", type=str)
parser.add_argument("--scheduler", default="warmuplinear", type=str)
parser.add_argument("--evaluation_steps", default=1000, type=int)
parser.add_argument("--checkpoint_save_steps", default=1000, type=int)
args = parser.parse_args()
print(args)
print()

if __name__ == "__main__":
    
    if args.project_run_root == ".":
        args.project_run_root = args.backbone_type.split("/")[-1]
    
    save_p_root = os.path.join(args.save_root, args.project_run_root)
    os.makedirs(save_p_root, exist_ok=True)
    save_p = os.path.join(save_p_root, f"ep{args.epochs}")
    
    DATA_PATH = "../../../input/"
    topics = pd.read_csv(DATA_PATH + "topics.csv")
    content = pd.read_csv(DATA_PATH + "content.csv")
    correlations = pd.read_csv(DATA_PATH + "correlations.csv")
    
    topics.rename(columns=lambda x: "topic_" + x, inplace=True)
    content.rename(columns=lambda x: "content_" + x, inplace=True)
    
    correlations["content_id"] = correlations["content_ids"].str.split(" ")
    corr = correlations.explode("content_id").drop(columns=["content_ids"])
    
    corr = corr.merge(topics, how="left", on="topic_id")
    corr = corr.merge(content, how="left", on="content_id")
    
    corr["set"] = corr[["topic_title", "content_title"]].values.tolist()
    train_df = pd.DataFrame(corr["set"])
    dataset = Dataset.from_pandas(train_df)
    
    train_examples = []
    train_data = dataset["set"]
    n_examples = dataset.num_rows

    for i in range(n_examples):
        example = train_data[i]
        if example[0] == None: #remove None
            print(example)
            continue        
        train_examples.append(InputExample(texts=[str(example[0]), str(example[1])]))
        
    model = SentenceTransformer(args.backbone_type)
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    
    # Might want to test a few loss functions?
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    warmup_steps = int(len(train_dataloader) * args.epochs * 0.1) #10% of train data

    run = wandb.init(project=args.project, config=vars(args), name=f"{args.project_run_root}", dir="/tmp")
    
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=args.epochs,
              warmup_steps=warmup_steps,
              scheduler=args.scheduler,
              # evaluator=,  # What's the best evaluator?
              evaluation_steps=args.evaluation_steps,
              use_amp=True,
              output_path=save_p,
              checkpoint_path=save_p,
              checkpoint_save_steps=args.checkpoint_save_steps)
    
    model.save(save_p)
    artifact = wandb.Artifact(args.backbone_type.replace('/', '-'), type='model')
    artifact.add_dir(save_p, name=f"ep{args.epochs}")
    run.log_artifact(artifact)
    
    run.finish()