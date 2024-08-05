from typing import List
import random
import pandas as pd

import numpy as np
import os
import json
from pathlib import Path

def split(codes: List[str], categories: List[str], threshold: int = 10, category_split: List[float] = (0.5, 0.5), code_split: List[float] = (0.8, 0.2), seed: int = 42):
    random.seed(seed)
    
    data = pd.DataFrame({'code': codes, 'category': categories})
    counts = data.groupby('category').size()

    small_categories = counts[counts <  threshold].index.tolist()
    big_categories   = counts[counts >= threshold].index.tolist()

    random.shuffle(small_categories)

    category_split_ratio = np.array(category_split)
    category_split_ratio = category_split_ratio / category_split_ratio.sum()
    category_split_ratio = category_split_ratio.cumsum()
    splits = []

    for i in range(category_split_ratio.shape[0]):
        start = 0 if i == 0 else int(category_split_ratio[i-1] * len(small_categories))
        end = int(category_split_ratio[i] * len(small_categories))
        splits.append(data[data['category'].isin(small_categories[start: end])])

    code_split_ratio = np.array(code_split)
    code_split_ratio = code_split_ratio / code_split_ratio.sum()
    code_split_ratio = code_split_ratio.cumsum()

    for cat in big_categories:
        curr = data[data['category'] == cat]
        curr = curr.sample(frac=1.0, random_state=seed)

        for i in range(code_split_ratio.shape[0]):
            start = 0 if i == 0 else round(code_split_ratio[i-1] * curr.shape[0])
            end = round(code_split_ratio[i] * curr.shape[0])
            splits[i] = pd.concat([splits[i], curr.iloc[start: end]], axis=0, ignore_index=True)
    return splits


grasping_dir = Path("data/oakink_shadow_dataset_valid_force_noise_accept_1")

frame = []
for code in os.listdir(grasping_dir):
    for pose in os.listdir(grasping_dir / code):
        filepath = grasping_dir / code / pose

        if filepath.suffix != ".json":
            continue

        with open(filepath, "r") as f:
            data = json.load(f)
            frame.append(
                {
                    "category": data["category"],
                    "code": code,
                    "pose": pose.replace(".json", ""),
                    "intent": data["intent"],
                    "filepath": filepath,
                }
            )
frame = pd.DataFrame(frame)

data = frame[['code', 'category']].drop_duplicates()
counts = data.groupby('category').size()
train, test = split(data['code'].tolist(), data['category'].tolist(), threshold=6, category_split=(0.5, 0.5), code_split=(0.7, 0.3))

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

print(train)
print(test)