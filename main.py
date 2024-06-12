import json
from dataset import ExtendetDataset
from combine_datasets import (
    combine_datasets,
    TEST_SET_MODE_CUT,
    TEST_SET_MODE_ORIGINAL,
    UNSUFFICIENT_DATA_MODE_REDUCE_TOTAL,
    UNSUFFICIENT_DATA_MODE_REDUCE_PROPORTION,
    UNSUFFICIENT_DATA_MODE_RESAMPLE,
)
import pickle
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pathlib
import threading

lock = threading.Lock()

# import DatasetCard from python hub
from huggingface_hub import DatasetCard, HfApi


def load_dataset(dataset):
    print(f"loading dataset {dataset['name']}...")
    return ExtendetDataset(**dataset, lock=lock)


def load_datasets_from_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    datasets = []
    with ThreadPoolExecutor() as executor:
        future_to_dataset = {
            executor.submit(load_dataset, dataset): dataset for dataset in data
        }
        for future in as_completed(future_to_dataset):
            datasets.append(future.result())
    return datasets


total_size = 50000
filename = "datasets.pkl"


datasets = load_datasets_from_json("datasets.json")
print(f"loaded {len(datasets)} datasets from json file.")

# store the datasets for later use
with open(filename, "wb") as file:
    pickle.dump(datasets, file)
print(f"stored datasets in datasets.pkl")


###################################################


# load the datasets from the pickle file
with open(filename, "rb") as file:
    datasets = pickle.load(file)


proportions = {
    "count chars in english words": 0.5,
    "count chars in scrambled words": 0.5,
    "count chars in paragraph": 0.5,
    "count words in paragraph": 0.5,
    # "reverse words": 0.5,
    # "SuperGLUE cb": 0.5,
    # "SuperGLUE copa": 0.5,
    # "SuperGLUE multirc": 0.5,
    # "SuperGLUE record": 0.5,
    # "SuperGLUE rte": 0.5,
    # "SuperGLUE wic": 0.5,
    # "SuperGLUE boolq": 0.5,
    "MMLU-Pro": 2,
    "DMath": 2,
    "MathQA": 2,
    "ape210k": 2,
    "LSAT-AR": 2,
    "crosswords": 2,
    "big bench hard": 10,
    "gpqa main": 0,
    "AquaRat": 0,
}

for dataset in datasets:
    if dataset.name not in proportions:
        proportions[dataset.name] = 1

amounts = [proportions[dataset.name] for dataset in datasets]

print(f"loaded {len(datasets)} datasets from pickle file.")
dataset = combine_datasets(
    datasets,
    total_size,
    amounts=amounts,
    in_col="final_input",
    out_col="final_target",
    test_size=0.1,
    test_set_mode=TEST_SET_MODE_CUT,
    unsufficient_data_mode=UNSUFFICIENT_DATA_MODE_REDUCE_PROPORTION,
    visualize=True,
    print_statistics=True,
)

# store the combined dataset for later use
with open("combined_dataset.pkl", "wb") as file:
    pickle.dump(dataset, file)


####################################################


# load the combined dataset from the pickle file
with open("combined_dataset.pkl", "rb") as file:
    dataset = pickle.load(file)


hf_name = "jeggers/CoT-Collection"

# push to hub
dataset.push_to_hub(hf_name)

dataset_card_str = """
---
## Dataset Card: CoT-Collection
Collection of various different tasks.
The tasks aim at needing reasoning and diverse Chain-of-Thought strategies.
Some tasks are chosen to be easy and don't require much reasoning.

*Note* that Big Bench Hard itself contains 27 different tasks.

*Also Note* that the test set has a very different distribution of tasks than the training set.

This dataset contains GPQA. Please do not publish it in plain text on the web.
"""

dataset_card_str += (
    f"### Statistics for the training set ({len(dataset['train'])} samples)\n\n"
)

api = HfApi()
repo = api.create_repo(hf_name, repo_type="dataset", exist_ok=True)


# check for train_dataset.png and add to card
if os.path.exists("train_dataset.png"):
    dataset_card_str += "![train_dataset](train_dataset.png)\n"
    # upload image to hub
    api.upload_file(
        path_or_fileobj="train_dataset.png",
        path_in_repo="train_dataset.png",
        repo_id=hf_name,
        repo_type="dataset",
    )

# check if train_statistics.md exists and add to card
if os.path.exists("train_statistics.md"):
    with open("train_statistics.md", "r") as file:
        dataset_card_str += file.read()


dataset_card_str += (
    f"### Statistics for the test set ({len(dataset['test'])} samples)\n\n"
)

# check for test_dataset.png and add to card
if os.path.exists("test_dataset.png"):
    dataset_card_str += "![test_dataset](test_dataset.png)\n"
    # upload image to hub
    api.upload_file(
        path_or_fileobj="test_dataset.png",
        path_in_repo="test_dataset.png",
        repo_id=hf_name,
        repo_type="dataset",
    )


# check if test_statistics.md exists and add to card
if os.path.exists("test_statistics.md"):
    with open("test_statistics.md", "r") as file:
        dataset_card_str += file.read()

# create a dataset card
card = DatasetCard(dataset_card_str)
card.push_to_hub(hf_name)
