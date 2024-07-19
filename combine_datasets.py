import math
import random
from datasets import concatenate_datasets, DatasetDict, Value, ClassLabel
from dataset import USE_ONLY_TEST, USE_ONLY_TRAIN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
from tqdm import tqdm

UNSUFFICIENT_DATA_MODE_RESAMPLE = "resample"
UNSUFFICIENT_DATA_MODE_REDUCE_TOTAL = "reduce_total"
UNSUFFICIENT_DATA_MODE_REDUCE_PROPORTION = "reduce_proportion"
UNSUFFICIENT_DATA_MODE_ERROR = "error"

TEST_SET_MODE_CUT = "cut"
TEST_SET_MODE_ORIGINAL = "original"


def combine_datasets(
    datasets,
    total,
    amounts=None,
    unsufficient_data_mode=UNSUFFICIENT_DATA_MODE_REDUCE_TOTAL,
    test_set_mode=TEST_SET_MODE_ORIGINAL,
    tokenizer=None,
    max_tokens=-1,
    test_size_in_dist=0.025,
    test_size_finetune=0.2,
    in_col="input",
    prompt_template="{}",
    out_col="target",
    shuffle_individual_datasets=False,
    shuffle_final_dataset=True,
    seed=None,
    visualize=False,
    print_statistics=True,
    map_kwargs={
        "batched": True,
        "batch_size": -1,
    },
    additional_keep_columns=[],
):
    if not seed:
        seed = random.randint(0, 1000000)

    # rename columns to uniform names
    datasets = [
        dataset.rename_columns({dataset.in_col: in_col, dataset.out_col: out_col})
        for dataset in datasets
    ]

    # apply prompt template
    print("applying prompt template")

    def batch_apply_prompt_template(batch):
        return {
            in_col: [prompt_template.format(x) for x in batch[in_col]],
        }

    datasets = [
        dataset.map(batch_apply_prompt_template, **map_kwargs) for dataset in datasets
    ]

    # Convert 'final_target' column to string type in all datasets
    new_datasets = []
    for dataset in datasets:

        # if type is ClassLabel, convert to string
        if isinstance(dataset["train"].features[out_col], ClassLabel):
            print(
                f"Converting column {out_col} from ClassLabel to string type in dataset {dataset.name}"
            )
            int2str = dataset["train"].features[out_col].int2str

            # cast to int first
            for split in dataset.keys():
                dataset[split] = dataset[split].cast_column(out_col, Value("int32"))

            def map_function(batch):
                batch[out_col] = [int2str(x) for x in batch[out_col]]
                return batch

            dataset = dataset.map(map_function, **map_kwargs)

        # if is float, format to string rounding to at most 4 decimal places if necessary
        if dataset["train"].features[out_col].dtype.startswith("float"):
            print(
                f"Converting column {out_col} from float to string type in dataset {dataset.name}"
            )

            def map_function(batch):
                batch[out_col] = [
                    f"{x:.4f}".rstrip("0").rstrip(".") if "." in f"{x:.4f}" else f"{x}"
                    for x in batch[out_col]
                ]
                return batch

            dataset = dataset.map(map_function, **map_kwargs)

        if dataset["train"].features[out_col].dtype != "string":
            print(
                f"Converting column {out_col} to string type in dataset {dataset.name}"
            )
            dataset = dataset.cast_column(out_col, Value("string"))
        new_datasets.append(dataset)
    datasets = new_datasets

    # filter all datasets for token length input and output
    if tokenizer and max_tokens > 0:
        print("filtering datasets for token length")

        def batch_filter_function(batch):
            tokens = tokenizer(
                [
                    input + output
                    for input, output in zip(batch[in_col], batch[out_col])
                ],
            )
            return [len(tokenized) <= max_tokens for tokenized in tokens["input_ids"]]

        datasets = [
            dataset.filter(batch_filter_function, **map_kwargs) for dataset in datasets
        ]

    if amounts is None:
        amounts = [1] * len(datasets)
    absolute_amounts = [int(amount / sum(amounts) * total) for amount in amounts]

    # remove datasets with 0 absolute amount
    new_datasets = []
    new_absolute_amounts = []
    for dataset, absolute_amount in zip(datasets, absolute_amounts):
        if absolute_amount > 0:
            new_datasets.append(dataset)
            new_absolute_amounts.append(absolute_amount)
    datasets = new_datasets
    absolute_amounts = new_absolute_amounts

    # perform split on datasets, if not already done
    print("splitting datasets")
    # out_of_dist_test_set_amounts = []
    data_and_amounts = list(enumerate(zip(datasets, absolute_amounts)))
    for i, (dataset, absolute_amount) in tqdm(
        data_and_amounts, total=len(data_and_amounts)
    ):
        # relative_amount = absolute_amount / total
        # target_test_size = int(
        #     test_size * min(relative_amount * total, len(dataset["train"]))
        # )

        if dataset.use_only == USE_ONLY_TRAIN:
            # merge every split into train
            datasets[i].dataset = DatasetDict(
                {
                    "train": concatenate_datasets(
                        [dataset[key].dataset for key in dataset.keys()]
                    ),
                    "test": dataset["train"].dataset.select(range(0)),
                }
            )
            # out_of_dist_test_set_amounts.append(0)
        elif dataset.use_only == USE_ONLY_TEST:
            # merge every split into test
            datasets[i].dataset = DatasetDict(
                {
                    "train": dataset["train"].dataset.select(range(0)),
                    "test": concatenate_datasets(
                        [dataset[key].dataset for key in dataset.keys()]
                    ),
                }
            )
            # out_of_dist_test_set_amounts.append(len(dataset["test"]))
        else:
            raise ValueError(
                "Dataset must have use_only set to USE_ONLY_TRAIN or USE_ONLY_TEST"
            )
        # elif "test" not in dataset:
        #     # if target test size is 0, sample 0 samples for test set
        #     if target_test_size == 0:
        #         datasets[i]["test"] = dataset["train"].select(range(0)).dataset
        #     else:
        #         datasets[i] = dataset["train"].train_test_split(
        #             test_size=target_test_size, seed=seed
        #         )
        # else:
        #     if (
        #         test_set_mode == TEST_SET_MODE_CUT
        #         and len(dataset["test"]) > target_test_size
        #     ):
        #         datasets[i]["test"] = (
        #             dataset["test"].select(range(target_test_size)).dataset
        #         )
        #         # add remaining samples to train set
        #         datasets[i]["train"] = concatenate_datasets(
        #             [
        #                 dataset["train"].dataset,
        #                 dataset["test"].select(range(target_test_size, -1)).dataset,
        #             ]
        #         )

    available_amounts = [len(dataset["train"]) for dataset in datasets]

    if any(
        amount > available
        for amount, available in zip(absolute_amounts, available_amounts)
    ):
        if unsufficient_data_mode == UNSUFFICIENT_DATA_MODE_ERROR:
            raise ValueError("Not enough data available for the requested amounts")

        # keep proportion of datasets but reduce total amount of final dataset
        if unsufficient_data_mode == UNSUFFICIENT_DATA_MODE_REDUCE_TOTAL:
            total, absolute_amounts = reduce_total(
                available_amounts, absolute_amounts, total
            )
            names_of_unsufficient_datasets = [
                dataset.name
                for dataset, amount, available in zip(
                    datasets, absolute_amounts, available_amounts
                )
                if amount > available
            ]
            print(
                f"Reducing total amount of final dataset to {total} due to unsufficient data in the datasets {names_of_unsufficient_datasets}"
            )

        if unsufficient_data_mode == UNSUFFICIENT_DATA_MODE_REDUCE_PROPORTION:
            for i in itertools.cycle(range(len(absolute_amounts))):
                if sum(absolute_amounts) >= total:
                    break
                absolute_amounts[i] += 1
            absolute_amounts = find_distribution(available_amounts, absolute_amounts)
            print("Proportions of datasets reduced due to insufficient data")

    if shuffle_individual_datasets:
        datasets = [
            dataset.shuffle(seed=seed).flatten_indices() for dataset in datasets
        ]

    # get all columns that should be kept
    all_keep_columns = set()
    for dataset in datasets:
        all_keep_columns.update(dataset.keep_columns)
    all_keep_columns = list(all_keep_columns)
    all_keep_columns += additional_keep_columns

    # add keep columns to datasets if not already present
    updated_datasets = []
    for dataset in datasets:
        updated_dataset = dataset
        for column in all_keep_columns:
            if column not in dataset["train"].column_names:
                updated_dataset["train"] = updated_dataset["train"].add_column(
                    column, [None] * len(dataset["train"])
                )
            if column not in dataset["test"].column_names:
                updated_dataset["test"] = updated_dataset["test"].add_column(
                    column, [None] * len(dataset["test"])
                )
        updated_datasets.append(updated_dataset)
    datasets = updated_datasets

    # combine datasets
    updated_datasets = []
    for dataset, amount in zip(datasets, absolute_amounts):
        if amount <= len(dataset["train"]):
            dataset["train"] = dataset["train"].select(range(amount))
        else:
            assert unsufficient_data_mode == UNSUFFICIENT_DATA_MODE_RESAMPLE
            indices = random.choices(range(len(dataset["train"])), k=amount)
            dataset["train"] = dataset["train"].select(indices)

        dataset = dataset.select_columns([in_col, out_col] + all_keep_columns)

        # dataset["train"] = dataset["train"].add_column(
        #     "evaluation_method", [dataset.evaluation_method] * len(dataset["train"])
        # )
        dataset["train"] = dataset["train"].add_column(
            "dataset", [dataset.name] * len(dataset["train"])
        )
        # dataset["test"] = dataset["test"].add_column(
        #     "evaluation_method", [dataset.evaluation_method] * len(dataset["test"])
        # )
        dataset["test"] = dataset["test"].add_column(
            "dataset", [dataset.name] * len(dataset["test"])
        )
        updated_datasets.append(dataset)
    datasets = updated_datasets

    train_set = concatenate_datasets([dataset["train"].dataset for dataset in datasets])
    test_out_dist = concatenate_datasets(
        [dataset["test"].dataset for dataset in datasets]
    )

    # check if features are the same in train and test set
    # if not and one of them is null type, change to other type
    for feature_name in train_set.features:
        if not feature_name in test_out_dist.features:
            raise ValueError(
                f"Feature {feature_name} is not present in test set but in train set"
            )
        if train_set.features[feature_name] != test_out_dist.features[feature_name]:
            if train_set.features[feature_name] == Value("null"):
                train_set = train_set.cast_column(
                    feature_name, test_out_dist.features[feature_name]
                )
            elif test_out_dist.features[feature_name] == Value("null"):
                test_out_dist = test_out_dist.cast_column(
                    feature_name, train_set.features[feature_name]
                )

    # split train set to get in_dist and finetune set
    splitted = train_set.train_test_split(test_size=test_size_in_dist, seed=seed)
    train_set, test_in_dist = splitted["train"], splitted["test"]
    splitted = train_set.train_test_split(
        test_size=(test_size_finetune * total) / len(train_set), seed=seed
    )
    train_set, finetune_set = splitted["train"], splitted["test"]

    train_proportion_left = 1- test_size_in_dist - test_size_finetune
    absolute_amounts = [int(amount * train_proportion_left) for amount in absolute_amounts]

    result = DatasetDict(
        {
            "train": train_set,
            "test_out_dist": test_out_dist,
            "test_in_dist": test_in_dist,
            "finetune": finetune_set,
        }
    )
    if shuffle_final_dataset:
        result = result.shuffle(seed=seed).flatten_indices()

    if print_statistics:
        print_dataset_statistics(
            datasets,
            absolute_amounts,
            [len(dataset["test"]) for dataset in datasets],
        )

    if visualize:
        visualize_datasets(
            datasets,
            absolute_amounts,
            [len(dataset["test"]) for dataset in datasets],
        )

    return result


# find a distribution of samples that best matches the requested amounts
# calculate the relative sizes of the groups based on the requested amounts
# set missing samples to total requested samples
# while total missing samples > 0
# assigning each group the minimum number of missing*relative size and available samples ( floor rounding)
# calcualte total missing samples as the difference between the requested and assigned samples
# to distribute the remaining samples that are not assigned due to rounding:
# while total missing samples > 0
# for every group
# if the group has available samples
# this group +1 sample
def find_distribution(available, requested):
    total_requested, total_available = sum(requested), sum(available)
    if total_available < total_requested:
        raise ValueError(
            f"Not enough data available for the requested amounts. Total available data: {total_available}, requested total: {total_requested}"
        )
    relative_sizes = [r / total_requested for r in requested]
    assigned = [0] * len(requested)

    missing = total_requested
    while missing > 0:
        for i, (r, av, ass) in enumerate(zip(relative_sizes, available, assigned)):
            assigned[i] += min(math.floor(r * missing), av - ass)
        new_missing = total_requested - sum(assigned)
        if new_missing == missing:
            break
        missing = new_missing
    while missing > 0:
        for i, (av, ass) in enumerate(zip(available, assigned)):
            if av - ass > 0 and missing > 0:
                assigned[i] += 1
                missing -= 1
    for av, ass in zip(available, assigned):
        assert ass <= av, f"assigned more ({ass}) then available ({av})"
    return assigned


def reduce_total(available, absolute, total):
    proportion_of_total = min(
        available / amount for amount, available in zip(absolute, available)
    )
    total = math.floor(total * proportion_of_total)
    absolute = [math.floor(amount * proportion_of_total) for amount in absolute]
    missing = total - sum(absolute)
    while missing > 0:
        for i in range(available):
            if missing > 0:
                absolute[i] += 1
                missing -= 1
    return total, absolute


# use seaportn chart to visualize the datasets
# store as png
def visualize_datasets(datasets, absolute_amounts_train, absolute_amounts_test):
    total_train = sum(absolute_amounts_train)
    total_test = sum(absolute_amounts_test)
    relative_amounts_train = [amount / total_train for amount in absolute_amounts_train]
    relative_amounts_test = [amount / total_test for amount in absolute_amounts_test]

    labels = [dataset.name for dataset in datasets]

    # Create DataFrames for plotting
    train_data = pd.DataFrame(
        {
            "Dataset": labels,
            "Proportion": relative_amounts_train,
            "Absolute": absolute_amounts_train,
        }
    )

    test_data = pd.DataFrame(
        {
            "Dataset": labels,
            "Proportion": relative_amounts_test,
            "Absolute": absolute_amounts_test,
        }
    )

    # filter out datasets with 0 samples
    train_data = train_data[train_data["Absolute"] > 0]
    test_data = test_data[test_data["Absolute"] > 0]

    # Create labels in the format "name (absolute)"
    train_data["Label"] = train_data.apply(
        lambda row: f"{row['Dataset']} ({row['Absolute']})", axis=1
    )
    test_data["Label"] = test_data.apply(
        lambda row: f"{row['Dataset']} ({row['Absolute']})", axis=1
    )

    # sort by proportion
    train_data = train_data.sort_values("Proportion", ascending=False)
    test_data = test_data.sort_values("Proportion", ascending=False)

    # Plot Train dataset
    plt.figure(figsize=(10, 8))
    train_barplot = sns.barplot(x="Proportion", y="Label", data=train_data)
    plt.title(f"Train Dataset Proportions (Total: {total_train})")
    plt.xlabel("Proportion")
    plt.yticks(fontsize=8)
    plt.ylabel("Dataset")
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.savefig("train_dataset.png")
    plt.close()

    # Plot Test dataset
    plt.figure(figsize=(10, 8))
    test_barplot = sns.barplot(x="Proportion", y="Label", data=test_data)
    plt.title(f"Out of Distribution Test Dataset Proportions (Total: {total_test})")
    plt.xlabel("Proportion")
    plt.yticks(fontsize=8)
    plt.ylabel("Dataset")
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.savefig("test_dataset.png")
    plt.close()


def print_dataset_statistics(datasets, absolute_amounts_train, absolute_amounts_test):
    total_train = sum(absolute_amounts_train)
    total_test = sum(absolute_amounts_test)
    relative_amounts_train = [amount / total_train for amount in absolute_amounts_train]
    relative_amounts_test = [amount / total_test for amount in absolute_amounts_test]

    random_correct_accuracy_train = sum(
        dataset.random_guess_accuracy_estimation * amount
        for dataset, amount in zip(datasets, relative_amounts_train)
    )
    random_correct_accuracy_test = sum(
        dataset.random_guess_accuracy_estimation * amount
        for dataset, amount in zip(datasets, relative_amounts_test)
    )

    train_stats = []
    test_stats = []

    print(f"Statistics for new dataset with {total_train} train samples:")
    for dataset, abs_amount, rel_amount in zip(
        datasets, absolute_amounts_train, relative_amounts_train
    ):
        train_stats.append(
            {
                "Dataset": dataset.name,
                "Proportion": rel_amount,
                "Absolute": abs_amount,
                "Random correct accuracy": f"{dataset.random_guess_accuracy_estimation*100:.2f}%",
            }
        )
        print(
            f"Dataset {dataset.name:-<35} {abs_amount:6} ({rel_amount*100:.2f}%) samples"
        )
    print(f"Random correct accuracy: {random_correct_accuracy_train*100:.2f}%")
    print()
    print(f"Statistics for new dataset with {total_test} test samples:")
    for dataset, abs_amount, rel_amount in zip(
        datasets, absolute_amounts_test, relative_amounts_test
    ):
        test_stats.append(
            {
                "Dataset": dataset.name,
                "Proportion": rel_amount,
                "Absolute": abs_amount,
                "Random correct accuracy": f"{dataset.random_guess_accuracy_estimation*100:.2f}%",
            }
        )
        print(
            f"Dataset {dataset.name:-<35} {abs_amount:6} ({rel_amount*100:.2f}%) samples"
        )
    print(f"Random correct accuracy: {random_correct_accuracy_test*100:.2f}%")

    train_stats.append(
        {
            "Dataset": "Total",
            "Proportion": 1,
            "Absolute": total_train,
            "Random correct accuracy": f"{random_correct_accuracy_train*100:.2f}%",
        }
    )

    test_stats.append(
        {
            "Dataset": "Total",
            "Proportion": 1,
            "Absolute": total_test,
            "Random correct accuracy": f"{random_correct_accuracy_test*100:.2f}%",
        }
    )

    train_df = pd.DataFrame(train_stats)
    test_df = pd.DataFrame(test_stats)

    # filter out datasets with 0 samples
    train_df = train_df[train_df["Absolute"] > 0]
    test_df = test_df[test_df["Absolute"] > 0]

    # sort by descending proportion
    train_df = train_df.sort_values("Proportion", ascending=False)
    test_df = test_df.sort_values("Proportion", ascending=False)

    # map proportion to percentage
    train_df["Proportion"] = train_df["Proportion"].map(lambda x: f"{x*100:.2f}%")
    test_df["Proportion"] = test_df["Proportion"].map(lambda x: f"{x*100:.2f}%")

    # make first row (total) bold (add markdown syntax)
    train_df.iloc[0] = train_df.iloc[0].apply(lambda x: f"**{x}**")
    test_df.iloc[0] = test_df.iloc[0].apply(lambda x: f"**{x}**")

    # store as markdown tables
    train_df.to_markdown("train_statistics.md", index=False)
    test_df.to_markdown("test_statistics.md", index=False)
