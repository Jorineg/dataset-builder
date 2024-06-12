from typing import Any
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from collections import Counter
import string
import copy
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
import contextlib

EVALUATION_STRING_MATCH = "string_match"
EVALUATION_NUMERIC_MATCH = "numeric_match"
EVALUATION_IN_LIST = "in_list"

TOOL_PLAIN_PYTHON = "plain_python"
TOOL_PYTHON_WITH_LIBRIARIES = "python_with_libraries"
TOOL_GOOGLE_SEARCH = "google_search"
TOOL_WEB_SCRAPING_TEXT = "web_scraping_text"
TOOL_WEB_SCRAPING_HTML = "web_scraping_html"

FORMATTED_MC_COL = "formatted_mc_col"
FEW_SHOT_COL = "few_shot_col"

USE_ONLY_TRAIN = "train"
USE_ONLY_TEST = "test"

_PROMPT_VARIATIONS_CACHE_FILE = "prompt_variations_cache.json"


class ExtendetDataset:
    def __init__(
        self,
        dataset,
        name,
        type,
        out_col,
        prompt_template,
        random_guess_accuracy_estimation=None,
        description="",
        in_col="final_input",
        subset=None,
        mc_options={},
        add_few_shot=0,
        few_shot_separator_str="\n\n",
        few_shot_question_str="Question:\n",
        few_shot_answer_str="\n\nAnswer:\n",
        few_shot_question_col=None,
        helpful_tools=[],
        keep_columns=[],
        use_only=None,
        evaluation_method=EVALUATION_STRING_MATCH,
        load_max_samples=100000,
        keep_in_memory=False,
        variations_for_prompt_template=None,
        lock=contextlib.nullcontext(),
        split=None,
        **kwargs,
    ):
        formatter = string.Formatter()
        parsed = list(formatter.parse(prompt_template))
        # check if all fields are provided, so no empty {} fields
        # but allow ignored fields {{ ignored }}
        # they will have field=None
        if not all([field != "" for _, field, _, _ in parsed]):
            raise ValueError(
                f"Prompt template '{prompt_template}' contains empty fields. Please provide all fields."
            )

        self.mc_col = mc_options.get("mc_col", None)
        self.mc_sep = mc_options.get("mc_sep", ", ")
        self.add_mc_labels = mc_options.get("add_mc_labels", False)

        if subset:
            self.dataset = load_dataset(
                dataset, subset, keep_in_memory=keep_in_memory, split=split
            )
        else:
            self.dataset = load_dataset(
                dataset, keep_in_memory=keep_in_memory, split=split
            )

        self.name = name
        self.type = type
        self.prompt_template = prompt_template
        self.in_cols = [field for _, field, _, _ in parsed if field] + (
            [self.mc_col] if self.mc_col else []
        )
        self.in_col = in_col
        self.description = description
        self.out_col = out_col
        self.random_guess_accuracy_estimation = random_guess_accuracy_estimation
        self.subset = subset
        self.add_few_shot = add_few_shot
        self.few_shot_separator_str = few_shot_separator_str
        self.few_shot_question_str = few_shot_question_str
        self.few_shot_answer_str = few_shot_answer_str
        self.few_shot_question_col = few_shot_question_col
        self.helpful_tools = helpful_tools
        self.keep_columns = keep_columns
        self.evaluation_method = evaluation_method
        self.use_only = use_only
        self.lock = lock
        self.split = split

        self.map_kwargs = {
            "batched": True,
            "batch_size": -1,
        }

        # convert to datasetdict if not already
        if not isinstance(self.dataset, DatasetDict):
            self.dataset = DatasetDict({"train": self.dataset})

        # merge every split that is not test to train
        # if no train split is available, merge all splits to train
        unknown_splits = (
            [key for key in self.dataset.keys() if key != "test"]
            if "train" in self.dataset
            else list(self.dataset.keys())
        )
        if unknown_splits:
            self.dataset["train"] = concatenate_datasets(
                [self.dataset[key] for key in unknown_splits]
            )
        for key in unknown_splits:
            if key != "train":
                del self.dataset[key]

        if FORMATTED_MC_COL in self.in_cols and not self.mc_col:
            raise ValueError(
                f"Column {FORMATTED_MC_COL} is in in_cols but mc_col is not set."
            )

        if add_few_shot > 0 and not self.few_shot_question_col:
            raise ValueError(
                f"add_few_shot is set to {add_few_shot} but few_shot_question_col is not set."
            )

        if FEW_SHOT_COL in self.in_cols and add_few_shot <= 0:
            raise ValueError(
                f"Column {FEW_SHOT_COL} is in in_cols but add_few_shot is not set."
            )

        exclude_cols = [FEW_SHOT_COL, FORMATTED_MC_COL]

        for col in self.in_cols:
            if (
                col not in self.dataset["train"].column_names
                and col not in exclude_cols
            ):
                raise ValueError(
                    f"Column {col} not found in dataset. Available columns: {self.dataset['train'].column_names}"
                )

        if load_max_samples is not None and load_max_samples > 0:
            for key in self.dataset.keys():
                if len(self.dataset[key]) > load_max_samples:
                    print(
                        f"Reducing {key} dataset from {len(self.dataset[key])} to {load_max_samples} samples"
                    )
                    self.dataset[key] = self.dataset[key].select(
                        range(load_max_samples)
                    )

        # filter out rows where any in_col or out_col is None or empty string
        original_len = len(self)

        relevant_cols = [
            x
            for x in self.in_cols + [self.out_col]
            if x in self.dataset["train"].column_names
        ]

        def batch_filter(batch):
            transformed_batch = [
                dict(zip(batch.keys(), x)) for x in zip(*batch.values())
            ]
            return [
                all([x[col] is not None and x[col] != "" for col in relevant_cols])
                for x in transformed_batch
            ]

        self.dataset = self.dataset.filter(batch_filter, **self.map_kwargs)

        # remove any empty splits
        for key in list(self.dataset.keys()):
            if len(self.dataset[key]) == 0:
                del self.dataset[key]

        if len(self) < original_len:
            print(
                f"Filtered out {original_len - len(self)} rows with empty in_col or out_col. This is {100 * (original_len - len(self)) / original_len:.2f}% of the dataset."
            )

        self._add_few_shot_col()
        self._add_formatted_mc_col()

        prompt_variations = [prompt_template]
        if variations_for_prompt_template and variations_for_prompt_template > 1:
            prompt_variations = self._add_prompt_variations(
                prompt_template, variations_for_prompt_template, parsed
            )

        # apply prompt template
        def batch_apply_prompt_template(batch):
            # transform batch from column dict to list of dicts
            transformed_batch = [
                dict(zip(batch.keys(), x)) for x in zip(*batch.values())
            ]
            # sample batch size prompt templates from variations
            prompt_templates = random.choices(
                prompt_variations, k=len(transformed_batch)
            )
            batch[in_col] = [
                template.format(**x)
                for template, x in zip(prompt_templates, transformed_batch)
            ]
            return batch

        self.dataset = self.dataset.map(batch_apply_prompt_template, **self.map_kwargs)

        # compute accuracy estimation if None
        # this is count of most frequent output divided by total count
        if self.random_guess_accuracy_estimation is None:
            self.random_guess_accuracy_estimation = Counter(
                [str(x) for x in self["train"][self.out_col]]
            ).most_common(1)[0][1] / len(self["train"])
        
        # invalidate lock to make file pickable
        self.lock = None

    def __getitem__(self, key):
        item = self.dataset[key]
        if isinstance(item, Dataset) or isinstance(item, DatasetDict):
            copy_self = copy.copy(self)
            copy_self.dataset = item
            return copy_self
        return item

    def __iter__(self):
        for item in self.dataset:
            yield item

    def __setitem__(self, key, value):
        if isinstance(value, ExtendetDataset):
            value = value.dataset
        self.dataset[key] = value

    def __contains__(self, key):
        return key in self.dataset

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, __name: str) -> Any:
        attr = getattr(self.dataset, __name)
        if callable(attr):

            def wrapper(*args, **kwargs):
                ret = attr(*args, **kwargs)
                if isinstance(ret, Dataset) or isinstance(ret, DatasetDict):
                    copy_self = copy.copy(self)
                    copy_self.dataset = ret
                    return copy_self
                return ret

            return wrapper
        return attr

    def __getstate__(self) -> object:
        return vars(self)

    def __setstate__(self, state: object) -> None:
        vars(self).update(state)

    def _add_prompt_variations(self, prompt_template, variation_count, original_fields):
        with self.lock:
            if os.path.exists(_PROMPT_VARIATIONS_CACHE_FILE):
                with open(_PROMPT_VARIATIONS_CACHE_FILE, "r") as file:
                    cache = json.load(file)
                if self.name in cache:
                    entry = cache[self.name]
                    if (
                        entry["variation_count"] == variation_count
                        and entry["prompt_template"] == prompt_template
                    ):
                        print(
                            f"Using cached prompt template variations for dataset {self.name}"
                        )
                        return entry["variations"]
        load_dotenv()
        if not "OPENAI_API_KEY" in os.environ:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. Please set it to use the prompt template variations feature."
            )
        print(
            f"Creating {variation_count} variations for prompt template using openai API"
        )

        client = OpenAI()
        prompt = f"""
        Create {variation_count} variations for prompt template:
        ----- Prompt Template -----
        '{prompt_template}'
        ---------------------------
        Make sure to use exaclty the same fields as in the original prompt template.
        Keep the intent of the original prompt template, but you may vary the wording.
        You can also change the position of the instructions in the prompt.
        Output a JSON object with a single key 'prompts' that contains a list of strings.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
            ],
        )
        prompt_variations = json.loads(response.choices[0].message.content)["prompts"]
        # check if the variations are valid i.e. contain same fields as original prompt template
        for variation in prompt_variations:
            if not all(
                [
                    ("{" + field + "}") in variation
                    for _, field, _, _ in original_fields
                    if field
                ]
            ):
                raise ValueError(
                    f"Variation '{variation}' does not contain all fields of the original prompt template '{prompt_template}'."
                )

        result = prompt_variations + [prompt_template]

        # read cache again and update it
        with self.lock:
            if os.path.exists(_PROMPT_VARIATIONS_CACHE_FILE):
                with open(_PROMPT_VARIATIONS_CACHE_FILE, "r") as file:
                    cache = json.load(file)
            else:
                cache = {}
            cache[self.name] = {
                "variation_count": variation_count,
                "prompt_template": prompt_template,
                "variations": result,
            }
            with open(_PROMPT_VARIATIONS_CACHE_FILE, "w") as file:
                json.dump(cache, file, indent=2)
        return result

    def _add_few_shot_col(self):
        """
        Add few shot examples to the dataset
        """
        if self.add_few_shot <= 0:
            return
        examples = []
        for i in range(self.add_few_shot):
            in_str = self["train"][self.few_shot_question_col][i]
            out_str = self["train"][self.out_col][i]
            example = f"{self.few_shot_question_str}{in_str}{self.few_shot_answer_str}{out_str}"
            examples.append(example)
        few_shot = self.few_shot_separator_str.join(examples)

        self.dataset = self.dataset.add_column([few_shot] * len(self), FEW_SHOT_COL)

        # remove the few shot examples from the train set
        self.dataset["train"] = (
            self["train"].select(range(self.add_few_shot, len(self["train"]))).dataset
        )
        print(f"Added {self.add_few_shot} few shot examples to the dataset")

    def _add_formatted_mc_col(self):
        """
        Add multiple choice options to the dataset
        """

        if not self.mc_col:
            return

        def num_to_char(num):
            return chr(ord("A") + int(num))

        def batch_add_mc_col(batch):
            mc_col = batch[self.mc_col]
            if self.add_mc_labels:
                # if outcol type is numeric, convert to char
                if isinstance(batch[self.out_col][0], int):
                    batch[self.out_col] = [num_to_char(x) for x in batch[self.out_col]]
                mc_col = [
                    [f"{num_to_char(i)}: {option}" for i, option in enumerate(options)]
                    for options in mc_col
                ]
            batch[FORMATTED_MC_COL] = [self.mc_sep.join(options) for options in mc_col]
            return batch

        self.dataset = self.dataset.map(batch_add_mc_col, **self.map_kwargs)
        print(f"Added formatted multiple choice column {FORMATTED_MC_COL}")
