from paraphrasor import (
    OpenAiParaphrasor,
    LLama3Paraphrasor,
    GLM4Paraphrasor,
    Qwen2Paraphrasor,
    MistralParaphrasor,
)
from datamodel import (
    Llama3EnInput,
    MistralEnInput,
    GLM4EnInput,
    OpenAiEnInput,
    OpenAiZhInput,
    GLM4ZhInput,
    Qwen2ZhInput,
    Output,
    SequencesOutput,
    RandomSelector,
    MaxPPLSelector,
    MinPPLSelector,
)
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import sys
import os

sys.path.append("..")
from utils import jdump, jload


def get_paraphrasor(cfg: DictConfig):
    model_config = cfg["model"]
    model_name = model_config["model_name"]
    model_path = model_config["model_path"]
    if model_config.get("other_config"):
        other_config = model_config["other_config"]
    else:
        other_config = {}

    generation_config = cfg["generation"]

    if model_name in ["gpt-4o-mini", "gpt-4o"]:
        return OpenAiParaphrasor(
            model_path,
            **generation_config,
            **other_config,
        )
    elif model_name == "glm4":
        return GLM4Paraphrasor(
            model_path,
            **generation_config,
            **other_config,
        )
    elif model_name in ["llama3-8B", "llama3-70B"]:
        return LLama3Paraphrasor(
            model_path,
            **generation_config,
            **other_config,
        )
    elif model_name == "mistral-7B":
        return MistralParaphrasor(
            model_path,
            **generation_config,
            **other_config,
        )
    elif model_name == "qwen2":
        return Qwen2Paraphrasor(
            model_path,
            **generation_config,
            **other_config,
        )


def get_dataset(cfg: DictConfig):
    data_config = cfg["data"]
    language = data_config["language"]
    dataset = data_config["dataset"]
    datalevel = data_config["datalevel"]
    df = pd.read_csv(dataset)
    return language, datalevel, df


def get_en_input_formator(model_name: str):
    if model_name in ["llama3-70B", "llama3-8B"]:
        return Llama3EnInput()
    elif model_name in ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"]:
        return OpenAiEnInput()
    elif model_name == "glm4":
        return GLM4EnInput()
    elif model_name == "mistral-7B":
        return MistralEnInput()
    else:
        raise ValueError(f"not support {model_name}")


def get_zh_input_formator(model_name: str):
    if model_name in ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo"]:
        return OpenAiZhInput()
    elif model_name == "glm4":
        return GLM4ZhInput()
    elif model_name == "qwen2":
        return Qwen2ZhInput()
    else:
        raise ValueError(f"not support {model_name}")


def get_input_formator(cfg: DictConfig):
    language = cfg["data"]["language"]
    assert language in ["en", "zh"]
    model_name = cfg["model"]["model_name"]
    if language == "en":
        return get_en_input_formator(model_name)
    else:
        return get_zh_input_formator(model_name)


def get_selecotr(cfg: DictConfig):
    mode = cfg["selector"]
    if mode == "random":
        return RandomSelector()
    elif mode == "max":
        return MaxPPLSelector()
    elif mode == "min":
        return MinPPLSelector()
    else:
        raise ValueError("only support: min, max, random")


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    input_formator = get_input_formator(cfg)

    print(type(input_formator))
    selector = get_selecotr(cfg)
    lang, datalevel, df = get_dataset(cfg)
    paraphrasor = get_paraphrasor(cfg)
    total_step = cfg["total_step"]
    out_path = os.path.join(
        cfg["out_dir"], lang, datalevel, str(paraphrasor), cfg["selector"], "data.jsonl"
    )

    if os.path.exists(out_path):
        results = jload(out_path)
    else:
        results = []

    df = df.iloc[len(results) :]

    for i, row in tqdm(df.iterrows(), total=len(df)):
        sequences_outputs = paraphrasor.recursive_rephrase(
            text=row["text"],
            input_formater=input_formator,
            selector=selector,
            total_step=total_step,
        )
        results.append(sequences_outputs.to_dict())
        jdump(results[-1], out_path, "a")


if __name__ == "__main__":
    main()
