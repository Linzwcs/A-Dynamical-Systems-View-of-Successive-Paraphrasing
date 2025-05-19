import pandas as pd
import random
from typing import Any, Dict, Tuple
from tqdm import tqdm
import sys

sys.path.append("..")
from utils import jload, jdump
import os
import random
import openai
import pandas as pd
from tqdm import tqdm
from requests import HTTPError
import requests
import json

openai.api_key = ""


def openai_sample(
    prompt,
    model,
    temperature=0.7,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=1,
    max_num_tokens=1024,
):

    if model in ["gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o-mini"]:
        print(f"\n{model}\nprompt:\n{prompt}\n")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": " You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response["choices"][0]["message"]["content"]
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_num_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    return response["choices"][0].text


def random_paraphrase(prompt, text, PMs, num_paraphrase=15):
    out = [{"text": text, "model": "original"}]
    for _ in range(num_paraphrase):
        PM = random.choice(PMs)
        out.append(PM(prompt.format(text=out[-1]["text"])))
    return out


class ParaModel:
    def paraphrase(self, text):
        pass


class OpenAiModel(ParaModel):
    def __init__(self, model_name):
        self.model_name = model_name

    def paraphrase(self, text):
        para_text = openai_sample(text, self.model_name)
        return {"text": para_text.split("\n")[-1], "model": self.model_name}

    def __str__(self) -> str:
        return self.model_name


class GLM4Model(ParaModel):

    def __init__(self) -> None:
        self.base_url = "http://127.0.0.1:8000/models/glm4"

    def paraphrase(self, text: str):
        query = {"text": text}
        r = requests.get(self.base_url, query)
        if r.status_code == 200:
            return json.loads(r.content.decode())
        else:
            return None

    def __str__(self) -> str:
        return "GLM-4"


class MistralModel(ParaModel):

    def __init__(self) -> None:
        self.base_url = "http://127.0.0.1:8000/models/mistral"

    def paraphrase(self, text: str):
        query = {"text": text}
        r = requests.get(self.base_url, query)
        if r.status_code == 200:
            return json.loads(r.content.decode())
        else:
            return None

    def __str__(self) -> str:
        return "Mistral"


class LLamaModel(ParaModel):
    def __init__(self) -> None:
        self.base_url = "http://127.0.0.1:8000/models/llama3"

    def paraphrase(self, text: str):
        query = {"text": text}
        r = requests.get(self.base_url, query)
        if r.status_code == 200:
            return json.loads(r.content.decode())
        else:
            return None

    def __str__(self) -> str:
        return "LLama"


def sample_sequences(text, prompt: str, models: list[ParaModel], total=15):
    sequences = [text]
    choices = []
    for _ in range(total):
        # print(models)
        pm = random.choice(models)
        print(pm)
        out = pm.paraphrase(prompt.format(text=sequences[-1]))
        sequences.append(out["text"])
        choices.append(out["model"])
    return {
        "sequences": sequences,
        "choices": choices,
    }


if __name__ == "__main__":
    df = pd.read_csv("../data/sentence/data-en.csv").iloc[:100]
    num = 15
    out_path = "./out/model_var.json"
    try:
        initial_state = jload(out_path)
    except:
        initial_state = []

    PMs = [
        GLM4Model(),
        LLamaModel(),
        OpenAiModel("gpt-4o"),
        OpenAiModel("gpt-4o-mini"),
    ]

    prompt = "please paraphrase the following text:\n\n{text}"
    df = df.iloc[len(initial_state) :]

    for i, row in tqdm(df.iterrows(), total=len(df)):
        print(PMs)
        seqs = sample_sequences(row["text"], prompt, PMs, num)
        jdump(seqs, out_path, "a")
