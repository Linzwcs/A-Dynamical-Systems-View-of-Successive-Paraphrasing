from typing import Union
import openai
import pandas as pd
from tqdm import tqdm
import numpy as np

openai.api_key = ""


def openai_sample(
    prompt,
    model,
    n=1,
    temperature=0.6,
):
    if model in ["gpt-4o", "gpt-3.5-turbo", "gpt-4-turbo", "gpt-4o-mini"]:
        print(f"\n{model} temperature: {temperature} n: {n}\nprompt:\n{prompt}\n")
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": " You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            # temperature=temperature,
            n=n,
            logprobs=True,
            temperature=1e-19,
            top_p=1e-9,
            seed=1234,
        )
        print(response["choices"][0]["message"]["content"])
        return response


def openai_rephrase_interface(prompt, model, temperature=0.6, n=1):
    response = openai_sample(
        prompt=prompt,
        model=model,
        n=n,
        temperature=temperature,
    )
    paraphases, perplexity = [], []
    for i in range(n):
        ctx = response["choices"][i]["message"]["content"]
        logprobs = [r["logprob"] for r in response["choices"][i]["logprobs"]["content"]]
        ppl = np.exp(-np.mean(logprobs))
        paraphases.append(ctx.strip().split("\n")[-1])
        # paraphases.append(ctx.split("**Rewritten:**")[-1].strip())
        perplexity.append(ppl)
    return paraphases, np.array(perplexity)
