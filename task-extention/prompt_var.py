import pandas as pd
import sys

sys.path.append("..")
from utils import jdump, jload
from tqdm import tqdm
from tasks import (
    Formator,
    FormalTransfer,
    PolishFormator,
    SimplerFormator,
    EnZhTransfer,
    ZhEnTransfer,
    InformalTransfer,
    ParaphraseFormator,
    RephraseFormator,
    RewriteFormator,
)

from api import openai_sample


def sample_sequences(text, formator_seqs: list[Formator]):

    sequences = [text]
    for f in formator_seqs:
        inputs = f.format_input(sequences[-1])
        r = openai_sample(inputs, "gpt-4o-mini")
        ctx = r["choices"][0]["message"]["content"]
        sequences.append(ctx.split("\n")[-1])
    return {"sequences": sequences}


if __name__ == "__main__":
    import random

    out_path = "./out/prompt_var.jsonl"
    df = pd.read_csv("../data/paragraph/paragraph.csv")
    # formator_seqs = [SimplerFormator()] * 10
    candidates = [
        ParaphraseFormator(),
        RewriteFormator(),
        PolishFormator(),
        RephraseFormator(),
    ]
    # formator_seqs = [FormalTransfer(), InformalTransfer()] * 5
    try:
        out = jload(out_path)
    except:
        out = []
    df = df.iloc[len(out) :]
    for i, row in tqdm(df.iterrows(), total=len(df)):
        text = row["text"]
        formator_seqs = [random.choice(candidates) for i in range(15)]
        prompts = [str(_) for _ in formator_seqs]
        seqs = sample_sequences(text, formator_seqs)
        seqs["prompts"] = prompts
        jdump(seqs, out_path, "a")
