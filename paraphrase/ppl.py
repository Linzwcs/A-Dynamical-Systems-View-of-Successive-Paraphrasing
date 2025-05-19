from paraphrasor import LLama3Paraphrasor
import pandas as pd
from tqdm import tqdm
import sys
from datamodel import Llama3EnInput

sys.path.append("..")
from utils import jdump, jload


if __name__ == "__main__":
    paraphrasor = LLama3Paraphrasor()
    formator = Llama3EnInput()
    data = jload("../paraphrase/outv3/en/sentence/gpt-4o-mini/")
    for item in tqdm(data, total=len(data)):
        pres, fulls = [], []
        seqs = [s["text"] for s in item["sequences"]]
        for i in range(1, len(seqs)):
            text, r = seqs[i], seqs[i - 1]
            pre, res = formator.format_response(text, r)
            pres.append(pre)
            fulls.append(pre + res)
        ppls = paraphrasor.condition_PPL(pres, fulls)
        jdump([ppls.tolist()], "./varppl/gpt-4o-mini_based_llama3.json", "a")
