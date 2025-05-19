import numpy as np
import random
from abc import abstractmethod


class ParaphraseInput:
    @abstractmethod
    def format_input(self, text: str):
        pass

    @abstractmethod
    def reverse_sample(self, text: str, response_text: str):
        pass


class EnglishInput(ParaphraseInput):
    def format_input(self, text: str):
        # return f"please paraphrase the following text:\n\n{text}"
        return f"""Please rewrite the following paragraph with the goal of enhancing grammatical and syntactical variety without changing the original meaning. Pay attention to increasing the complexity and variation of sentence structures, using different conjunctions and clause constructions to make the expression more diverse and rich, while maintaining the core information and logical coherence of the original text. Specifically, avoid repetitive sentence patterns and try to express the same ideas in different ways.\n\nparagraph: {text}\n\n"""

    def format_response(self, text: str, response: str):
        return self.format_input(text), [
            {
                "role": "assistant",
                "content": response,
            },
        ]

    def reverse_sample(self, text: str, response_text: str):
        response = response_text.split("\n")
        R_formated_input = self.format_input(response[-1])
        R_response = [
            {
                "role": "assistant",
                "content": "\n".join(response[:-1] + [text]),
            },
        ]
        return R_formated_input, R_response


class Llama3EnInput(EnglishInput):
    def format_input(self, text: str):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": super().format_input(text),
            },
        ]
        return messages


class MistralEnInput(EnglishInput):
    def format_input(self, text: str):
        messages = [
            {
                "role": "user",
                "content": super().format_input(text),
            },
        ]
        return messages


class GLM4EnInput(EnglishInput):
    def format_input(self, text: str):
        messages = [
            {
                "role": "user",
                "content": super().format_input(text),
            },
        ]
        return messages


class OpenAiEnInput(EnglishInput):
    def format_input(self, text: str):
        return super().format_input(text)


class ChineseInput(ParaphraseInput):
    def format_input(self, text: str):
        return f"请改写下面文本:\n\n{text}"

    def reverse_sample(self, text: str, response_text: str):
        response = response_text.split("\n")
        R_formated_input = self.format_input(response[-1])
        R_response = [
            {
                "role": "assistant",
                "content": "\n".join(response[:-1] + [text]),
            },
        ]
        return R_formated_input, R_response

    def format_response(self, text: str, response: str):
        return self.format_input(text), [
            {
                "role": "assistant",
                "content": response,
            },
        ]


class OpenAiZhInput(ChineseInput):
    def format_input(self, text: str):
        return super().format_input(text)


class GLM4ZhInput(ChineseInput):
    def format_input(self, text: str):
        messages = [
            {
                "role": "user",
                "content": super().format_input(text),
            },
        ]
        return messages


class Qwen2ZhInput(ChineseInput):
    def format_input(self, text: str):
        messages = [
            {"role": "system", "content": "你是一个改写助手"},
            {
                "role": "user",
                "content": super().format_input(text),
            },
        ]
        return messages


class Output:
    _text: str
    _paraphrases: list[str]
    _perplexity: np.ndarray
    _reverse_perplexity: np.ndarray
    _model: str

    def __init__(
        self,
        text: str,
        paraphrases: list[str],
        perplexity: np.ndarray,
        reverse_ppls: np.ndarray,
        model: str = None,
    ):
        self._text = text
        self._paraphrases = paraphrases
        self._perplexity = perplexity
        self._reverse_perplexity = reverse_ppls
        self._model = model

    @property
    def paraphrase(self):
        return random.choice(self._paraphrases)

    @property
    def max_ppl_paraphrase(self):
        if self._perplexity is None:
            raise ValueError("paraphrases have not perplexity")
        idx = np.argmax(self._perplexity)
        return self._paraphrases[idx]

    @property
    def min_ppl_paraphrase(self):
        if self._perplexity is None:
            raise ValueError("paraphrases have not perplexity")
        idx = np.argmin(self._perplexity)
        return self._paraphrases[idx]

    def to_dict(self):
        return {
            "text": self._text,
            "paraphrases": self._paraphrases,
            "perplexity": (
                self._perplexity.tolist() if self._perplexity is not None else None
            ),
            "reverse_perplexity": (
                self._reverse_perplexity.tolist()
                if self._reverse_perplexity is not None
                else None
            ),
            "model": self._model,
        }


class SequencesOutput:
    sequences: list[Output]

    def __init__(self, sequences=None):
        if sequences is None:
            self.sequences = []
        if isinstance(sequences, list):
            self.sequences = sequences
        else:
            raise ValueError("sequences must be list")

    def append(self, output: Output):
        self.sequences.append(output)

    def to_list(self):
        return [output.to_dict() for output in self.sequences]

    def to_dict(self):
        return {"sequences": self.to_list()}


class Selector:
    @abstractmethod
    def select_paraphrase(self, output: Output) -> str:
        return output.paraphrase


class RandomSelector(Selector):
    def select_paraphrase(self, output: Output) -> str:
        return super().select_paraphrase(output)


class MaxPPLSelector(Selector):
    def select_paraphrase(self, output: Output) -> str:
        return output.max_ppl_paraphrase


class MinPPLSelector(Selector):
    def select_paraphrase(self, output: Output) -> str:
        return output.min_ppl_paraphrase
