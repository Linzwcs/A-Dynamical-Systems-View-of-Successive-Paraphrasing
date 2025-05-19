from datamodel import (
    ParaphraseInput,
    Output,
    SequencesOutput,
    Selector,
)
from api import openai_rephrase_interface
from abc import abstractmethod
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch
import numpy as np

from torch.nn.functional import cross_entropy


def calculate_perplexity(outputs, input_ids, terminators):
    scores = outputs.scores
    scores = [torch.log_softmax(s, dim=-1) for s in scores]
    log_likelihoods = []
    for i, seq in enumerate(outputs.sequences):
        response = seq[input_ids.shape[-1] :]
        response = [tok for tok in response if tok not in terminators]
        log_prob = 0
        for j, tok in enumerate(response):
            log_prob += scores[j][i, tok]
        log_prob /= len(response)
        log_likelihoods.append(log_prob)
    ppls = torch.exp(-torch.Tensor(log_likelihoods))
    return ppls.numpy().tolist()


class Paraphrasor:
    def __init__(
        self,
        model_name,
        temperature=0.6,
        n=1,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.n = n

    @abstractmethod
    def rephrase(
        self,
        text: str,
        input_formater: ParaphraseInput,
    ) -> Output:
        pass

    def recursive_rephrase(
        self,
        text: str,
        input_formater: ParaphraseInput,
        selector: Selector,
        total_step: int = 15,
    ) -> SequencesOutput:
        sequences, current_text = [], text
        for _ in range(total_step):
            output = self.rephrase(current_text, input_formater)
            sequences.append(output)
            current_text = selector.select_paraphrase(output)
            # if len(sequences) >= 3 and sequences[-1]._text == sequences[-3]._text:
            #    break
        return SequencesOutput(sequences)

    def __str__(self) -> str:
        return self.model_name


class OpenAiParaphrasor(Paraphrasor):
    def __init__(self, model_name, temperature=0.6, n=1) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            n=n,
        )

    def rephrase(self, text: str, input_formater: ParaphraseInput) -> Output:
        formated_input = input_formater.format_input(text)
        paraphrases, ppls = openai_rephrase_interface(
            prompt=formated_input,
            model=self.model_name,
            temperature=self.temperature,
            n=self.n,
        )
        return Output(
            text=text,
            paraphrases=paraphrases,
            perplexity=ppls,
            reverse_ppls=None,
            model=self.model_name,
        )


class ModelParaprhasor(Paraphrasor):
    def __init__(
        self,
        model_name,
        device_map="auto",
        temperature=0.6,
        n=1,
        top_p=0.9,
        max_new_tokens=256,
    ):
        model_path = model_name
        model_name = model_name.split("/")[-1]
        super().__init__(model_name, temperature, n)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.temperature = temperature
        self.num_return_sequences = n
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_ids: torch.Tensor):
        return self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    def condition_PPL(self, pre_inputs, full_inputs):

        pre_inputs = self.tokenizer.apply_chat_template(
            pre_inputs,
            add_generation_prompt=True,
        )
        pre_inputs = [len(pre) for pre in pre_inputs]
        full_inputs = self.tokenizer.apply_chat_template(
            full_inputs,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        ).to(self.model.device)
        full_inputs_idxs, full_inputs_mask = (
            full_inputs.input_ids,
            full_inputs.attention_mask,
        )
        full_target = full_inputs_idxs.clone()
        for i in range(full_inputs_idxs.shape[0]):
            full_target[i, 0 : pre_inputs[i]] = -100
            tail = (1 - full_inputs_mask[i]).sum()
            if tail > 0:
                full_target[i, -tail:] = -100
        outputs = self.model(
            input_ids=full_inputs_idxs,
            attention_mask=full_inputs_mask,
            labels=full_target,
        )
        logits = outputs.logits
        full_target = full_target[:, 1:]
        losses = []
        for logit, label in zip(logits, full_target):
            losses.append(cross_entropy(logit[:-1], label).item())

        perplexity = torch.exp(torch.Tensor(losses))
        return perplexity

    def get_preparation(
        self, text: str, response: str, input_formater: ParaphraseInput
    ):
        formated_input = input_formater.format_input(text)
        input_ids = self.tokenizer.apply_chat_template(
            formated_input, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

    @torch.no_grad()
    def rephrase(self, text: str, input_formater: ParaphraseInput) -> Output:
        formated_input = input_formater.format_input(text)
        input_ids = self.tokenizer.apply_chat_template(
            formated_input, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        outputs = self.generate(input_ids)

        responses = self.tokenizer.batch_decode(
            outputs.sequences[:, input_ids.shape[-1] :], skip_special_tokens=True
        )

        r_inputs, r_full = [], []
        for r in responses:
            r_formated_input, r_response = input_formater.reverse_sample(text, r)
            r_inputs.append(r_formated_input)
            r_full.append(r_formated_input + r_response)

        r_perplexity = self.condition_PPL(r_inputs, r_full)
        inputs, full = [], []
        for r in responses:
            pre_inputs, res = input_formater.format_response(text, r)
            inputs.append(pre_inputs)
            full.append(pre_inputs + res)
        perplexity = self.condition_PPL(inputs, full)

        paraphrases = [r.split("\n")[-1] for r in responses]
        return Output(
            text=text,
            paraphrases=paraphrases,
            perplexity=np.array(perplexity),
            reverse_ppls=np.array(r_perplexity),
            model=self.model_name,
        )


class LLama3Paraphrasor(ModelParaprhasor):
    def __init__(
        self,
        model_name,
        device_map="auto",
        temperature=0.6,
        n=1,
        top_p=0.9,
        max_new_tokens=256,
    ):
        super().__init__(model_name, device_map, temperature, n, top_p, max_new_tokens)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    def generate(self, input_ids: torch.Tensor):
        return self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.terminators,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=self.num_return_sequences,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
        )


class Qwen2Paraphrasor(ModelParaprhasor):
    def __init__(
        self,
        model_name,
        device_map="auto",
        temperature=0.6,
        n=1,
        top_p=0.9,
        max_new_tokens=256,
    ):
        super().__init__(model_name, device_map, temperature, n, top_p, max_new_tokens)


class GLM4Paraphrasor(ModelParaprhasor):
    def __init__(
        self,
        model_name,
        device_map="auto",
        temperature=0.6,
        n=1,
        top_p=0.9,
        max_new_tokens=256,
    ):
        super().__init__(model_name, device_map, temperature, n, top_p, max_new_tokens)

    def condition_PPL(self, pre_inputs, full_inputs):
        pre_inputs = self.tokenizer.apply_chat_template(
            pre_inputs,
            add_generation_prompt=True,
        )
        pre_inputs = [len(pre) for pre in pre_inputs]
        full_inputs = self.tokenizer.apply_chat_template(
            full_inputs,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        ).to(self.model.device)
        full_inputs_idxs, full_inputs_mask = (
            full_inputs.input_ids,
            full_inputs.attention_mask,
        )
        full_target = full_inputs_idxs.clone()
        for i in range(full_inputs_idxs.shape[0]):
            bound = (1 - full_inputs_mask[i]).sum() + pre_inputs[i]
            full_target[i, 0:bound] = -100
        outputs = self.model(
            input_ids=full_inputs_idxs,
            attention_mask=full_inputs_mask,
            labels=full_target,
        )
        logits = outputs.logits
        full_target = full_target[:, 1:]
        losses = []
        for logit, label in zip(logits, full_target):
            losses.append(cross_entropy(logit[:-1], label).item())
        perplexity = torch.exp(torch.Tensor(losses))
        return perplexity


class MistralParaphrasor(ModelParaprhasor):
    def __init__(
        self,
        model_name,
        device_map="auto",
        temperature=0.6,
        n=1,
        top_p=0.9,
        max_new_tokens=256,
    ):
        super().__init__(model_name, device_map, temperature, n, top_p, max_new_tokens)


# class LLama3Paraphrasor(ModelParaprhasor):
#     def __init__(
#         self,
#         model_name,
#         device_map="auto",
#         temperature=0.6,
#         n=1,
#         top_p=0.9,
#         max_new_tokens=256,
#     ):
#         model_path = model_name
#         model_name = model_name.split("/")[-1]
#         super().__init__(model_name, temperature, n)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_path)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             device_map=device_map,
#             torch_dtype=torch.bfloat16,
#         )
#         self.terminators = [
#             self.tokenizer.eos_token_id,
#             self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
#         ]
#         self.temperature = temperature
#         self.num_return_sequences = n
#         self.top_p = top_p
#         self.max_new_tokens = max_new_tokens
#
#     @torch.no_grad()
#     def rephrase(self, text: str, input_formater: ParaphraseInput) -> Output:
#         formated_input = input_formater.format_input(text)
#         input_ids = self.tokenizer.apply_chat_template(
#             formated_input, add_generation_prompt=True, return_tensors="pt"
#         ).to(self.model.device)
#
#         outputs = self.model.generate(
#             input_ids,
#             max_new_tokens=self.max_new_tokens,
#             eos_token_id=self.terminators,
#             temperature=self.temperature,
#             top_p=self.top_p,
#             num_return_sequences=self.num_return_sequences,
#             do_sample=True,
#             return_dict_in_generate=True,
#             output_scores=True,
#         )
#
#         responses = self.tokenizer.batch_decode(
#             outputs.sequences[:, input_ids.shape[-1] :], skip_special_tokens=True
#         )
#         perplexity = calculate_perplexity(outputs, input_ids, self.terminators)
#         r_perplexity = []
#         for r in responses:
#             r_formated_input, r_response = input_formater.reverse_sample(text, r)
#             input_ids = self.tokenizer.apply_chat_template(
#                 r_formated_input, add_generation_prompt=True, return_tensors="pt"
#             ).to(self.model.device)
#             full_ids = self.tokenizer.apply_chat_template(
#                 r_formated_input + r_response,
#                 add_generation_prompt=True,
#                 return_tensors="pt",
#             ).to(self.model.device)
#             target_ids = full_ids.clone()
#             target_ids[:, : input_ids.shape[-1]] = -100
#             outputs = self.model(full_ids, labels=target_ids)
#             neg_log_likelihood = outputs.loss
#             r_perplexity.append(torch.exp(neg_log_likelihood).item())
#
#         paraphrases = [r.split("\n")[-1] for r in responses]
#         return Output(
#             text=text,
#             paraphrases=paraphrases,
#             perplexity=np.array(perplexity),
#             reverse_ppls=np.array(r_perplexity),
#             model=self.model_name,
#         )
