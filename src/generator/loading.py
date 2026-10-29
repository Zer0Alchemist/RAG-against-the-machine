import torch
from typing import List, Dict
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM)


class LoadingLLM():
    """
    this class handles loading a language model and its tokenizer,
    formatting chat messages, encoding prompts, generating tokens
    and decoding generated outputs
    """
    def __init__(self, model_name: str) -> None:
        """
        this function initializes the language model and tokenizer
        from a pretrained Hugging Face model

        args:
            model_name: str = name or path of the pretrained model

        return:
            None
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

    def build_template(self, message: List[Dict[str, str]],
                       thinking: bool = False) -> str:
        """
        this function applies the model's chat template to a list
        of messages and prepares the prompt for generation

        args:
            message: List[Dict[str, str]] = list of chat messages
            thinking: bool = enables or disables thinking mode

        return:
            string containing the formatted chat prompt
        """
        result = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking
        )
        assert isinstance(result, str)
        return result

    def encode(self, prompt: str) -> torch.Tensor:
        """
        this function tokenizes the formatted prompt and converts
        it into a tensor of token IDs on the model's device

        args:
            prompt: str = formatted text to encode

        return:
            torch.Tensor containing the encoded token IDs
        """
        result = self.tokenizer(
                prompt,
                return_tensors="pt").to(self.model.device)['input_ids']
        assert isinstance(result, torch.Tensor)
        return result

    def generate(self, encoded_list: torch.Tensor,
                 max_tokens: int = 200) -> torch.Tensor:
        """
        this function generates new token IDs using the loaded
        language model and the provided encoded prompt

        args:
            encoded_list: torch.Tensor = encoded input token IDs
            max_tokens: int = maximum number of new tokens to generate

        return:
            torch.Tensor containing the generated token sequence
        """
        result = self.model.generate(  # type: ignore[misc]
            encoded_list,
            max_new_tokens=max_tokens)

        if not isinstance(result, torch.Tensor):
            raise TypeError("Excpected a Tensor from generate()")

        return result

    def decode(self, generated_ids: torch.Tensor,
               skip_s_tokens: bool = True) -> str:
        """
        this function decodes generated token IDs into readable text
        and removes the first decoded character using indexing

        args:
            generated_ids: torch.Tensor = generated token IDs
            skip_s_tokens: bool = whether to skip special tokens

        return:
            string containing the first character of decoded text
        """
        tokens: List[List[int]] = [generated_ids.tolist()]
        result = self.tokenizer.decode(
            tokens,
            skip_special_tokens=skip_s_tokens)[0].strip("\n")
        assert isinstance(result, str)
        return result
