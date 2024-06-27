import gc

import torch
import torch.nn.functional as F
from tqdm.autonotebook import trange
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel)

from TAGLAS.utils.gpu import get_available_devices
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
LLM_DIM_DICT = {"ST": 768, "BERT": 768, "e5": 1024, "llama2_7b": 4096, "llama2_13b": 5120}


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-10)


class LLMModel(torch.nn.Module):
    """
    Large language model from transformers.
    """

    def __init__(self, llm_name: str, cache_dir: str = "./model_data/model", max_length: int = 500):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        self.llm_name = llm_name
        self.indim = LLM_DIM_DICT[self.llm_name]
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.model, self.tokenizer = self.get_llm_model()
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = 'right'

    def get_llm_model(self):
        if self.llm_name == "llama2_7b":
            model_name = "meta-llama/Llama-2-7b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer

        elif self.llm_name == "llama2_13b":
            model_name = "meta-llama/Llama-2-13b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer

        elif self.llm_name == "e5":
            model_name = "intfloat/e5-large-v2"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "BERT":
            model_name = "bert-base-uncased"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "ST":
            model_name = "sentence-transformers/multi-qa-distilbert-cos-v1"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        else:
            raise ValueError(f"Unknown language model: {self.llm_name}.")
        model = ModelClass.from_pretrained(model_name, cache_dir=self.cache_dir)
        tokenizer = TokenizerClass.from_pretrained(model_name, cache_dir=self.cache_dir, add_eos_token=True,
                                                   pading_side="left")
        if self.llm_name[:6] == "llama2":
            tokenizer.pad_token = tokenizer.bos_token
        return model, tokenizer

    def pooling(self, outputs, text_tokens=None):
        return F.normalize(mean_pooling(outputs, text_tokens["attention_mask"]), p=2, dim=1)

    def forward(self, text_tokens):
        outputs = self.model(input_ids=text_tokens["input_ids"],
                             attention_mask=text_tokens["attention_mask"],
                             output_hidden_states=True,
                             return_dict=True)["hidden_states"][-1]

        return self.pooling(outputs, text_tokens)

    def encode(self, text_tokens, pooling=False):

        with torch.no_grad():
            outputs = self.model(input_ids=text_tokens["input_ids"],
                                 attention_mask=text_tokens["attention_mask"],
                                 output_hidden_states=True,
                                 return_dict=True)["hidden_states"][-1]
            outputs = outputs.to(torch.float32)
            if pooling:
                outputs = self.pooling(outputs, text_tokens)

            return outputs, text_tokens["attention_mask"]


class SentenceEncoder:
    r"""Sentence encoder that can convert the input text sentence to embedding (mean pooling) with the specified LLM model.
    Args:
        llm_name (str): Name of LLM model, choose from avaliable_model.
        cache_dir (str, optional): Cache directory for model.
        batch_size (int, optional): Batch size for inference.
    """
    available_model = list(LLM_DIM_DICT.keys())

    def __init__(
            self,
            llm_name: str,
            cache_dir: str = None,
            batch_size: int = 1):
        self.llm_name = llm_name
        self.cache_dir = cache_dir
        self.device, _ = get_available_devices()
        self.batch_size = batch_size
        self.model = None

    def get_model(self):
        if self.model is None:
            self.model = LLMModel(self.llm_name, cache_dir=self.cache_dir)
            self.model.to(self.device)

    def encode(self, texts, to_tensor=True):
        if self.model is None:
            self.get_model()

        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                text_tokens = self.model.tokenizer(sentences_batch, return_tensors="pt", padding="longest",
                                                   truncation=True,
                                                   max_length=500).to(self.device)
                embeddings, _ = self.model.encode(text_tokens, pooling=True)
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def flush_model(self):
        # delete llm from gpu to save GPU memory
        if self.model is not None:
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()
