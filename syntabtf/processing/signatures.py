import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import json
from typing import Dict, Optional



class TransformerEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, max_seq_length: Optional[int] = None,
                 model_args: Dict = {}, cache_dir: Optional[str] = None,
                 tokenizer_args: Dict = {}, 
                 do_lower_case: bool = True,
                 tokenizer_name_or_path : str = None,
                 decomposited_size: int = 32
                 ):
        super(TransformerEncoder, self).__init__()
        
        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self.sent_encoder = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path if tokenizer_name_or_path is not None else model_name_or_path, 
                                                       cache_dir=cache_dir, 
                                                       **tokenizer_args)
        
        pooling_size: int = self.get_word_embedding_dimension()//decomposited_size
        self.avgPolling = nn.AvgPool1d(pooling_size, stride=pooling_size) if pooling_size > 0 else None

        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case
        # No max_seq_length set. Try to infer from model
        if max_seq_length is None:
            self.max_seq_length = min(self.sent_encoder.config.max_position_embeddings, 
                                      self.tokenizer.model_max_length) if self.__is_model_max_length_avaible() else 32
        else:
            self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.sent_encoder.config.tokenizer_class = self.tokenizer.__class__.__name__

    def tokenize(self, sentences, device="cpu"):
        assert isinstance(sentences, str), "Input must be a string"
        sentences = [sentences.replace("_", " ").lower() if self.do_lower_case else sentences.replace("_", " ")]
        toks = self.tokenizer.batch_encode_plus(sentences, max_length=self.max_seq_length, padding='max_length', truncation=True)
        ids, mask = (torch.LongTensor(toks["input_ids"]).to(device), torch.LongTensor(toks["attention_mask"]).to(device))
        return {"input_ids": ids, "attention_mask": mask}

    def __repr__(self):
        return "Transformer({}) with Transformer model: {} ".format(self.get_config_dict(), self.sent_encoder.__class__.__name__)
    
    def __is_model_max_length_avaible(self):
        tmp = hasattr(self.sent_encoder, "config") and hasattr(self.sent_encoder.config, "max_position_embeddings")
        return  tmp and hasattr(self.tokenizer, "model_max_length")
    
    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}
        
    def get_word_embedding_dimension(self) -> int:
        return self.sent_encoder.config.hidden_size

    def encode(self, features, to_numpy=True, device="cpu"):
        if to_numpy:
            with torch.no_grad():
                pooler_output = self.sent_encoder(features['input_ids'], attention_mask=features['attention_mask']).pooler_output  # [bs, dim]
                if self.avgPolling is not None:
                    pooler_output = self.avgPolling(pooler_output.unsqueeze(dim=1))
                return pooler_output.squeeze().detach().cpu().numpy()
        else:
            pooler_output = self.sent_encoder(features['input_ids'], attention_mask=features['attention_mask']).pooler_output  # [bs, dim]
            if self.avgPolling is not None:
                    pooler_output = self.avgPolling(pooler_output.unsqueeze(dim=1))
            return pooler_output.squeeze()

    def encode_list(self, sentences_list):
        signature_dict = {} 
        for field in sentences_list:
            features = self.tokenize(field)
            embedding = self.encode(features, to_numpy=True)
            signature_dict[field] = embedding
        return signature_dict
    
    def encode_col_name(self, column_name):
        features = self.tokenize(column_name)
        embedding = self.encode(features, to_numpy=True)
        return embedding

    def forward(self, features, to_numpy=True):
        return self.encode(features, to_numpy)
