from models.args import ModelArgs
from typing import List
import pathlib
import transformers

class Tokenizer():
    """Tokenizer class."""
    def encode(self, 
               text: str,
               bos: bool = True,
               eos: bool = True) -> List[int]:
        raise NotImplementedError("Tokenizer must implement encode method")
    
    def decode(self,
               tokens: List[int]) -> str:
        raise NotImplementedError("Tokenizer must implement decode method")
    
    @property
    def vocab(self) -> dict:
        raise NotImplementedError("Tokenizer must implement vocab property")
    
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError("Tokenizer must implement vocab_size property")
        
    @property
    def special_ids(self) -> set:
        return set([self.bos_id, self.eos_id])
    

class TransformerTokenizer(Tokenizer):
    """Transformer tokenizer."""
    def __init__(self,
                 args: ModelArgs,
                 tokenizer_dir: str = None):
        
        self.args = args
        assert pathlib.Path(tokenizer_dir).exists(), f"Tokenizer directory {tokenizer_dir} does not exist."

        self.model = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)
        if args.bos_token_id is None:
            args.bos_token_id = self.model.bos_token_id
        else:
            self.bos_id = args.bos_token_id

        if args.eos_token_id is None:
            args.eos_token_id = self.model.eos_token_id
        else:
            self.eos_id = args.eos_token_id

        if args.pad_token_id is None:
            args.pad_token_id = self.model.pad_token_id
        else:
            self.pad_id = args.pad_token_id

    def encode(self, 
               text: str,
               bos: bool = True,
               eos: bool = True) -> List[int]:
        """Encode text."""
        tokens = self.model.encode(text, add_special_tokens=False)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self,
               tokens: List[int]) -> str:
        """Decode tokens."""
        return self.model.decode(tokens, skip_special_tokens=True)

    @property
    def vocab(self) -> dict:
        """Vocabulary."""
        return self.model.get_vocab()

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return len(self.vocab)

    @property
    def special_ids(self) -> set:
        """Special IDs."""
        return set([self.bos_id, self.eos_id] + list(self.model.all_special_ids))
    
    def save(self,
             tokenizer_dir: str):
        """Save tokenizer."""
        self.model.save_pretrained(tokenizer_dir)

      