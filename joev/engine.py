import torch
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from typing import List, Optional

class Engine:
    def __init__(self, model_name: str, quantization: Optional[str] = None):
        self.model_name = model_name
        self.quantization = quantization
        self.model = None
        self.tokenizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.precision = torch.bfloat16
        self.last_logits = None

    def load(self, checkpoint_path: Path):
        from model import Transformer  # Assuming this is the correct import

        tokenizer_path = checkpoint_path.parent / "tokenizer.model"
        assert tokenizer_path.is_file(), tokenizer_path

        with torch.device('meta'):
            self.model = Transformer.from_name(self.model_name)

        if self.quantization == "int8":
            from quantize import WeightOnlyInt8QuantHandler
            simple_quantizer = WeightOnlyInt8QuantHandler(self.model)
            self.model = simple_quantizer.convert_for_runtime()
        elif self.quantization == "int4":
            from quantize import WeightOnlyInt4QuantHandler
            simple_quantizer = WeightOnlyInt4QuantHandler(self.model, groupsize=128)
            self.model = simple_quantizer.convert_for_runtime()

        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        self.model.load_state_dict(checkpoint, assign=True)
        
        # Move the entire model to the specified device and dtype
        self.model = self.model.to(device=self.device, dtype=self.precision).eval()

        self.tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))

    def prefill(self, prompt: str) -> List[int]:
        assert self.model is not None, "Model must be loaded before prefill"
        
        tokens = self.tokenizer.encode(prompt)
        tokens = [self.tokenizer.bos_id()] + tokens
        input_ids = torch.tensor(tokens, dtype=torch.int, device=self.device)
        
        with torch.device(self.device):
            self.model.setup_caches(max_batch_size=1, max_seq_length=len(tokens))

        with torch.no_grad():
            input_pos = torch.arange(len(tokens), device=self.device)
            logits = self.model(input_ids.unsqueeze(0), input_pos)
        
        self.last_logits = logits
        return tokens

    def forward(self, input_ids: List[int]) -> torch.Tensor:
        assert self.model is not None, "Model must be loaded before forward pass"
        
        input_ids = torch.tensor(input_ids, dtype=torch.int, device=self.device).unsqueeze(0)
        #input_pos = torch.tensor([len(input_ids[0]) - 1], device=self.device)
        input_pos = torch.arange(len(input_ids[0]), device=self.device)
        
        with torch.device(self.device):
            self.model.setup_caches(max_batch_size=1, max_seq_length=len(input_ids[0]))

        with torch.no_grad():
            logits = self.model(input_ids, input_pos)
        
        self.last_logits = logits
        return logits

    def sample(self) -> int:
        assert self.last_logits is not None, "No logits available for sampling"
        
        next_token = torch.argmax(self.last_logits[0, -1]).item()
        return next_token

    def unload(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None
        self.last_logits = None