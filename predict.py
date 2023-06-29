import torch
from cog import BasePredictor, Input
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda"
MODEL_NAME = 'lmsys/vicuna-33b-v1.3'
MODEL_CACHE = "cache"

class Predictor(BasePredictor):
    def setup(self):
        """Load the model and tokenizer into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE,
            local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE,
            local_files_only=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.eval()
        self.model.to(device, dtype=torch.bfloat16)

    def predict(self,
        prompt: str = Input(description="Instruction for the model"),
        max_new_tokens: int = Input(description="max tokens to generate", default=64),
        temperature: float = Input(description="0.01 to 1.0 temperature", default=0.75),
    ) -> str:    
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        new_tokens = output_ids[0, len(input_ids[0]) :]
        output = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return output
