# llma.py
import torch

class LLMAgent:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def generate_response(self, prompt_ids):
        """
        Generate a response based on the prompt.
        """
        prompt_ids = prompt_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=prompt_ids,
                max_length=prompt_ids.size(1) + 100,  # Extend by 100 tokens
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
