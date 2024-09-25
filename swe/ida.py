# ida.py
import torch
class InnerDialogueAgent:
    def __init__(self, tokenizer, max_prompt_length=512):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def generate_prompt(self, user_task, previous_response):
        """
        Generate a new prompt based on the user task and previous response.
        """
        prompt = f"User Task: {user_task}\nPrevious Response: {previous_response}\nPlease provide a refined response."
        encoded_prompt = self.tokenizer.encode(prompt, truncation=True, max_length=self.max_prompt_length)
        return torch.tensor([encoded_prompt])
