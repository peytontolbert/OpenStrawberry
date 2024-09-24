import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class SwoDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        Initializes the dataset by loading the data from the specified path.

        Args:
            data_path (str): Path to the dataset file.
            tokenizer (PreTrainedTokenizer): Tokenizer for encoding the text.
            max_length (int): Maximum sequence length.
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_data(self, data_path):
        # Implement data loading logic (e.g., JSON, CSV)
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_request = self.data[idx]['user_request']
        agent_action = self.data[idx]['agent_action']
        input_text = f"User: {user_request}\nAgent: {agent_action}"
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask

def load_dataset(data_path, tokenizer, batch_size=32, max_length=512, shuffle=True):
    """
    Loads the dataset and returns DataLoader objects for training and validation.

    Args:
        data_path (str): Path to the dataset file.
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding the text.
        batch_size (int): Batch size.
        max_length (int): Maximum sequence length.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Training DataLoader.
        DataLoader: Validation DataLoader.
    """
    dataset = SwoDataset(data_path, tokenizer, max_length)
    # Split into training and validation (e.g., 80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader