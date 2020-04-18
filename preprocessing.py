# Imports
# Textual imports
from torch.utils.data import Dataset
from tokenizers import WordTokenizer
import nltk

# torch-y
import torch

class AFFRDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self):
        """Returns the number of items in the dataset"""
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        Returns the datapoint at index i as a tuple (sentence, label),
        where the sentence is tokenized.
        """
        encoded = self.tokenizer.encode(
            self.sentences[idx], add_special_tokens=True)
        return encoded

# Function that gives our training, validation and test data
def get_data():
    """
    Returns: 
        train_data      : Fold containing training data
        validation_data : Fold containing validation data
        test_data       : Fold containing test data
    """

    # Train your tokenizer.
    reader = nltk.corpus.BracketParseCorpusReader(".", "02-21.10way.clean")
    tree = reader.parsed_sents()[:]
    text = [" ".join(line.leaves()).lower() for line in tree]
    tokenizer = WordTokenizer(text, max_vocab_size=10000)
    EOS = tokenizer.encode('.')

    # Get output dict and remapping of file names to dataset names
    files = ["02-21.10way.clean", "22.auto.clean", "23.auto.clean"]
    data_dict = {}
    remap_dict = {"02-21.10way.clean":"train", 
                "22.auto.clean":"validation",
                "23.auto.clean":"test"}

    # Parse the data with nltk bracket parser
    for file in files:
        reader = nltk.corpus.BracketParseCorpusReader(".", file)
        tree = reader.parsed_sents()[:]
        
        # Assign a dataset dict to train, validation or test
        data_dict[remap_dict[file]] = [" ".join(line.leaves()).lower() for line in tree]

    train_data = AFFRDataset(data_dict['train'], tokenizer)
    validation_data = AFFRDataset(data_dict['validation'], tokenizer)
    test_data = AFFRDataset(data_dict['test'], tokenizer)

    return train_data, validation_data, test_data, tokenizer

# Function that performs padding on a batch of sentences
def padded_collate(batch):
    """
    Args:
        batch: Batch containing a predefined number of sentences
    
    Returns:
        padded_sents : The sentences now with padding
        padded_labels: The labels now with padding
        lengths      : List with all lengths 
    """
    sentences = [sentence[:-1] for sentence in batch]
    labels = [sentence[1:] for sentence in batch]

    lengths = [len(s) for s in sentences]
    max_length = max(lengths)
    # Pad each sentence with zeros to max_length
    padded_sents = [s + [0] * (max_length - len(s)) for s in sentences]
    padded_labels = [s + [0] * (max_length - len(s)) for s in labels]
    return torch.LongTensor(padded_sents), torch.LongTensor(padded_labels), lengths