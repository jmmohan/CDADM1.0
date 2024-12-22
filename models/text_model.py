# text_model.py
from transformers import BertForSequenceClassification, BertTokenizer

def get_text_model():
    """
    Load a pre-trained BERT model for text classification.
    
    Returns:
        model: Pre-trained BERT model.
        tokenizer: Tokenizer for BERT.
    """
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer