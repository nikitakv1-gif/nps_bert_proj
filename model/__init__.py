from model.ThreeInputModel import ThreeInputModel
from model.dataset import ReviewsDataset
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

base_path = os.path.dirname(__file__)  # Путь до текущего файла model/__init__.py

path_to_model = os.path.join(base_path, 'presave', 'model.pt')
path_to_tokenizer = os.path.join(base_path, 'presave')

def model():
    model = ThreeInputModel('cointegrated/rubert-tiny2', num_labels=5)
    model.load_state_dict(torch.load(path_to_model, map_location = 'cpu'))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(path_to_tokenizer)
    model.eval()
    emotion_model = "blanchefort/rubert-base-cased-sentiment"
    emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model)
    model_sent = AutoModelForSequenceClassification.from_pretrained(emotion_model)

    return model, tokenizer, model_sent, emotion_tokenizer
