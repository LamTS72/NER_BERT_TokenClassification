import token
import safetensors
from transformers import (
    AutoModelForTokenClassification
)
from models.preprocessing import Preprocessing
from configs.config import ConfigModel

import torch
class Predictor():
    def __init__(self):
        self.process = Preprocessing(flag_predict=True)
        self.model = self.load_model()
        self.tokenizer = self.process.tokenizer
        self.name_tags = ConfigModel.NAME_TAGS
    
    def load_model(self):
        model = AutoModelForTokenClassification.from_pretrained(
            "./token_classifier_scratch",
            use_safetensors=True,
        )
        return model
    
    def tokenized_sample(self, sample):
        inputs = self.tokenizer(
            sample,
            truncation=True, 
            is_split_into_words=True,
            padding=True,
            return_tensors="pt"
        )
        return inputs
        
    def predict(self, sample):
        self.model.eval()
        input = self.tokenized_sample([sample])
        with torch.no_grad():
            output = self.model(**input)
        logits = output.logits
        predictions = torch.argmax(logits, dim=-1)[:, 1:-1]
        predictions = [
            [self.name_tags[p] for p in prediction if p != -100]
            for prediction in predictions
        ]
        return predictions
        
if __name__ == '__main__':
    predict = Predictor()
    print(predict.model)
    print(predict.predict("EU rejects German call to boycott British lamb ."))
