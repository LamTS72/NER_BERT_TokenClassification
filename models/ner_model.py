from transformers import (
    AutoModelForTokenClassification
)

class CustomModel():
    def __init__(self, model_name, id2label, label2id,
                 flag_training):
        self.model_name = model_name
        self.id2label = id2label
        self.label2id = label2id
        self.model = self.create_model()
        if flag_training:
            print("-"*50, "Information of Model", "-"*50)
            print(self.model)
            print("Parameters: ", int(self.model.num_parameters() / 1000000),  "M")
            print("-"*50, "Information of Model", "-"*50)
    def create_model(self):
        return AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            id2label=self.id2label,
            label2id=self.label2id
        )