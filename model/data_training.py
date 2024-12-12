from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, path_dataset="eriktks/conll2003", revision=None):
        self.raw_data = load_dataset(path_dataset)
        self.name_tags = self.raw_data["train"].features["ner_tags"].feature.names
        self.num_classes = self.raw_data["train"].features["ner_tags"].feature.num_classes
        self.train_set = self.raw_data["train"]
        self.test_set = self.raw_data["test"]
        self.val_set = self.raw_data["validation"]
        self.size = len(self.train_set) + len(self.test_set) + len(self.val_set)
        print("-"*40, "Information of Dataset", "-"*40)
        print(self.raw_data)
        print("Labels tag name: ", self.name_tags)
        print("Number of tag name: ", self.num_classes)
        print("-"*40, "Information of Dataset", "-"*40)
        
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        data = self.train_set[index]["tokens"]
        target = self.train_set[index]["ner_tags"]
        return {
            "data_text": data,
            "target_text": target
        }
    
    def illustrate_sample(self, index):
        sample = self[index]
        words = sample["data_text"]
        labels = sample["target_text"]
        line1 = line2 = ""
        for word, label in zip(words, labels):
            name_tag = self.name_tags[label]
            max_length = max(len(name_tag), len(word))
            line1 += word + " "*(max_length - len(word) + 1)
            line2 += name_tag + " "*(max_length - len(name_tag) + 1)
        print("Example " + str(index) + ":\n" + line1 + "\n" + line2)
        
    
# if __name__ == "__main__":
#     data = CustomDataset()
#     a = data[0]
#     print(a)
#     print(a["target_text"][2])