from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification
)
from data.data_training import CustomDataset
from torch.utils.data import DataLoader

class Preprocessing():
    def __init__(self, model_tokenizer="bert-base-cased", batch_size=8, dataset=None, flag_predict=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_tokenizer)
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        if flag_predict == False:
            dataset = CustomDataset()
            print("-"*50, "Information of Tokenizer", "-"*50)
            print(self.tokenizer)
            print("-"*50, "Information of Tokenizer", "-"*50)
            self.id2label, self.label2id = self.hashmap_id_label(dataset=dataset)
            self.tokenized_train_set, self.tokenized_test_set, self.tokenized_val_set = self.map_tokenize_dataset(dataset=dataset)
            self.train_loader, self.test_loader, self.val_loader = self.data_loader(batch_size=batch_size)
            self.step_train_loader, self.step_test_loader, self.step_val_loader = len(self.train_loader), len(self.test_loader), len(self.val_loader)
    
    def align_labels_from_tokens(self, name_tags, word_ids):
        """After Tokenizer the length of labels is changed,
        preprocess the labels to new labels
        Args:
            name_tags (list): list of name tags [O, B-xxx, I-xxx]
            word_ids (list): position of tokens

        Returns:
            new labels: list of new labels
        """
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # start new token
                current_word = word_id
                label = -100 if word_id is None else name_tags[word_id]
                new_labels.append(label)
            elif word_id == None:
                # special token
                new_labels.append(-100)
            else:
                # word_id same previous word_id
                label = name_tags[word_id]
                # Nếu word_id giống cái trước đó => B-xxx convert to I-xxx 
                # Do token bị tách không có nghĩa luôn được gán B-xxx do cùng word_id với trước đó
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)
        return new_labels
            
    def tokenize_with_align_labels(self, sample):
        tokenized_inputs = self.tokenizer(
            sample["tokens"], 
            truncation=True, 
            is_split_into_words=True
        )
        all_labels = sample["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.align_labels_from_tokens(labels, word_ids))
            
        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs
    
    def map_tokenize_dataset(self, dataset):
        tokenized_train_set = dataset.train_set.map(
            self.tokenize_with_align_labels,
            batched=True,
            remove_columns=dataset.train_set.column_names
        )
        tokenized_test_set = dataset.test_set.map(
            self.tokenize_with_align_labels,
            batched=True,
            remove_columns=dataset.test_set.column_names
        )
        tokenized_val_set = dataset.val_set.map(
            self.tokenize_with_align_labels,
            batched=True,
            remove_columns=dataset.val_set.column_names
        )
        return tokenized_train_set, tokenized_test_set, tokenized_val_set
        
    def hashmap_id_label(self, dataset):
        id2label = {i: label for i, label in enumerate(dataset.name_tags)}
        label2id = {label: i for i, label in id2label.items()}
        return id2label, label2id
    
    def data_loader(self, batch_size):
        train_loader = DataLoader(
            self.tokenized_train_set,
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=batch_size
        )
        
        val_loader = DataLoader(
            self.tokenized_val_set,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=batch_size
        )
        
        test_loader = DataLoader(
            self.tokenized_test_set,
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=batch_size
        )
        return train_loader, test_loader, val_loader
        
if __name__ == "__main__":
    proc = Preprocessing()
    # sample = proc[0]
    # # Sample after Tokenizer Processing
    # print(sample)
    # # Split into tokens and add special tokens
    # print("List Tokens: ", sample.tokens())
    # # Encode tokens into position of tokens
    # print("List WordID: ", sample.word_ids())
    # print(proc.tokenized_train_set)
    # print(proc.tokenized_test_set)
    # print(proc.tokenized_val_set)
