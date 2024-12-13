from transformers import (
    TrainingArguments,
    Trainer,
    get_scheduler,
    AutoModelForTokenClassification
)
import evaluate
import torch
import os
from models.preprocessing import Preprocessing
from data.data_training import CustomDataset
from configs.config import ConfigModel, ConfigHelper
import numpy as np
import evaluate
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import Repository, HfApi, HfFolder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Used Device: ", device)

class Training():
    def __init__(self, model_name=ConfigModel.MODEL_NAME, 
                 learning_rate=ConfigModel.LEARNING_RATE, 
                 epoch=ConfigModel.EPOCHS, 
                 num_warmup_steps=ConfigModel.NUM_WARMUP_STEPS, 
                 name_metric=ConfigModel.METRICs, 
                 path_tensorboard=ConfigModel.PATH_TENSORBOARD, 
                 path_save=ConfigModel.PATH_SAVE
                ):
        self.dataset = CustomDataset()
        self.process = Preprocessing(dataset=self.dataset)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            id2label=self.process.id2label,
            label2id=self.process.label2id
        )
        print("-"*50, "Information of Model", "-"*50)
        print(self.model)
        print("Parameters: ", int(self.model.num_parameters() / 1000000),  "M")
        print("-"*50, "Information of Model", "-"*50)
        self.epochs = epoch
        self.num_steps = self.epochs * self.process.step_train_loader
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate
        )
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_steps
        )
        self.metric = evaluate.load(name_metric)
        self.writer = SummaryWriter(path_tensorboard)
        
        # Define necessary variables
        self.api = HfApi(token=ConfigHelper.TOKEN_HF)
        self.repo_name = path_save  # Replace with your repo name
        self.author = ConfigHelper.AUTHOR
        self.repo_id = self.author + "/" + self.repo_name
        self.token = HfFolder.get_token()
        self.repo = self.setup_hf_repo(self.repo_name, self.repo_id, self.token)
        
    def setup_hf_repo(self, local_dir, repo_id, token):
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        
        try:
            self.api.repo_info(repo_id)
            print(f"Repository {repo_id} exists. Cloning...")
        except Exception as e:
            print(f"Repository {repo_id} does not exist. Creating...")
            self.api.create_repo(repo_id=repo_id, token=token, private=True)
        
        repo = Repository(local_dir=local_dir, clone_from=repo_id)
        return repo
    
    def save_and_upload(self, epoch, final_commit=False):
        # Save model, tokenizer, and additional files
        self.model.save_pretrained(self.repo_name)
        self.process.tokenizer.save_pretrained(self.repo_name)

        # Push to Hugging Face Hub
        self.repo.git_add(pattern=".")
        commit_message = "Final Commit: Complete fine-tuned model" if final_commit else f"Epoch {epoch}: Update fine-tuned model and metrics"
        self.repo.git_commit(commit_message)
        self.repo.git_push()

        print(f"Model and files pushed to Hugging Face Hub for epoch {epoch}: {self.repo_id}")
    
    def compute_metrics(self, eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        
        # Xoá token đặc biệt và chuyển chúng về name tags
        true_labels = [[self.dataset.name_tags[l] for l in label if l != -100]for label in labels]
        true_predictions = [
            [self.dataset.name_tags[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = self.metric.compute(
            predictions=true_predictions,
            references=true_labels
        )
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }
    
    def postprocess(self, predictions, labels):
        predictions = predictions.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        # Xoá token đặc biệt và chuyển chúng về name tags
        true_labels = [[self.dataset.name_tags[l] for l in label if l != -100]for label in labels]
        true_predictions = [
            [self.dataset.name_tags[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        return true_predictions, true_labels
    
    def fit(self, flag_step=False):
        progress_bar = tqdm(range(self.num_steps))
        interval = 200
        for epoch in range(self.epochs):
            # training
            self.model.train()
            n_train_samples = 0
            total_train_loss = 0
            for i, batch in enumerate(self.process.train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                n_train_samples += len(batch)
                outputs = self.model.to(device)(**batch)
                losses = outputs.loss
                losses.backward()
                
                total_train_loss += round(losses.item(),4)
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)
                if (i + 1) % interval == 0 and flag_step == True:
                    print("Epoch: {}/{}, Iteration: {}/{}, Train Loss: {}".format(
                        epoch + 1,
                        self.epochs,
                        i + 1,
                        self.process.step_train_loader,
                        losses.item())
                    )
                    self.writer.add_scalar('Train/Loss', round(losses.item(),4), epoch * self.process.step_train_loader + i)
            
            # evaluate
            self.model.eval()
            n_val_samples = 0
            total_val_loss = 0
            for i, batch in enumerate(self.process.val_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                n_val_samples += len(batch)
                with torch.no_grad():
                    outputs = self.model.to(device)(**batch)
                logits = outputs.logits
                losses = outputs.loss
                predictions = torch.argmax(logits, dim=-1)
                
                total_val_loss += round(losses.item(),4)
                
                labels = batch["labels"]
                true_predictions, true_labels = self.postprocess(predictions, labels)
                self.metric.add_batch(predictions=true_predictions, references=true_labels)
                if (i + 1) % interval == 0 and flag_step == True:
                    print("Epoch: {}/{}, Iteration: {}/{}, Val Loss: {}".format(
                        epoch + 1,
                        self.epochs,
                        i + 1,
                        self.process.step_val_loader,
                        losses.item())
                    )
                    self.writer.add_scalar('Val/Loss', round(losses.item(),4), epoch * self.process.step_val_loader + i)         
            
            epoch_train_loss = total_train_loss / n_train_samples
            epoch_val_loss = total_val_loss / n_val_samples
            print(f"train_loss: {epoch_train_loss}  - val_loss: {epoch_val_loss} -")
    
            metrics = self.metric.compute()
            print(
                f"epoch {epoch+1}:",
                {
                    key: metrics[f"overall_{key}"]
                    for key in ["precision", "recall", "f1", "accuracy"]
                },
            )
            # Save and upload after each epoch
            final_commit = ((epoch+1) == self.epochs)
            self.save_and_upload((epoch+1), final_commit)
    
    def test(self):
        self.model.eval()
        for i, batch in enumerate(self.process.test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():   
                outputs = self.model.to(device)(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            labels = batch["labels"]
            true_predictions, true_labels = self.postprocess(predictions, labels)
            self.metric.add_batch(predictions=true_predictions, references=true_labels)
            metrics = self.metric.compute()
            print(
                f"Result Test:",
                {
                    key: metrics[f"overall_{key}"]
                    for key in ["precision", "recall", "f1", "accuracy"]
                },
            )       
        
                
if __name__ == '__main__':
    train = Training()
    train.fit()