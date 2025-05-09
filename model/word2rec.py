"""
作者：yueyue
日期：2024年02月13日
"""
import torch
from torch import nn
from model.TCN_oscale import TCN
from gensim.models import word2vec
import numpy as np
import os
class skipgarmModel(nn.Module):
    """Skip gram model of word2vec.

       Attributes:
           emb_size: Embedding size.
           emb_dimention: Embedding dimention, typically from 50 to 500.
           u_embedding: Embedding for center word.
           v_embedding: Embedding for neibor words.
       """
    def __init__(self,vocab_size,embed_size):
        super(skipgarmModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embeddings = nn.Linear(self.vocab_size, self.embed_size)
        self.proj = nn.Linear(self.embed_size,self.vocab_size)
    def forward(self, x):
        x = self.embeddings(x)
        x = torch.mean(x,dim=1)
        x = self.proj(x)
        return x.squeeze()
class linearModel(nn.Module):
    def __init__(self):
        super(linearModel, self).__init__()
        self.embeddings = nn.Linear(875,875)
        self.TCN =TCN(1,3)
    def forward(self, x):
        embedding_x = self.embeddings(x)
        x = self.TCN(embedding_x)
        return x,embedding_x
    def frozen_embeding(self):
        for param in self.embeddings.parameters():
            param.requires_grad = False
class word2vec:
    def __init__(self,model,epochs,train_dataloader,val_dataloader,criterion,optimizer,
                 device,model_dir,model_name,checkpoint_frequency,lr_scheduler):
        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_dir = model_dir
        self.model_name = model_name
        self.loss = {"train": [], "val": []}
        self.model.to(self.device)
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler=lr_scheduler
    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch()
            self._validate_epoch()
            print(
                "word2vec Epoch: {}/{}, Train Loss={:.5f}, Val Loss={:.5f}".format(
                    epoch + 1,
                    self.epochs,
                    self.loss["train"][-1],
                    self.loss["val"][-1],
                )
            )
            self.lr_scheduler.step()
            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)

    def _train_epoch(self):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)
            # print(inputs.shape,labels.shape)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.squeeze())
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())


        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss.append(loss.item())

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def _save_checkpoint(self, epoch):
        """Save model checkpoint to `self.model_dir` directory"""
        epoch_num = epoch + 1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num).zfill(3))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        """Save final model to `self.model_dir` directory"""
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    # def save_loss(self):
    #     """Save train/val loss as json file to `self.model_dir` directory"""
    #     loss_path = os.path.join(self.model_dir, "loss.json")
    #     with open(loss_path, "w") as fp:
    #         json.dump(self.loss, fp)
def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
    """
    Scheduler to linearly decrease learning rate,
    so thatlearning rate after the last epoch is 0.
    """
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    return lr_scheduler