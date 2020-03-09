import os
import time
import torch.nn.functional as F
import torch
from models.BiLstm import BiLstmModel
from models.BiLstmMultiInputs import MultipleInputModel
import numpy as np
from torch import nn, optim
from utils.configuration import get_base_config
from utils.data_utils import load_dataset
from pathlib import Path
import pandas as pd
import glob
from utils.EarlyStopping import EarlyStopping

os.chdir(os.getcwd())

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.model_name = config.model_name
        self.seed = config.seed
        self.lr = config.lr
        self.epochs = config.epochs
        self.save_model = config.save_model
        self.upload_model = config.upload_model
        self.model_weights_path = config.model_weights_path
        self.batch_size = config.batch_size
        self.momentum = config.momentum
        self.milestones = config.milestones
        self.gamma = config.gamma
        self.save_points = config.save_points
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=config.weight_decay,
                                   momentum=self.momentum)
        #optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        self.criterion = torch.nn.BCELoss(reduction='none')
        # initialize the early_stopping object
        self.early_stopping = EarlyStopping(patience=config.patience, verbose=True)
        self.results = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sgd_spv_matrix = {}
        self.eps = config.eps
        if torch.cuda.is_available():
            self.model.to(self.device)

    def calc_accuracy(self, log_ps, labels):
        self.model.eval()
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        acc = torch.mean(equals.type(torch.FloatTensor))
        self.model.train()
        return acc

    def save_checkpoint(self, measure, epoch):
        weights_path_model = Path(f"{self.model_weights_path}/{self.model_name}_{epoch}_seed_{self.seed}.pth")
        if (epoch in self.save_points) and (not weights_path_model.exists()) and self.save_model:
            print('saving_model: ')
            torch.save({'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(), 'loss': measure}, weights_path_model)

    def load_checkpoint(self, weights_path, epoch):
        checkpoint = torch.load(weights_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.to(self.device)
        loss = checkpoint['loss']
        print(f"Uploaded weights succesfuly at epoch number {epoch}")
        return loss

    def record(self, epoch, **kwargs):
        epoch = "{:02d}".format(epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in kwargs.items():
            key = f"{self.model_name}_{key}"
            if not self.results.get(key):
                self.results[key] = []
            self.results[key].append(value)
            val = '{:.2f}'.format(np.round(value, 2))
            temp += f"{key} : {val}      |       "
        print(temp)

    def fit(self, trainloader, validloader, config):
        for epoch in range(1, self.epochs + 1):
            weights_path = Path(f"{self.model_weights_path}/{config.model_name}_{epoch}.pth")
            if weights_path.exists() and self.upload_model:
                epoch_train_loss = self.load_checkpoint(weights_path, epoch)
            else:
                epoch_train_loss, epoch_train_acc = self.train_model(trainloader, epoch)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones,
                                                            gamma=self.gamma, last_epoch=epoch)
            epoch_valid_loss, epoch_valid__acc = self.eval_model(validloader, epoch)
            self.record(epoch, train_loss=epoch_train_loss, validation_loss=epoch_valid_loss)
            self.save_checkpoint(weights_path, epoch_train_acc)
            self.scheduler.step()
            self.early_stopping(epoch_valid_loss)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def clip_gradient(self, clip_value):
        params = list(filter(lambda p: p.grad is not None, self.model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)

    def train_model(self, train_iter, epoch):
        total_epoch_loss, total_epoch_acc, steps = 0, 0, 0
        self.model.train()
        for idx, batch in enumerate(train_iter):
            # retrieve text and no. of words
            text, text_lengths = batch.text
            target = batch.label
            target = torch.autograd.Variable(target).long()
            text = text.to(self.device)
            target = target.to(self.device)
            # if (text.size()[0] is not self.batch_size):  # One of the batch returned by BucketIterator has length different than 32.
            #     continue
            self.optimizer.zero_grad()
            prediction = self.model(text,text_lengths)
            loss = self.criterion(prediction, target)

            # calculate accuracy
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
            acc = 100.0 * num_corrects / len(batch)

            loss.backward()
            self.clip_gradient(1e-1)
            self.optimizer.step()

            steps += 1
            if steps % 100 == 0:
                print(
                    f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

        return total_epoch_loss / len(train_iter), total_epoch_acc / len(train_iter)


    def eval_model(self, val_iter, epoch):
        total_epoch_loss, total_epoch_acc = 0, 0
        self.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(val_iter):
                text, text_lengths = batch.text
                # if (text.size()[0] is not self.batch_size):
                #     continue
                target = batch.label
                target = torch.autograd.Variable(target).long()
                text = text.to(self.device)
                target = target.to(self.device)
                prediction = self.model(text, text_lengths)
                loss = self.criterion(prediction, target)

                # calculate accuracy
                num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
                acc = 100.0 * num_corrects / len(batch)

                total_epoch_loss += loss.item()
                total_epoch_acc += acc.item()

        return total_epoch_loss / len(val_iter), total_epoch_acc / len(val_iter)

config = get_base_config()
path_data = Path(f"{config.ROOT_PATH_DATA}combined_data.csv")
extension = 'csv'
filenames = [i for i in glob.glob(f"{config.ROOT_PATH_DATA}*.{extension}")]
if not path_data.exists():
    combined_csv = pd.concat([pd.read_csv(f, usecols=["text", "class"]) for f in filenames])
    combined_csv.to_csv("combined_data.csv", index=False)

TEXT, vocab_size, word_embeddings, train_iter, valid_iter =\
    load_dataset(f"{config.ROOT_PATH_DATA}combined_data.csv", config.embedding_dim,
                 config.seq_max_len, config.seed)

exp_name = "BiLSTM"
config.add_attributes(model_name=exp_name)
# TODO:  replace BiLstmModel with MultipleInputModel
model_bilstm = BiLstmModel(config, word_embeddings.cuda())
#model_multipleInput = MultipleInputModel(config, word_embeddings.cuda())
trainer = Trainer(model_bilstm, config)
trainer.fit(train_iter, valid_iter, config)