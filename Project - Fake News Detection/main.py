import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from models.LSTM import LSTMClassifier
from utils.configuration import Config
TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()


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
        self.criterion = nn.NLLLoss(reduction='none')
        self.results = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sgd_spv_matrix = {}
        self.eps = config.eps
        self.per_sample_prediction = None
        if torch.cuda.is_available():
            self.model.to(self.device)

    def train_model(self, train_iter, epoch):
        total_epoch_loss, total_epoch_acc, steps = 0, 0, 0
        self.model.train()
        for idx, batch in enumerate(train_iter):
            text = batch.text[0]
            target = batch.label
            target = torch.autograd.Variable(target).long()
            text = text.to(self.device)
            target = target.to(self.device)
            if (text.size()[0] is not self.batch_size):  # One of the batch returned by BucketIterator has length different than 32.
                continue
            self.optimizer.zero_grad()
            prediction = self.model(text)
            loss = self.criterion(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
            acc = 100.0 * num_corrects / len(batch)
            loss.backward()
            clip_gradient(model, 1e-1)
            self.optimizer.step()
            steps += 1

            if steps % 100 == 0:
                print(
                    f'Epoch: {epoch + 1}, Idx: {idx + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {acc.item(): .2f}%')

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

        return total_epoch_loss / len(train_iter), total_epoch_acc / len(train_iter)


    def eval_model(self, val_iter):
        total_epoch_loss, total_epoch_acc = 0, 0
        self.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(val_iter):
                text = batch.text[0]
                if (text.size()[0] is not 32):
                    continue
                target = batch.label
                target = torch.autograd.Variable(target).long()
                text = text.to(self.device)
                target = target.to(self.device)
                prediction = self.model(text)
                loss = self.criterion(prediction, target)
                num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
                acc = 100.0 * num_corrects / len(batch)
                total_epoch_loss += loss.item()
                total_epoch_acc += acc.item()

        return total_epoch_loss / len(val_iter), total_epoch_acc / len(val_iter)

ROOT_PATH = '/BiLSTM/'
MODEL_WEIGHTS_DIR = 'model_weights'
GRAPHS_FOLDER_NAME = 'graphs'
PER_SAMPLE_RESULTS_DIR = 'per_samples_results'
model_weights_dir = f"{ROOT_PATH}{MODEL_WEIGHTS_DIR}"
graphs_dir = f"{ROOT_PATH}{GRAPHS_FOLDER_NAME}"
sample_results_dir = f"{ROOT_PATH}{PER_SAMPLE_RESULTS_DIR}"
SAVE_FIGS = True
BATCH_SIZE = 32
learning_rate = 2e-5
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300

def get_base_config():
  ####################################################################
  # model consistency options
  SAVE_TO_CHECKPOINTS = True # if ture, saves model.name_epcoch file into the weights folder
  LOAD_CHECKPOINTS = True # # if ture, every epoch tries to load pretrained weights
  ####################################################################
  # if needed, can be modified to upload the 'best model'
  return Config(lr=2e-5,
                epochs=100,
                eps=0.00001,
                step_size=2,
                gamma=0.001,
                weight_decay=5e-4,
                dropout_std_n_times=15,
                momentum=0.9,
                milestones=[150],
                save_points=[100, 150, 170],
                save_model=SAVE_TO_CHECKPOINTS,
                upload_model=LOAD_CHECKPOINTS,
                model_weights_path=model_weights_dir,
                batch_size=BATCH_SIZE)

model = LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
train = (model)

for epoch in range(10):
    train_loss, train_acc = train.train_model(train_iter, epoch)
    val_loss, val_acc = train.eval_model(valid_iter)

    print(
        f'Epoch: {epoch + 1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

# test_loss, test_acc = eval_model(model, test_iter)
# print(f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')
#
# ''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
# test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
# test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."
#
# test_sen1 = TEXT.preprocess(test_sen1)
# test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]
#
# test_sen2 = TEXT.preprocess(test_sen2)
# test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]
#
# test_sen = np.asarray(test_sen1)
# test_sen = torch.LongTensor(test_sen)
# test_tensor = Variable(test_sen, volatile=True)
# test_tensor = test_tensor.cuda()
# model.eval()
# output = model(test_tensor, 1)
# out = F.softmax(output, 1)
# if (torch.argmax(out[0]) == 1):
#     print("Sentiment: Positive")
# else:
#     print("Sentiment: Negative")