from os import mkdir
from os.path import isdir
import numpy as np
import torch

class Config:
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  def __init__(self, **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)

  def add_attributes(self,**kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)

def get_base_config():
  ROOT_PATH = '/models/'
  MODEL_WEIGHTS_DIR = 'model_weights'
  #ROOT_PATH_DATA = r"Fake-news-detection-ny+guar+kaggle/data/DataSets/"
  ROOT_PATH_DATA = r"data/DataSets/"
  model_weights_dir = f"{ROOT_PATH}{MODEL_WEIGHTS_DIR}"
  GRAPHS_FOLDER_NAME = 'graphs'
  graphs_dir = f"{ROOT_PATH}{GRAPHS_FOLDER_NAME}"
  ####################################################################
  BATCH_SIZE = 128
  output_size = 2
  hidden_size = 50
  embedding_length = 100
  ####################################################################
  # model consistency options
  SAVE_FIGS = True
  SAVE_TO_CHECKPOINTS = True # if true, saves model.name_epcoch file into the weights folder
  LOAD_CHECKPOINTS = True # # if true, every epoch tries to load pretrained weights
  ####################################################################
  # if needed, can be modified to upload the 'best model'
  #learning_rate = 2e-5
  return Config(lr=0.0001,
                epochs=200,
                dropout=0.3,
                eps=0.00001,
                step_size=2,
                gamma=0.1,
                weight_decay=5e-4,
                momentum=0.9,
                seed=5,
                n_classes=2,
                hidden_size=hidden_size,
                hidden_layer=1,
                patience=5,
                seq_max_len=500,
                embedding_dim=embedding_length,
                milestones=[10],
                save_points=np.arange(1,5)*50,
                save_model=SAVE_TO_CHECKPOINTS,
                upload_model=LOAD_CHECKPOINTS,
                model_weights_path=model_weights_dir,
                ROOT_PATH_DATA=ROOT_PATH_DATA,
                batch_size=BATCH_SIZE,
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

def create_directories(l):
  for directory_path in l:
    if not (isdir(directory_path)):
      mkdir(directory_path)