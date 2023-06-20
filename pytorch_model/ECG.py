import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import os
import tensorflow as tf
import numpy as np
from sklearn import metrics
import torch
import csv
from sklearn import metrics
from matplotlib import pyplot as plt
import math

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sys
sys.path.append('../pre_processing_and_recover_network/')
import data_preprocessing
import importlib
importlib.reload(data_preprocessing)


class EmotionRec(pl.LightningModule):
  def __init__(self):
    super().__init__()
    dirname = "./"
    self.files=["test_stress.csv","test_arousal.csv","test_valence.csv",
    "train_stress.csv","train_arousal.csv","train_valence.csv"]
    self.output = os.path.join(os.path.dirname(dirname), 'light_output')
    self.cnn = nn.Sequential(
                            ## conv block 1
                              nn.Conv1d(in_channels = 1, out_channels = 32, kernel_size = 32, padding= 'same'),
                              nn.ReLU(),
                              nn.Conv1d(in_channels = 32, out_channels = 32, kernel_size = 32, padding= 'same'),
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size = 8, stride = 2 ),
                            ## conv block 2
                              nn.Conv1d(in_channels = 32, out_channels = 64, kernel_size = 16, padding= 'same'),
                              nn.ReLU(),
                              nn.Conv1d(in_channels = 64, out_channels = 64, kernel_size = 16, padding= 'same'),
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size = 8, stride = 2),
                            ## conv block 3
                              nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 8, padding= 'same'),
                              nn.ReLU(),
                              nn.Conv1d(in_channels = 128, out_channels = 128, kernel_size = 8, padding= 'same'),
                              nn.ReLU(),
                              nn.MaxPool1d(kernel_size =  635 ,stride = 1))

    self.stress = nn.Sequential(nn.Flatten(),
                                nn.Linear(128, 512),
                                nn.ReLU(),
                                nn.Linear(512, 512), 
                                nn.ReLU(),
                                nn.Linear(512, 3)
                                )

    self.arousal = nn.Sequential(nn.Flatten(),
                                nn.Linear(128, 512),
                                nn.ReLU(),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Linear(512,9)
                                )
                  
    self.valence = nn.Sequential(nn.Flatten(),
                                nn.Linear(128, 512),
                                nn.ReLU(),
                                nn.Linear(512, 512),
                                nn.ReLU(),
                                nn.Linear(512,9)
                                )
              


  
  def forward(self, tensor_dataset):
    x = tensor_dataset[0]
    pp = tensor_dataset[1][:,0]
    y_stress = tensor_dataset[1][:,1]
    y_arousal = tensor_dataset[1][:,2]
    y_valence = tensor_dataset[1][:,3]
    z = self.cnn(x)
    y_hat_stress = self.stress(z)
    y_hat_arousal = self.arousal(z)
    y_hat_valence = self.valence(z)
    y_hat_stress = np.argmax(y_hat_stress.cpu().detach().numpy(), axis = 1)
    y_hat_arousal = np.argmax(y_hat_arousal.cpu().detach().numpy(), axis = 1)
    y_hat_valence = np.argmax(y_hat_valence.cpu().detach().numpy(), axis = 1)
    return y_hat_stress,y_hat_arousal,y_hat_valence

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
    return optimizer

  def training_step(self, tensor_dataset):
    x = tensor_dataset[0]
    pp = tensor_dataset[1][:,0]
    y_stress = tensor_dataset[1][:,1]
    y_arousal = tensor_dataset[1][:,2]
    y_valence = tensor_dataset[1][:,3]
    #x,y = tensor_dataset.tensors
    #x_train = x_train.view(x_train.size(0), -1) don't know why they did it, needs check
    z = self.cnn(x)

    y_hat_stress = self.stress(z)
    y_hat_arousal = self.arousal(z)
    y_hat_valence = self.valence(z) 
    
    #log loss
    loss_stress = F.cross_entropy(y_hat_stress, y_stress)
    self.log('train_loss_stress', loss_stress)
    loss_arousal = F.cross_entropy(y_hat_arousal, y_arousal)
    self.log('train_loss_arousal',loss_arousal)
    loss_valence = F.cross_entropy(y_hat_valence, y_valence)
    self.log('train_loss_valence', loss_valence)



    #log acc
    y_hat_stress = np.argmax(y_hat_stress.cpu().detach().numpy(), axis = 1)
    acc_stress = metrics.accuracy_score(y_stress.cpu() , y_hat_stress)
    acc_stress = np.round(acc_stress, 4)
    stage = "train_stress"
    self.log(f"{stage}_acc", acc_stress, prog_bar=True)
    self.store_results("train_stress.csv",acc_stress,loss_stress)


    y_hat_arousal = np.argmax(y_hat_arousal.cpu().detach().numpy(), axis = 1)
    acc_arousal = metrics.accuracy_score(y_arousal.cpu() , y_hat_arousal)
    acc_arousal = np.round(acc_arousal, 4)
    stage = "train_arousal"
    self.log(f"{stage}_acc", acc_arousal, prog_bar=True)
    self.store_results("train_arousal.csv",acc_arousal,loss_arousal)

    y_hat_valence = np.argmax(y_hat_valence.cpu().detach().numpy(), axis = 1)
    acc_valence = metrics.accuracy_score(y_valence.cpu() , y_hat_valence)
    acc_valence = np.round(acc_valence, 4)
    stage = "train_valence"
    self.log(f"{stage}_acc", acc_valence, prog_bar=True)
    self.store_results("train_valence.csv",acc_valence,loss_valence)

    loss = loss_stress + loss_arousal + loss_valence
    return loss
    
  def validation_step(self, val_batch, batch_idx):
    self.evaluate(val_batch, "test")
    

  def evaluate(self, batch, stage=None):
    x = batch[0]
    pp = batch[1][:,0]
    y_stress = batch[1][:,1]
    y_arousal = batch[1][:,2]
    y_valence = batch[1][:,3]

    z = self.cnn(x)  

    y_hat_stress = self.stress(z)
    y_hat_arousal = self.arousal(z)
    y_hat_valence = self.valence(z) 

    loss_stress = F.cross_entropy(y_hat_stress, y_stress)
    loss_arousal = F.cross_entropy(y_hat_arousal, y_arousal)
    loss_valence = F.cross_entropy(y_hat_valence, y_valence)


    y_hat_stress = np.argmax(y_hat_stress.cpu().detach().numpy(), axis = 1)
    acc_stress = metrics.accuracy_score(y_stress.cpu() , y_hat_stress)
    acc_stress = np.round(acc_stress, 4)
    

    
 
    y_hat_arousal = np.argmax(y_hat_arousal.cpu().detach().numpy(), axis = 1)
    acc_arousal = metrics.accuracy_score(y_arousal.cpu() , y_hat_arousal)
    acc_arousal = np.round(acc_arousal, 4)

    y_hat_valence = np.argmax(y_hat_valence.cpu().detach().numpy(), axis = 1)
    acc_valence = metrics.accuracy_score(y_valence.cpu() , y_hat_valence)
    acc_valence = np.round(acc_valence, 4)

    if stage:
      #stress
      self.log(f"{stage}_loss_stress", loss_stress, prog_bar=True)
      self.log(f"{stage}_acc_stress", acc_stress, prog_bar=True)
      self.store_results("test_stress.csv",acc_stress,loss_stress)

      #arousal
      self.log(f"{stage}_loss_arousal", loss_arousal, prog_bar=True)
      self.log(f"{stage}_acc_arousal", acc_arousal, prog_bar=True)
      self.store_results("test_arousal.csv",acc_arousal,loss_arousal)

      #valence
      self.log(f"{stage}_loss_valence", loss_valence, prog_bar=True)
      self.log(f"{stage}_acc_valence", acc_valence, prog_bar=True)
      self.store_results("test_valence.csv",acc_valence,loss_valence)

      # follow after problemaic samples:

    wrong_label_stress=[]
    for i in range(len(y_hat_stress)):
      if(int(y_hat_stress[i])!=int(y_stress[i])):
        wrong_label_stress.append(int(pp[i]))
    file = os.path.join(self.output, "wrong_pp_stress.csv")
    with open(file, 'a', newline='') as csvfile:
        fieldnames = ['pp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'pp': wrong_label_stress})

    wrong_label_arousal=[]
    for i in range(len(y_hat_arousal)):
      if(int(y_hat_arousal[i])!=int(y_arousal[i])):
        wrong_label_arousal.append(int(pp[i]))   
    file = os.path.join(self.output, "wrong_pp_arousal.csv")
    with open(file, 'a', newline='') as csvfile:
        fieldnames = ['pp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'pp': wrong_label_arousal})
    
    wrong_label_valence=[]
    for i in range(len(y_hat_valence)):
      if(int(y_hat_valence[i])!=int(y_valence[i])):
        wrong_label_valence.append(int(pp[i]))   
    file = os.path.join(self.output, "wrong_pp_valence.csv")
    with open(file, 'a', newline='') as csvfile:
        fieldnames = ['pp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'pp': wrong_label_valence})
    
    


  def test_step(self, batch, batch_idx):
    self.evaluate(batch, "test")

  def store_results(self,csv_file_name,accuracy_value,loss_val):
    file = os.path.join(self.output, csv_file_name)
    with open(file, 'a', newline='') as csvfile:
        fieldnames = ['accuracy','loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'accuracy': accuracy_value, 'loss': loss_val})
  
  def declare_new_fold_csv(self, k):
        for i in self.files:
          file = os.path.join(self.output, i)
          with open(file, 'a', newline = '') as csvfile:
            csvfile.write("init fold number " + str(k))
            csvfile.write("\n")
  
  #extract only the valuse of the accuracy
  def csv_to_list(self,file):
    Path="./light_output/"+file
    my_list=[]
    with open(Path,'r') as csvfile:
      lines = csv.reader(csvfile, delimiter=',')
      my_list = []
      for row in lines:
        for e in row:
            # for skipping on tensor values (loss)
            # and for skipping on line between diffrent
            # folds that init by the line: "init fold number "+ k
            if((e[0]!='t')and(e[0]!='i')):
                my_list.append(float(e))
    return my_list

  # third input argument is relevant only for plot of kfold
  # plot only graphs of accuracy
  def plot_results(self, is_k_fold, k):
    if(is_k_fold == False):
      plt.figure(figsize = (10,15))
      index = 1
      for i in self.files:
        plt.subplot(2,3,index)
        res = self.csv_to_list(i)
        batches_per_epoch = int(math.ceil(len(res)/250))
        acc = []
        for j in range(250):
          tmp = np.mean(res[j*batches_per_epoch:(j+1)*batches_per_epoch])
          acc.append(tmp)
        plt.plot(acc)
        plt.ylim(0, 1.1)
        plt.title(i[:i.find('.')]+" accuracy")
        index += 1

  def get_loss_and_acc_from_csv(self,file):
    Path="./light_output/"+file
    loss=[]
    acc=[]
    with open(Path,'r') as csvfile:
      lines = csv.reader(csvfile, delimiter=',')
      for row in lines:
        for e in row:
            # for skipping on line between diffrent
            # folds that init by the line: "init fold number "+ k
            if((e[0]!='i')):
              if((e[0]!='t')):
                  acc.append(float(e))
              else:
                s = e[e.find('(')+1:e.find(',')]
                loss.append(float(s))
      fin_acc = []
      num_of_epochs = 250
      batches_per_epoch = int(len(acc)/num_of_epochs)
      for j in range(num_of_epochs):
          tmp = np.mean(acc[j*batches_per_epoch:(j+1)*batches_per_epoch])
          fin_acc.append(tmp)
          
      fin_loss = []
      batches_per_epoch = int(len(loss)/num_of_epochs)
      for j in range(num_of_epochs):
          tmp = np.mean(loss[j*batches_per_epoch:(j+1)*batches_per_epoch])
          fin_loss.append(tmp)
    return fin_acc, fin_loss


  # third input argument is relevant only for plot of kfold
  def plot_acc_and_loss(self, is_k_fold, k):
    if(is_k_fold == False):
      plt.figure(figsize = (10,15))
      for i in range(3):
        acc_test, loss_test = self.get_loss_and_acc_from_csv(self.files[i])
        acc_train, loss_train = self.get_loss_and_acc_from_csv(self.files[i+3])

        plt.subplot(3,2,2*i+1)
        plt.plot(acc_test,color='r',label=self.files[i][:self.files[i].find('.')])
        plt.plot(acc_train,color='b',label=self.files[i+3][:self.files[i+3].find('.')])
        plt.ylim(0, 1.1)
        plt.legend()
        plt.title("accuracy")

        plt.subplot(3,2,2*i+2)
        plt.plot(loss_test,color='r',label=self.files[i][:self.files[i].find('.')])
        plt.plot(loss_train,color='b',label=self.files[i+3][:self.files[i+3].find('.')])
        plt.legend()
        plt.title("loss")

def extract_model(model_path):
    # load model and dataset
    path='./lighning_saved_params/'+model_path
    model_recovered=EmotionRec()
    model_recovered.load_state_dict(torch.load(path))
    model_recovered.eval()
    trainer = pl.Trainer()
    test_loader_recovered=torch.load("./lighning_saved_params/test_loader.pt")
    y_hat=trainer.predict(model_recovered,test_loader_recovered)

    # split y_hat (the prediction of the model) to y_hat of each classifier
    y_hat_stress=[]
    y_hat_arousal=[]
    y_hat_valence=[]
    for i in range(len(y_hat)):
        y_hat_stress.append(y_hat[i][0])
        y_hat_arousal.append(y_hat[i][1])
        y_hat_valence.append(y_hat[i][2])

    y_hat_stress=np.concatenate(y_hat_stress, axis=0)
    y_hat_arousal=np.concatenate(y_hat_arousal, axis=0)
    y_hat_valence=np.concatenate(y_hat_valence, axis=0)

    # split y (the true labels) to y of each classifier
    y_stress=test_loader_recovered.dataset.tensors[1][:,1].numpy()
    y_arousal=test_loader_recovered.dataset.tensors[1][:,2].numpy()
    y_valence=test_loader_recovered.dataset.tensors[1][:,3].numpy()
    samples=test_loader_recovered.dataset.tensors[0].numpy()
    pp=test_loader_recovered.dataset.tensors[1][:,0].numpy()

    return y_hat_stress,y_hat_arousal,y_hat_valence,y_stress,y_arousal,y_valence,samples,pp