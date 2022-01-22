# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 00:33:50 2021

@author: Azooz95
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
rows = [9,49,99]
path = "D:\\Data\\FYP_data\\Results\\full_fine_tuning_up_sample_addjusting_selu_L2_g_loss_patience_5_6learning_withSelu_3667im100e_16"
p = os.listdir(path)
resuls = {"InceptionV3":{},
          "VGG16":{},
          "ResNet50":{},}
for i in p:
    print("__________"+ i.split('.')[0] + "_________________")
    if i.split(".")[-1] == "json":
        continue
    model = i.split("history")[-1].split('.')[0]
    data = pd.read_csv(os.path.join(path,i))
    for j in rows:
        
        if model == "InceptionV3":
            print(data.loc[j][["val_loss",'val_accuracy','val_recall','val_precision']])
            resuls[model][j] = data.loc[j][["val_loss",'val_accuracy','val_recall','val_precision']]
        elif model =="VGG16":
            print(data.loc[j][["val_loss",'val_accuracy','val_recall_1','val_precision_1']])
            resuls[model][j] = data.loc[j][["val_loss",'val_accuracy','val_recall_1','val_precision_1']]                         
        else:
            print(data.loc[j][["val_loss",'val_accuracy','val_recall_2','val_precision_2']])
            resuls[model][j] = data.loc[j][["val_loss",'val_accuracy','val_recall_2','val_precision_2']]
            
def dictTolist(data):
     e = [10,50,100]
     res = {}
     for i in data:
         los = []
         acc = []
         rec = []
         pre = []
         for j in e:
             print(data[i][j-1]["val_loss"])
             los.append(data[i][j-1]["val_loss"])
             acc.append(data[i][j-1]["val_accuracy"])
             if i == "InceptionV3":
                 rec.append(data[i][j-1]["val_recall"])
                 pre.append(data[i][j-1]["val_precision"])
             elif i == "VGG16":
                 rec.append(data[i][j-1]["val_recall_1"])
                 pre.append(data[i][j-1]["val_precision_1"])
             else:
                 rec.append(data[i][j-1]["val_recall_2"])
                 pre.append(data[i][j-1]["val_precision_2"])
                 print("it is here")
         res[i] = {
             "Entropy loss":los,
             "Accuracy":acc,
             "Recall":rec,
             "Precision":pre
             }
         
     return res
         
data = dictTolist(resuls)
def vas(data):
    e = [10,50,100]
    models = ["InceptionV3", 'VGG16', 'ResNet50']
    for i in models:
        for j in data[i]:
            plt.plot(e, data[i][j])
        plt.legend(data[i].keys(),loc="upper right")
        plt.grid()
        plt.ylabel("Performance Metrics Results")
        plt.xlabel("Epochs")
        plt.title(i)
        plt.show()
vas(data)

def plotting_train_val(paths,name,sel,yl,possition):
    le = ["Train","Test"]
    for i in paths:
        print(i)
        if i.split(".")[-1] == "json":
            continue
        data = pd.read_csv(os.path.join(path,i))
        epochs = np.linspace(0,len(data),num=len(data),dtype=int)
        plt.plot(epochs,data[sel[0]])
        plt.plot(epochs,data[sel[1]])
        plt.ylabel(yl)
        plt.xlabel("epochs")
        plt.legend(le,loc=possition)
        plt.title(name+" of "+i.split("history")[-1].split(".")[0])
        plt.show()

sel = ['loss','val_loss']
n = "loss"
plotting_train_val(p,"Model loss",sel,n, "upper right")
