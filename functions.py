# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 03:17:38 2021

@author: Azooz95
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from PIL import Image
import math
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,f1_score
import tensorflow_addons as tfa
import pandas as pd
import winsound
#from numba import cuda 
import cv2 as cv
import shutil
from sklearn.model_selection import KFold
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
'''config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.9
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = True
device = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(device[0], True)'''
################################## building preprocessing ######################
class pre_processing:
    uploaded_data1 = None
    ver = '[' +'.'*20+']'
    
    def uploading_data(self,path, data, dirctFrom_path = 0):
        images = []
        
        if dirctFrom_path == 1:
            for i in os.listdir(path):
                images.append(np.asarray(Image.open(path+"\\"+i)))
        if dirctFrom_path == 0:
            for i in data:
                print("this is "+str(i))
                images.append(np.asarray(Image.open(path+"\\"+i)))
             
        self.uploaded_data1 = images
        return images 
    
    def flipping_images(self):
        a = []
        for i in self.uploaded_data1:
            a.append(i)
            a.append(np.asarray(Image.fromarray(i).transpose(Image.FLIP_LEFT_RIGHT)))
            a.append(np.asarray(Image.fromarray(i).transpose(Image.FLIP_TOP_BOTTOM)))
            
        self.uploaded_data1 = a
            
    def rotation_images(self,degrees):
        a =[]
        for i in degrees:
            for j in self.uploaded_data1:
               a.append(j)
               a.append(np.asarray(Image.fromarray(j).rotate(i)))
        self.uploaded_data1 = a
    
    def get_data(self):
        return self.uploaded_data1
    
        
           
         
    def saving(self, path, name, size = (224,224)):
        created_path = path+"\\"+name
        current_path = os.getcwd()
        print(current_path)
        os.chdir(path)
        if not os.path.isdir(created_path):
            os.mkdir(created_path)
        for index in range(len(self.uploaded_data1)):
            Image.fromarray(np.array(self.uploaded_data1[index])).save(created_path+"\\"+str(index)+".png")
        
        
    def down_sampling(self, number_of_donwn):
        random.shuffle(self.uploaded_data1)
        a = self.uploaded_data1[:number_of_donwn]
        self.uploaded_data1 = a
        return a
    
    def get_data_from_file(self,path):
        a = os.listdir(path)
        self.uploaded_data1 = a
        return a
    
    def changing_names(self, path):
        paths = [path+"\\"+i for i in os.listdir(path)]
        names = [x.split("\\")[-1] for x in paths]
        a = []
        for index,i in enumerate(paths):
            for index1,j in enumerate(os.listdir(i)):
                os.rename(i+"\\"+j, i+"\\"+names[index]+"("+str(index1)+".png")
    
    def get_labels(self,path):
        paths = [path+"\\"+i for i in os.listdir(path)]
        labels = [x.split("\\")[-1] for x in paths]
        
        return labels
    def extraction_data(self,path):
        paths = [path+"\\"+i for i in os.listdir(path)]
        a = []
        data = []
        for i in paths:
            for j in os.listdir(i):
                data.append(i+"\\"+j)
            a.append([i+"\\"+j for j in os.listdir(i)])
        #data = list(chain.from_iterable(a))
        return data,a

    def extration_label(self,data):
        isList = type(data) == list
        isDict = type(data) == pd.core.frame.DataFrame
        a =[]
        if isList:
            a = [i.split('\\')[-1].split(".")[0].split("(")[0] for i in data]
        if isDict:
            for i in data:  
                for j in data[i]:
                    g = j.split('\\')[-1].split(".")[0].split("(")[0]
                    a.append(g)
        return a
    
    def splitting(self, data,precentage):
        train,test = train_test_split(data, random_state=50,test_size=precentage)
        return train, test
    
    def converting_labels_to_number(self,label,labels):
        a = [label.index(i) for i in labels]
        return a
        
    def uploading_data_v2(self,paths,size=(224,224), grey_hist = 0, dilation_canny = 0,sharpening = 0, exce_sharp = 0, gussian = 0,normlization=0,grey =0, laplcian_filter = 0, canny = 0,grey_canny =0,gussian_blur = 0,hist = 0,dilation = 0,morphologyEx=0,addwieted = 0):
        # sharp
        kernel_del = np.ones((5,5),np.uint8)
        if sharpening == 1:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            
            
            data = np.array([cv.filter2D(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),-1,kernel) for i in paths])
            
        # excessive sharp, laplcian, canny    
        elif exce_sharp == 1:
            kernel = np.array([[1,1,1], [1,-7,1], [1,1,1]])
            if grey == 1:
                data = np.array([cv.cvtColor(cv.filter2D(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),-1,kernel), cv.COLOR_BGRA2GRAY) for i in paths])
            elif laplcian_filter:
                data = np.array([cv.Laplacian(cv.filter2D(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),-1,kernel),cv.CV_64F) for i in paths])
            elif canny:
                data = np.array([cv.Canny(cv.filter2D(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),-1,kernel),100,180) for i in paths])
            else:
               data = np.array([cv.filter2D(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),-1,kernel) for i in paths])
       
       # Gussian filtter with grey,laplcian,canny 

        elif gussian == 1:
            kernel = np.array([[-1,-1,-1,-1,-1],
                                [-1,2,2,2,-1],
                                [-1,2,8,2,-1],
                                [-1,2,2,2,-1],
                                [-1,-1,-1,-1,-1]]) / 8.0
            
            if grey == 1:
                data = np.array([cv.cvtColor(cv.filter2D(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),-1,kernel),cv.COLOR_BGR2GRAY) for i in paths])
            elif laplcian_filter:
                data = np.array([cv.Laplacian(cv.filter2D(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),-1,kernel),cv.CV_64F) for i in paths])
            elif canny:
                data = np.array([cv.Canny(cv.cvtColor(cv.filter2D(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),-1,kernel),cv.COLOR_BGR2GRAY),25,150) for i in paths])
            elif grey_canny:
                data = np.array([cv.Canny(cv.cvtColor(cv.filter2D(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),-1,kernel),cv.COLOR_BGR2GRAY),50,250) for i in paths])
            elif dilation:
                data = np.array([cv.dilate(cv.filter2D(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),-1,kernel),kernel_del,iterations=1) for i in paths])
            elif dilation_canny:
                data = np.array([cv.Canny(cv.dilate(cv.filter2D(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),-1,kernel),kernel_del,iterations=1),50,100) for i in paths])
            elif morphologyEx:
                data = np.array([cv.morphologyEx(cv.filter2D(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),-1,kernel),cv.MORPH_OPEN,kernel_del) for i in paths])
            elif addwieted:
                 data = np.array([cv.filter2D(np.asarray(Image.fromarray(self.gussain_blur_with_weited(self.resize_image(np.asarray(Image.open(i).convert('RGB'))),4,target=size[0])).resize((size[0],size[1]))),-1,kernel) for i in paths])
            else:
                data = np.array([cv.filter2D(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),-1,kernel) for i in paths])
        elif dilation:
            if canny:
                print("it is here")
                data = np.array([cv.Canny(cv.dilate(np.asarray(Image.open(i).convert('L').resize((size[0],size[1]))),kernel_del,iterations=1),15,50) for i in paths])
            else:
                data = np.array([cv.dilate(np.asarray(Image.open(i).convert('L').resize((size[0],size[1]))),kernel_del,iterations=2) for i in paths])
        # Normlization images    
        elif normlization == 1:
            norm = np.zeros((800,800))
            #final = cv.normalize(img,  norm, 0, 255, cv.NORM_MINMAX)
            data = np.array([cv.normalize(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),norm,0,255,cv.NORM_MINMAX) for i in paths])
            
        elif gussian_blur:
            if canny:
                data = np.array([cv.Canny(cv.GaussianBlur(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),(5,5),3),5,50) for i in paths])
            elif hist:
                data = np.array([cv.equalizeHist(cv.GaussianBlur(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),(5,5),3)) for i in paths])
            elif laplcian_filter:
                data = np.array([cv.Laplacian(cv.GaussianBlur(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),(5,5),3),cv.CV_64F) for i in paths])
            elif dilation:
                data = np.array([cv.dilate(cv.GaussianBlur(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),(5,5),3),kernel_del,iterations=1) for i in paths])
            else:
                data = np.array([cv.GaussianBlur(np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))),(5,5),3) for i in paths])
        elif addwieted:
                data = np.array([np.asarray(Image.fromarray(self.gussain_blur_with_weited(self.resize_image(np.asarray(Image.open(i).convert('RGB'))),4,target=size[0])).resize((size[0],size[1]))) for i in paths])
        elif grey_hist:
            clahe = cv.createCLAHE(clipLimit=0.1, tileGridSize=(8,8))
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            data = []
            for i in paths:
                im = np.asarray(Image.open(i).convert('RGB'))
                img = self.resize_image(im)
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                x = clahe.apply(img)
                a=np.repeat(x[..., np.newaxis], 3, -1)
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                a = cv.filter2D(a,-1,kernel)
                data.append(a)
            data = np.array(data)
            #data = np.array([cv.filter2D(np.repeat(clahe.apply(cv.cvtColor(self.resize_image(np.asarray(Image.open(i).convert('RGB')),target=224),cv.COLOR_BGR2GRAY))[..., np.newaxis],3,1),-1,kernel) for i in paths])
        else:
            data = np.array([np.asarray(Image.open(i).convert('RGB').resize((size[0],size[1]))) for i in paths])
            
            
        return data
    
    def hot_coding(self, labels):
        return tf.keras.utils.to_categorical(labels)

    def get_propotion(self, data):
        total_number = 0
        for i in range(len(data.count())):
            total_number += data[data.columns[i]].count()
        a = {i: len(data[i])/total_number for i in data}
        return a
    
    def change_value(self, value):
        self.uploaded_data1 = value
    
    '''def verbose(self, total, current):
        pre = total/20
        sybol = "="
        if round(pre) == current:
            self.ver = self.ver[:current_index+1] + sybol + self.ver[current+2:]'''
        
        
    def precentage(self,total, current):
        pre = (current / total) * 100
        print("Precentage: "+str(pre)+"%")
    
    def resize_image(self,img,target=224):
        x = img[int(img.shape[0]/2),:,:].sum(1)
        r=(x>x.mean()/10).sum()/2
        s=target/r
        return cv.resize(img,(target,target),fx=s,fy=s)
    
    def gussain_blur_with_weited(self,img,wight,target=224):
        im = cv.GaussianBlur(img,(0,0),target/30)
        image = cv.addWeighted(img,wight,im,-wight, 128)
        mask = mask = np.zeros_like(image)
        center = (image.shape[1]//2, image.shape[0]//2)
        radius = int(target *  0.90) 
        color = (1, 1, 1)
        mask = cv.circle(mask, center=center, radius=radius, color=color, thickness=-1)
        image = image * mask + (1-mask) *128
        return image
        #return cv.addWeighted(img,wight,cv.GaussianBlur(img,(0,0),target/30),-wight,128)
        

    def pairs(self,data, all_labels):
        pairImages = []
        pairLabels = []
        numClasses = len(np.unique(all_labels))
        idx = [np.where(np.array(all_labels) == i)[0] for i in range(0, numClasses)]
           	# loop over all images
        for idxA in range(data.shape[0]):
            currentImage = data[idxA]
            label = all_labels[idxA]
            idxB = np.random.choice(idx[label])
            posImage = data[idxB]
               		# prepare a positive pair and update the images and labels
               		# lists, respectively
            pairImages.append([currentImage, posImage])
            pairLabels.append([1])
               		# grab the indices for each of the class labels *not* equal to
               		# the current label and randomly pick an image corresponding
               		# to a label *not* equal to the current label
            negIdx = np.where(np.array(all_labels) != label)[0]
            negImage = data[np.random.choice(negIdx)]
               		# prepare a negative pair of images and update our lists
            pairImages.append([currentImage, negImage])
            pairLabels.append([0])
               
        return (np.array(pairImages), np.array(pairLabels))
            
            
               
############################# machine learning toolkit ###########################
class machine_learning_toolkit:
    history = {}
    def visulaization(self, data,name,figsize=(10,10), number_images=9):
        plt.figure(figsize=(figsize[0],figsize[1]))
        number = round(math.sqrt(number_images))
        for i in range(number_images):
            ax = plt.subplot(number, number, i + 1)
            plt.title(name)
            plt.imshow(data[i])
            ax.axis('off')
    
    
    def convussion_matrix(self, y_true, y_predict, labels):
        conv_data = confusion_matrix(y_true, y_predict, labels=labels)
        dataframe = pd.DataFrame(conv_data, index=labels, columns=labels)
        sn.set(font_scale = 1.4)
        sn.heatmap(dataframe, annot=True, annot_kws={"size": 16})
        plt.show()
        
        
    def fine_tune(self, model,n):
        model.trainable = True
        if n != 0:
            print("it is herer")
            for layer in model.layers[:n]:
                layer.trainable =  False
        return model

    # def fine_tuneV2(self, model,n):
    #     model.trainable = False
    #     if model.layers.
    #     return model
    
    
    def data(self, splitting = 0.2, rescale = 1./255):
        return ImageDataGenerator(rescale = rescale , validation_split=splitting)
    
    
    def uploading_images(self,Path, subset,Image_size = [224,224],batch_size = 16):
        data = self.data()
        return data.flow_from_directory(Path, # same directory as training data
               target_size=(Image_size[0], Image_size[1]),
               batch_size=batch_size,
               class_mode='categorical',
               subset=subset)
    
    def test_data(self,validation_dataset, batches_taken, NO_fine_tuneing):
        val_batches = tf.data.experimental.cardinality(validation_dataset)
        test_dataset = validation_dataset.take(val_batches // batches_taken)
        validation_dataset = validation_dataset.skip(val_batches // batches_taken)
        return test_dataset, validation_dataset
    
    def train(self, data, models, epochs, bs,callback,NO_fine_tuneing, path , name,conv=0, simense=0, alarm = 0, simple = 0,no_dense=0, dense_layers=0, less_dense_layers =0,number_folds =0,addjusting_selu=0,baby_adult_layers=0):
        for index,i in enumerate(models): 
            # tf.keras.backend.clear_session()
            print("_______________________"+i+"_______________________")
            model = models[i]
            model = self.fine_tune(model,NO_fine_tuneing[index])
            if(no_dense == 1):
                x = Dropout(0.5)(model.output)
                x = Flatten()(x)
                predction = Dense(5,activation='softmax')(x)
            
            elif(dense_layers == 1):
                x = Flatten()(model.output)
                x = Dropout(0.4)(x)
                x = Dense(512, activation="relu")(x)
                x = Dense(512, activation="relu")(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = Dense(256, activation="relu")(x)
                x = Dense(128, activation="relu")(x)
                x = Dropout(0.2)(x)
                x = Dense(64,activation="relu")(x)
                x = Dense(64,activation="relu")(x)
                x = Dropout(0.2)(x)
                predction = Dense(5,activation='softmax')(x)

            elif addjusting_selu:
                x = GlobalAveragePooling2D()(model.output)
                x = Flatten()(model.output)
                x = Dense(512,activation="selu")(x)
                x = Dense(512,activation="selu")(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = Dense(256,activation="selu")(x)
                x = Dense(128,activation="selu")(x)
                # x = Dense(128,activation="relu")(x)
                # x = tf.keras.layers.BatchNormalization()(x)
                x = Dropout(0.2)(x)
                # x = Dense(64,kernel_regularizer=regularizers.l2(1e-4),activation="relu")(x)
                x = Dense(64,activation="selu")(x)
                x = Dense(64,activation="selu")(x)
                # x = Dropout(0.2)(x)
                # x = Dense(32,activation='selu')(x)
                predction = Dense(5,activation='softmax')(x)

            elif less_dense_layers ==1:
                x = Flatten()(model.output)
                x = Dropout(0.2)(x)
                x = Dense(128, activation="selu")(x)
                x = Dense(128, activation="selu")(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = Dropout(0.2)(x)
                x = Dense(64,activation="selu")(x)
                x = Dense(64,activation="selu")(x)
                predction = Dense(5,activation='softmax')(x)

            elif baby_adult_layers:
                x = GlobalAveragePooling2D()(model.output)
                x = Flatten()(model.output)
                x = Dense(5,activation="softmax")(x)
                x = Dense(64,activation="selu")(x)
                x = Dense(64,activation="selu")(x)
                x = Dropout(0.2)(x)
                x = Dense(256, activation="selu")(x)
                x = Dense(128,activation="selu")(x)
                x = Dropout(0.4)(x)
                x = Dense(64,activation="selu")(x)
                x = Dense(64,activation="selu")(x)
                predction = Dense(5,activation='softmax')(x)

            elif simple:
                print("it is here 4")
                x = GlobalAveragePooling2D()(model.output)
                x = Flatten()(x)
                x = Dense(128, activation="selu")(x)
                x = Dense(64, activation="selu")(x)
                x = Dropout(0.4)(x)
                predction = Dense(5,activation='softmax')(x)
                
            elif conv:
                my_model = Sequential()
                my_model.add(models[i])
                # my_model.add(tf.keras.layers.Conv2D(32, (2,2), activation="relu", padding='same')),
                my_model(Flatten())
                # x = GlobalAveragePooling2D()(x)
                # x = Flatten()(x)
                # x = Dense(512, kernel_regularizer=regularizers.l2(1e-4), activation="selu")(x)
                # x = Dense(512, kernel_regularizer=regularizers.l2(1e-4),activation="selu")(x)
                # x = tf.keras.layers.BatchNormalization()(x)
                # x = Dense(256, kernel_regularizer=regularizers.l2(1e-4),activation="selu")(x)
                # x = Dense(128, kernel_regularizer=regularizers.l2(1e-4),activation="selu")(x)
                # x = tf.keras.layers.BatchNormalization()(x)
                # x = Dropout(0.2)(x)
                # x = Dense(64,kernel_regularizer=regularizers.l2(1e-4),activation="selu")(x)
                # x = Dense(64,kernel_regularizer=regularizers.l2(1e-4),activation="selu")(x)
                # x = Dropout(0.2)(x)
                # x = Dense(32,activation='selu')(x)
                # my_model.add(Dense(5,activation='softmax'))

            model = Model(inputs=model.input, outputs=predction)
            model.summary()
            model.compile(
                   loss='categorical_crossentropy',
                   optimizer=tf.keras.optimizers.Adam(lr=0.00001),
                   metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),tfa.metrics.F1Score(5),tf.keras.metrics.AUC()]
                   )
            if number_folds:
                kf = KFold(n_splits=number_folds,shuffle=True, random_state=50)
                data_input = data["train_x"]
                data_target = data["train_y"]
                for index,(train, test) in enumerate(kf.split(data_input,data_target)):
                    print("folds: " + str(index))
                    self.history[i+str(index)] = model.fit(data_input[train],data_target[train],
                                                validation_data = (data_input[test],data_target[test]),
                                                verbose = 1,
                                                epochs = epochs,
                                                batch_size= bs,
                                                callbacks=[callback])
                model.save(path+"\\"+name+"_"+i+".h5")
                # outputs = Dense(48)(x)
            elif simense:
                model = Model(inputs=model.input, outputs=x)
                model = model.layers[1:]
                model = tf.keras.Sequential(model)
                # model.summary()
                imgA = Input((224,224,3))
                imgB = Input((224,224,3))
                feateA = model(imgA)
                feateB = model(imgB)
                
                
                # distance = Lambda(self.euclidean_distance)([feateA, feateB])
                L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))
                L1_distance = L1_layer([feateA, feateB])
                prediction_last = Dense(1, activation="sigmoid")(L1_distance)
                model = Model(inputs=[imgA, imgB], outputs=prediction_last)
                model.compile(
                    loss='binary_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(lr=0.00001),
                    metrics=['accuracy',tf.keras.metrics.Recall(),tf.keras.metrics.Precision(),tfa.metrics.F1Score(2),tf.keras.metrics.AUC()]
                    )
                model.summary()
                self.history[i] = model.fit([data["train"][:, 0], data["train"][:, 1]], data["train_label"],
                                                validation_data=([data["test"][:, 0], data["test"][:, 1]], data["test_label"]),
                                                verbose = 1,
                                                epochs = epochs,
                                                batch_size= bs,
                                                callbacks=[callback])
                model.save(path+"\\"+name+"_"+i+".h5")

            else:
                self.history[i] = model.fit(data["train_x"],data["train_y"],
                                                validation_data = (data["test_x"],data["test_y"]),
                                                verbose = 1,
                                                epochs = epochs,
                                                batch_size= bs,
                                                callbacks=[callback])
                model.save(path+"\\"+name+"_"+i+".h5")
                # plot_model(model, to_file="D:\\Data\\\FYP_data\\"+i+'_model_plot.png', show_shapes=True, show_layer_names=True)
            # if alarm == 1:
            #     self.alarm()
            #     if index == len(models):
            #         for _ in range(len(models)):
            #             self.alarm()
            
    def get_history(self):
        return self.history
    
    def saving_history(self, main_path, name):
        created_path = main_path+"\\"+name
        current_path = os.getcwd()
        os.chdir(main_path)
        if not os.path.isdir(created_path):
            os.mkdir(created_path)
        for i in self.history:
            hist_json_file = created_path+'\\history'+i+'.json' 
            hist_csv_file = created_path+'\\history'+i+'.csv' 
            df = pd.DataFrame(self.history[i].history)
            with open(hist_json_file, mode='w') as f:
                df.to_json(f)
            with open(hist_csv_file, mode='w') as f:
                df.to_csv(f)
        os.chdir(current_path)
        
        
    def alarm(self):
        fre = 1000
        winsound.Beep(fre, 500)
        
    def details_layers(self, model):
        for i, layer in enumerate(model.layers):
            print(i, layer.name, layer.trainable)
            
    def evaluate(self, model, test_x, test_y, batch_size=16):
        results = model.evaluate(test_x, test_y, batch_size=batch_size)
        return results
    
    def loading_model(self, path):
        return tf.keras.models.load_model(path)
    
    def euclidean_distance(vectors):
	# unpack the vectors into separate lists
        (featsA, featsB) = vectors
        # compute the sum of squared distances between the vectors
        sumSquared = K.sum(K.square(featsA - featsB), axis=1,
            keepdims=True)
        # return the euclidean distance between the vectors
        return K.sqrt(K.maximum(sumSquared, K.epsilon()))
########################### Image processing ###########################

class Image_Processing:
    
    def canny_Edge(self, image):
        img = cv.imread(image,0)
        edges = cv.Canny(img,100,200)
        plt.imshow(edges)
        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()
        
    
    def edge(self, image):
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Using the Canny filter to get contours
        edges = cv.Canny(gray, 20, 30)
        # Using the Canny filter with different parameters
        edges_high_thresh = cv.Canny(gray, 60, 120)
        # Stacking the images to print them together
        # For comparison
        images = np.hstack((gray, edges, edges_high_thresh))
        
        # Display the resulting frame
        plt.imshow(images)
####################### testing #########################################

'''
img = "D:\\Data\\FYP_data\\Third_data\\Mild\\Mild(24.png"
im = Image.open(img)
p = Image_Processing()
p.canny_Edge(img)
p.edge(img)
'''
'''
la = {"0":"No_DR", "1":"Mild", "2":"Moderate", "3":"Sever","4":"Proliferate_DR"}
total_data = {"No_DR":[],"Mild":[],"Moderate":[] , "Sever":[] , "Proliferate_DR":[]}
path_csv = "D:\\Data\\FYP_data\\Fourth_data\\test.csv"
path_img = "D:\\Data\\FYP_data\\Fourth_data\\test_images"
imags = os.listdir(path_img)
images = [i.split(".")[0] for i in imags]
data = {"No"}
d = pd.read_csv(path_csv)
t = d.to_dict()

for i in images:
    for index,j in enumerate(t['id_code']):
        if i == t['id_code'][j]:
            print(index)
            no = d["diagnosis"][index]
            total_data[la[str(no)]].append(path_img+"\\"+i+".png")
            break

path_target = "D:\\Data\\FYP_data\\Fiveth_data"
os.chdir(path_target)
for i in total_data:
    if not os.path.isdir(i):
        os.mkdir(i)
    for j in total_data[i]:
        shutil.move(j,path_target+"\\"+i)
'''
            
'''path = "D:\Data\FYP_data\First_data\colored_images\colored_images\Severe"
path1 = "D:\\Data\\FYP_data\\First_data\\colored_images\\colored_images\\test"
path_normal = "D:\\Data\\FYP_data\\First_data\\colored_images\\colored_images\\No_DR"
path2 = "D:\\Data\\FYP_data\\second_data\\No_DR"
path3 = "D:\Data\FYP_data\First_data\colored_images\colored_images\Mild"
degrees = [45,135]
preProcessing = pre_processing()
data = preProcessing.get_data_from_file(path_normal)
a = preProcessing.down_sampling(873)
preProcessing.uploading_data(path_normal,a,dirctFrom_path=0)
print(len(preProcessing.get_data()))


preProcessing.flipping_images()
preProcessing.rotation_images(degrees)

print(preProcessing.get_data()[0].shape)

ml = machine_learning_toolkit(preProcessing.get_data())
ml.visulaization(number_images=18)

aug_im = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomRotation(
    0.2
)])
data_augmentation = tf.keras.Sequential([
              tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
            ])
a = []
for i in factors:
    print(i)
    data_augmentation = tf.keras.Sequential([
                  tf.keras.layers.experimental.preprocessing.RandomRotation(i)
                ])
    data_augmentation = None
    
a = tf.image.rotate(aug_images_flipped[0],20,flip_mode="reflect")
images = []
for i in a:
    images.append(tf.expand_dims(np.asarray(Image.open(path+"\\"+i)),0))
    
a = np.asarray(Image.fromarray(preProcessing.get_data()[0]).rotate(45))
b = Image.fromarray(preProcessing.get_data()[0]).transpose(Image.FLIP_LEFT_RIGHT)
t = Image.fromarray(np.asarray(b)).transpose(Image.FLIP_TOP_BOTTOM)
plt.imshow(b)
plt.imshow(t)'''
