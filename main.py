# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 19:31:57 2021

@author: Azooz95
"""

import os
os.chdir('C:\\Users\\Azooz95\\Desktop\\venv')
import functions  
import time
import tensorflow as tf
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.nasnet import NASNetLarge
import tensorflow_addons as tfa
import winsound
import numpy as np
import pandas as pd
from itertools import chain
from numba import cuda  
import matplotlib.pyplot as plt  
import random
import cv2 
from imutils import build_montages
from sklearn.metrics import confusion_matrix
import seaborn as sns

# models = {
#     'InceptionV3':InceptionV3(input_shape=(224,224,3), weights='imagenet', include_top=False),
#     # 'VGG16': VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False),
#     # 'ResNet50': ResNet50(input_shape=(224,224,3), weights='imagenet', include_top=False),
#     # 'NASNetLarge': NASNetLarge(input_shape=(331,331,3), weights='imagenet', include_top=False),
#     # 'VGG19': VGG19(input_shape=(224,224,3), weights='imagenet', include_top=False),
#     # 'ResNet152V2': ResNet152V2(input_shape=(224,224,3), weights='imagenet', include_top=False),
#     # 'InceptionResNetV2': InceptionResNetV2(input_shape=(224,224,3), weights='imagenet', include_top=False),
#     }

# model_path = "D:\\Data\\FYP_data\\models\\full_fine_tuning_up_sample_addjusting_selu_L2_g_loss_patience_5_6learning_withSelu_3667im100e_16_InceptionV3.h5"
# path = "D:\\Data\\FYP_data\\up_sample"
# # path_test = "D:\Data\\archive\\test"
# _name = path.split("\\")[-1] + "version2"
# #device = cuda.get_current_device()
# #device.reset()

ml = functions.machine_learning_toolkit()
pre = functions.pre_processing()
# #pre.changing_names(path_train)
# data,a = pre.extraction_data(path)
# # test,a = pre.extraction_data(path_test)
# # total_data = len(data)
# # label = pre.get_labels(path)
# label = ['No_DR','Mild', 'Moderate','Sever','Proliferate_DR']
# # d_ = {label[index]: a[index] for index in range(len(a))}


# # df = pd.DataFrame(d_)
# # propotion = pre.get_propotion(df)

# train_x_paths, test_x_paths = pre.splitting(data,0.2)
# # _,validation_x_paths = pre.splitting(data,0.1)

# # ee = pre.get_propotion(train_x_paths)


# # # train_x_ps = list(chain.from_iterable(train_x_paths.values.tolist()))
# # # test_x_ps = list(chain.from_iterable(test_x_paths.values.tolist()))
# # # validation_x_ps = list(chain.from_iterable(validation_x_paths.values.tolist()))

# #all_labels = pre.extration_label(data)
# train_y = pre.extration_label(train_x_paths)
# test_y = pre.extration_label(test_x_paths)
# # validation_y = pre.extration_label(validation_x_paths)

# #all_labels = pre.converting_labels_to_number(label,all_labels)
# train_y = pre.converting_labels_to_number(label,train_y)
# test_y = pre.converting_labels_to_number(label,test_y)
# # validation_y = pre.converting_labels_to_number(label,validation_y)

# train_y = pre.hot_coding(train_y)
# test_y = pre.hot_coding(test_y)
# # validation_y = pre.hot_coding(validation_y)

# train_x = pre.uploading_data_v2(train_x_paths, gussian =1)
# test_x = pre.uploading_data_v2(test_x_paths,gussian=1) 
# ml.visulaization(train_x,'i hope it is')
# ################ pair ###############

# # image_list = np.split(train_x, train_x.shape[0])
# # image_label = np.split(train_y, train_y.shape[0])

# # left_input = []
# # right_input = []
# # targets = []

# # pairs = 2
# # for i in range(len(image_label)):
# #     for _ in range(pairs):
# #         compare_to = i
# #         while compare_to == i: #Make sure it's not comparing to itself
# #             compare_to = random.randint(0,999)
# #             print("compare to :",compare_to, "number of data: ", i)
# #         left_input.append(image_list[i])
# #         right_input.append(image_list[compare_to])
# #         if np.all(image_label[i] == image_label[compare_to]):# They are the same
# #             targets.append(1.)
# #         else:# Not the same
# #             targets.append(0.)

# # left_input = np.squeeze(np.array(left_input))
# # right_input = np.squeeze(np.array(right_input))
# # targets = np.squeeze(np.array(targets)) 

# ############## end ##########33
# #train_x = np.expand_dims(train_x, axis=-1)
# #test_y = np.expand_dims(test_x, axis=-1)
# # (pairtrain, pairtrain_y) = pre.pairs(train_x, train_y)
# # (pairtest, pairtest_y) = pre.pairs(test_x, test_y)

# ################## print the ##############
# # images =[]
# # for i in np.random.choice(np.arange(0, len(pairtrain)), size=(49,)):
# # 	# grab the current image pair and label
# # 	imageA = pairtrain[i][0]
# # 	imageB = pairtrain[i][1]
# # 	label = pairtrain_y[i]
# # 	# to make it easier to visualize the pairs and their positive or
# # 	# negative annotations, we're going to "pad" the pair with four
# # 	# pixels along the top, bottom, and right borders, respectively
# # 	output = np.zeros((36, 60), dtype="uint8")
# # 	pair = np.hstack([imageA, imageB])
# # 	#output[224, 448] = pair
# # 	# set the text label for the pair along with what color we are
# # 	# going to draw the pair in (green for a "positive" pair and
# # 	# red for a "negative" pair)
# # 	text = "neg" if label[0] == 0 else "pos"
# # 	color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)
# # 	# create a 3-channel RGB image from the grayscale pair, resize
# # 	# it from 60x36 to 96x51 (so we can better see it), and then
# # 	# draw what type of pair it is on the image
# # 	#vis = cv2.merge(pair)
# # 	#vis = cv2.resize(vis, (96, 51,3), interpolation=cv2.INTER_LINEAR)
    
# # 	cv2.putText(pair, text, (int(224/2), 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
# # 		color, 2)
# # 	# add the pair visualization to our list of output images
# # 	images.append(pair)
    
# # montage = build_montages(images, (96, 51), (7, 7))[0]
# # # show the output montage
# # cv2.imshow("Siamese Image Pairs", montage)
# # cv2.waitKey(0)
# ###########################
# # im = pre.gussain_blur_with_weited(test_x[0],4)
# # plt.figure()
# # plt.imshow(im.astype('uint8'))
# # validation_x = pre.uploading_data_v2(validation_x_paths,dilation=1)
# if test_x.shape[-1] !=3:
#     train_x =  np.repeat(train_x[..., np.newaxis], 3, -1)
#     test_x = np.repeat(test_x[..., np.newaxis], 3, -1)
#     # validation_x = np.repeat(validation_x[..., np.newaxis], 3, -1)

# '''train_x_nas = pre.uploading_data_v2(train_x_paths, size=(331,331))
# test_x_nas = pre.uploading_data_v2(test_x_paths,size=(331,331)) 
# validation_x_nas = pre.uploading_data_v2(validation_x_paths,size=(331,331))'''

# # # ml.alarm()

# # #final_data = {
# # #    'InceptionV3':{
# # #        "train_x":train_x,"train_y":train_y, "validation_x":validation_x, "validation_y":validation_y},
# # #    'VGG16':{
# # #        "train_x":train_x,"train_y":train_y, "validation_x":validation_x, "validation_y":validation_y},
# # #    'ResNet50': {
# # #        "train_x":train_x,"train_y":train_y, "validation_x":validation_x, "validation_y":validation_y},
    
# # #    }

# final_data = {"train_x":train_x,"train_y":train_y,"test_x":test_x,"test_y":test_y}
# # final_data = {"train": pairtrain , "train_label":pairtrain_y , "test": pairtest , "test_label":pairtest_y}
# # ######################### training the model ##############

# epochss = 100
# # fine_tuning_layers = [249,7,64]
# fine_tuning_layers = [0,0,0]
# batch_size = 16
# path_model = "D:\\Data\\FYP_data\\models"
# main_path = "D:\\Data\\FYP_data\\Results"


# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
# #epochs = 10
# name = 'l'+_name+'_3667im'+str(epochss)+'e_'+str(batch_size)
# print("begin")
# ml.train(final_data,models,epochss,batch_size,callback,fine_tuning_layers,path_model,name,alarm = 1,addjusting_selu=1)
# print("end")
# ml.saving_history(main_path, name)

#     #print(ml.get_history()["ResNet50"].history["loss"])
#     ml.alarm()
#  device = cuda.get_current_device()
#  device.reset()
#     tf.keras.backend.clear_session()
#     time.sleep(90) fv mnjcaudghkXFY 4arhamnooradi@gmail.com


# ######################################################

# ############################ pre_processing#######################
path_data = "D:\\Data\\archive\\test"
saved_data = "D:\\Data\\archive\\aug_data2"

if not os.path.isdir(saved_data):
    os.mkdir(saved_data)

saved_data = "D:\\Data\\archive\\aug_data2\\test"
if not os.path.isdir(saved_data):
    os.mkdir(saved_data)

paths_of_data = os.listdir(path_data)
paths_of_data = [path_data+"\\"+i for i in paths_of_data]
degres =  [60,225, 150, 330]
d = {'Severe':12,'Moderate':2,'Minimal':0.5,'Doubtful':2}

for i in paths_of_data:
    print(i)
    # #a = pre.down_sampling(193)  
    a = pre.get_data_from_file(i)
    
    t = pre.uploading_data(i,a,dirctFrom_path=0)
    # pre.flipping_images()
    pre.rotation_images(degres)
    a = pre.down_sampling(408)
    pre.saving(saved_data,i.split("\\")[-1])

    # #if i.split('\\')[-1] == "No_DR":
    # #    continue
    # pre.flipping_images()
    # pre.rotation_images(degres)
    # pre.saving(saved_data,i.split("\\")[-1])
pre.changing_names(saved_data)
# for index,i in enumerate(models):
#     print(i)
#     m = ml.fine_tune(models[i], fine_tuning_layers[index])
#     ml.details_layers(m)
# ##################################################################



# ####################### Visulization ###################
# ml.visulaization(test_x)

# #####################################
# model = models["ResNet50"]   
# model = ml.fine_tune(model,64)
# ml.details_layers(model) 
# model.save(main_path+"\\test"+".h5")    
# e = [10,50,100]
# mods = ["InceptionV3", "VGG16", "ResNet50"]
# for i in mods:
#     for j in e:
#         print("__________"+str(i)+"________ebochs: "+str(j))
#         name = 'fine_tuning_up_sample_3667im'+str(10)+'e_'+str(batch_size)    
#         path1 = "D:\\Data\\FYP_data\\models"+"\\"+name+"_"+mods[2]+".h5"
#         model1 =ml.loading_model(path1)
#         re = ml.evaluate(model1,test_x,test_y,16)   
    
# data_inceptionv3_up = {"Entropy loss":[0.9511,0.7295,1.5196],
#      "Accuracy":[0.7026,0.7470, 0.7483],
#      "Recall":[0.6754,0.7418,0.7422],
#      "Precision ":[0.7295,0.7535,0.7533]}



# data_VGG16_up = {"Entropy loss":[0.5168,0.5514,0.4915],
#      "Accuracy":[0.8069,0.8931, 0.8953],
#      "Recall":[0.7983,0.8918,0.8922],
#      "Precision ":[0.8220,0.8937,0.8973]}


# data_ResNet50_up = {"Entropy loss":[0.4240,0.5091,0.5950],
#      "Accuracy":[0.8862,0.8931, 0.8858],
#      "Recall":[0.8793,0.8918,0.8849],
#      "Precision ":[0.8943,0.8980,0.8872]}


# def vas(data):
#      e = [10,50,100] 
#      for i in data:
#          plt.plot(e, data[i])
#      plt.legend(data.keys(),loc="upper right")
#      plt.grid()
#      plt.ylabel("Performance Metrics Results")
#      plt.xlabel("Epochs")
#      plt.title("ResNet50")
#      plt.show()
    
# vas(data_ResNet50_up)
# name2 = 'fine_tuning_'+str(10)+'e'
# mod = "InceptionV3"
# path_ins = "D:\\Data\\FYP_data\\models\\fine_tuning_50e_InceptionV3"
# path_re ="D:\\Data\\FYP_data\\models\\fine_tuning_50e_ResNet50"

# path2 = "D:\\Data\\FYP_data\\models"+"\\"+name+"_"+mod+".h5"
# model =  tf.keras.models.load_model(main_path+"\\test"+".h5")

# model2 = ml.loading_model(model_path)

# # re1 = ml.evaluate(model2,test_x,test_y,16)
# y_pred = model2.predict_classes(test_x)
# cf_matrix = confusion_matrix(test_y, y_pred)
# print(cf_matrix)
# sns.heatmap(cf_matrix, annot=True)


# a = []
# for i in train_x_paths:
#     for j in train_x_paths[i]:
#         g = j.split('\\')[-1].split(".")[0].split("(")[0] 
        
#         a.append(g)

# results = ml.evaluate(model11,test_x, test_y, batch_size=16)


# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
# mtcnn_graph = tf.Graph()
# with mtcnn_graph.as_default():
#     gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1)
#     mtcnn_sess = tf.compat.v1.Session(graph=mtcnn_graph,
#                             config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#     mtcnn_sess.run(tf.compat.v1.global_variables_initializer())
#     with mtcnn_sess.as_default():
#         pnet, rnet, onet = align.detect_face.create_mtcnn(mtcnn_sess, None)

########################################### old codes #################################




# ml_test = final_project.machine_learning_toolkit()
# ml.data(splitting=0.1)


# train_data = ml.uploading_images(path,"training")
# validation_data = ml.uploading_images(path,'validation')


# training_dataNas = ml.uploading_images(path,"training", Image_size = [331,331])
# validation_dataNas = ml.uploading_images(path, 'validation', Image_size =[331,331])

# ml_test.data(splitting = 0.2)
# test_data = ml_test.uploading_images(path,'validation')
# test_dataNas = ml_test.uploading_images(path,'validation', Image_size = [331,331])

# model = models["ResNet50"]
# model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
# model.evaluate(test_data)

# train_dataset = image_dataset_from_directory(train_data,
#                                              shuffle=True,
#                                              batch_size=16,
#                                              image_size=[221,221])

'''def _verbose(self, numberOfIteration, current):
        adding = numberOfIteration/20
        sybol = "="
        a = []
        if len(a) != 20: 
            a.append(current*adding)
        if current in a:
           current_index = a.index(current)
           self.ver = self.ver[:current_index+1] + sybol + self.ver[current+2:]
           print(self.ver)'''