import os
import tensorflow as tf
from tensorflow.keras.models import Model
# from keras.applications.vgg16 import VGG16
from tensorflow.keras import preprocessing
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras import applications
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras import backend as K
# from tensorflow import keras
# import matplotlib.pyplot as plt



# session_config = tf.compat.v1.ConfigProto().gpu_options.a
# session_config.gpu_options.per_process_gpu_memory_fraction = 0.3
# config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir, session_config=session_config)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     # Currently, memory growth needs to be the same across GPUs
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
#     print(e)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#   except RuntimeError as e:
#     print(e)
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)

# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
# set_session(session)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def main():
  os.chdir("D:\Data\DataAnalysistraining\project\Animals-10")
  path = os.getcwd()
  IMAGE_SIZE = [224,224]
  nas = applications.vgg16.VGG16(input_shape=(331,331,3), weights='imagenet', include_top=False)
  # freeze the layers to be train
  for layer in nas.layers:
      layer.trainable = False



  x = Flatten()(nas.output)
  predction = Dense(10,activation='softmax')(x)
  model = Model(inputs=nas.input, outputs=predction)
  model.summary()

  model.compile(
      loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy',tf.keras.metrics.AUC(),recall_m,precision_m,f1_m]
  )
  img_height = 331
  img_width = 331
  batch_size = 16


  data = ImageDataGenerator(rescale = 1./255 , validation_split=0.2)

  train_generator = data.flow_from_directory(
  path,
  target_size=(img_height, img_width),
  batch_size=batch_size,
  class_mode='categorical',
  subset='training') # set as training data

  validation_generator = data.flow_from_directory(
  path, # same directory as training data
  target_size=(img_height, img_width),
  batch_size=batch_size,
  class_mode='categorical',
  subset='validation') # set as validation data

  
  r = model.fit_generator(
  train_generator,
  validation_data = validation_generator,
  verbose = 1,
  epochs = 2)
  print(r.history["accuracy"])


if __name__ == '__main__':
    main()

