import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
# import keras

tf.enable_eager_execution()


filename_train = 'train.tfrecords' 
filename_test = 'validation.tfrecords' 
HEIGHT = 32
WIDTH = 32
DEPTH = 3
start = time.time()
# def normalize(image):
#   """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
#   image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
#   return image
def normalize(image):
  """Convert `image` from [0, 255] -> [0, 1] floats."""
  image = tf.cast(image, tf.float32) * (1. / 255)
  return image  

def preprocess(image):
    """Preprocess a single image in [height, width, depth] layout."""
    # Pad 4 pixels on each dimension of feature map, done in mini-batch
    image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
    image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
    image = tf.image.random_flip_left_right(image)
    return image

def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # Dimensions of the images in the CIFAR-10 dataset.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
    # input format.
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])
    image = normalize(image)

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, 10)

    # Custom preprocessing.
    image = preprocess(image)

    return image, label  

# Hyperparameters
# batch_size = 128
num_classes = 10
# epochs = 50
l = 8
num_filter = 20
compression = 0.5
dropout_rate = 0.1
batch_size = 64
num_epochs = 100
print("dropout -- ",dropout_rate)
print("batch_size",batch_size)
print("num_filters",num_filter)
print("layers per dense block",l)
# -----------TRAIN TFRecords
dataset_train = tf.data.TFRecordDataset(filename_train).repeat(num_epochs)
dataset_train = dataset_train.map(parser, num_parallel_calls=batch_size)
dataset_train = dataset_train.shuffle(buffer_size=40000) 

# Batch it up.
dataset_train = dataset_train.batch(batch_size)
iterator = dataset_train.make_one_shot_iterator()
image_batch, label_batch = iterator.get_next()

# -------------TEST TFRecords
dataset_test = tf.data.TFRecordDataset(filename_test).repeat(1)
dataset_test = dataset_test.map(parser, num_parallel_calls=batch_size)
dataset_test = dataset_test.shuffle(buffer_size=10000) 

# Batch it up.
dataset_test = dataset_test.batch(batch_size)
iterator = dataset_train.make_one_shot_iterator()
image_batch, label_batch = iterator.get_next()



############################# MODEL ################################
# Dense Block
def add_denseblock(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    temp = input
    for _ in range(l):
        BatchNorm = tf.keras.layers.BatchNormalization()(temp)
        relu = tf.keras.layers.Activation('relu')(BatchNorm)
        Conv2D_3_3 = tf.keras.layers.Conv2D(int(num_filter*compression), (3,3), use_bias=False ,padding='same')(relu)
        if dropout_rate>0:
          Conv2D_3_3 = tf.keras.layers.Dropout(dropout_rate)(Conv2D_3_3)
        concat = tf.keras.layers.Concatenate(axis=-1)([temp,Conv2D_3_3])
        
        temp = concat
        
    return temp
def add_transition(input, num_filter = 12, dropout_rate = 0.2):
    global compression
    BatchNorm = tf.keras.layers.BatchNormalization()(input)
    relu = tf.keras.layers.Activation('relu')(BatchNorm)
    Conv2D_BottleNeck = tf.keras.layers.Conv2D(int(num_filter*compression), (1,1), use_bias=False ,padding='same')(relu)
    if dropout_rate>0:
      Conv2D_BottleNeck = tf.keras.layers.Dropout(dropout_rate)(Conv2D_BottleNeck)
    avg = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(Conv2D_BottleNeck)
    
    return avg
def output_layer(input):
    global compression
    BatchNorm = tf.keras.layers.BatchNormalization()(input)
    relu = tf.keras.layers.Activation('relu')(BatchNorm)
    AvgPooling = tf.keras.layers.AveragePooling2D(pool_size=(2,2))(relu)
    flat = tf.keras.layers.Flatten()(AvgPooling)
    output = tf.keras.layers.Dense(num_classes, activation='softmax')(flat)
    
    return output   

input = tf.keras.Input(shape=(HEIGHT, WIDTH, DEPTH,))
First_Conv2D = tf.keras.layers.Conv2D(num_filter, (3,3), use_bias=False ,padding='same')(input)

First_Block = add_denseblock(First_Conv2D, num_filter, dropout_rate)
First_Transition = add_transition(First_Block, num_filter, dropout_rate)

Second_Block = add_denseblock(First_Transition, num_filter, dropout_rate)
Second_Transition = add_transition(Second_Block, num_filter, dropout_rate)

Third_Block = add_denseblock(Second_Transition, num_filter, dropout_rate)
Third_Transition = add_transition(Third_Block, num_filter, dropout_rate)

Last_Block = add_denseblock(Third_Transition,  num_filter, dropout_rate)
output = output_layer(Last_Block)

model = tf.keras.Model(inputs=[input], outputs=[output])
sgd = tf.keras.optimizers.SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])
# model.summary()
# model.fit_generator(dataset_train,steps_per_epoch=624,epochs=num_epochs,verbose=1)#,validation_data=dataset_test)
# model.evaluate_generator(dataset_test,steps=156,verbose=1)
#######################################################################################
for zz in range(1,num_epochs+1):
  # determine Loss function and Optimizer
  print("TrainEpochs number{}/{}".format(zz,num_epochs))
  
  model.fit_generator(dataset_train,steps_per_epoch=624,epochs=1,verbose=1)#,validation_data=dataset_test)

  print("Test Epochs number{}/{}".format(zz,num_epochs))
  model.evaluate_generator(dataset_test,steps=156,verbose=1)
  print("  ")
  print("------------------------")

# for x,y in zip(enumerate(dataset_train),enumerate(dataset_test)):
#     j,i = x
#     k,l = y
#     print("train",i[0].shape,i[1].shape,j)
#     print("test",l[0].shape,l[1].shape,k)
# for j,i in enumerate(dataset_test):
#     print("test",i[0].shape,i[1].shape,j)
print(dataset_train,dataset_test)    
print('total time', time.time()-start)    
print("----------------END---------------------------")