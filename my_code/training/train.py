import os
import sys
sys.path.append(os.path.abspath(sys.path[0]+'/..'))
from dataset.dataset import ImageDataset, Dataset, KerasGenerator, KerasAugGenerator
from keras.applications import VGG16, ResNet50
from keras.models import Model
from keras.layers import Dense, Flatten, GlobalMaxPooling2D
import tensorflow as tf
import keras.backend as K
from keras.models import load_model
import keras
import numpy as np
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras import losses
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from math import exp
from models.models import build_age_model


def age_loss(y_true, y_pred):
    weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1))/(K.int_shape(y_pred)[1] - 1), dtype='float32')
    return (1.0 + weights) * losses.categorical_crossentropy(y_true, y_pred)

def age_mae(y_true, y_pred):
    true_age = K.sum(y_true * K.arange(0, 101, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, 101, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae

ckpt = ModelCheckpoint(
    "checkpoints/best.h5",
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode="auto"
)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=0.00005)

LR = 0.0001
def exp_decay(epoch):
    initial_lrate = LR
    k = 0.1
    lrate = initial_lrate * exp(-k*epoch)
    return lrate

lrate = LearningRateScheduler(exp_decay)

callbacks = [ckpt, reduce_lr]

img_size = (224,224)
training_dataset = Dataset(csv_file="data/train_aug.csv", num_classes=101, to_categorical=False)
training_dataset.balance_dataset(max_occ_class=1500)
training_dataset.labels_to_categorical()

validation_dataset = Dataset(csv_file="data/val.csv", num_classes=101)
train_generator = KerasAugGenerator(training_dataset, batch_size=128, img_size=img_size)
val_generator = KerasGenerator(validation_dataset, batch_size=32, img_size=img_size)


tf_config = tf.ConfigProto(allow_soft_placement=False)
tf_config.gpu_options.allow_growth = True
s = tf.Session(config=tf_config)
K.set_session(s)

def change_model(model, new_input_shape=(None, 128,128, 3)):
    # replace input shape of first layer
    model._layers[0].batch_input_shape = new_input_shape

    # feel free to modify additional parameters of other layers, for example...
    #model._layers[2].pool_size = (8, 8)
    #model._layers[2].strides = (8, 8)

    # rebuild model architecture by exporting and importing via json
    new_model = keras.models.model_from_json(model.to_json())
    new_model.summary()

    # copy weights from old model to new one
    for i,layer in enumerate(new_model.layers):
        if i%10==0:
            print(i)
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    # test new model on a random input image
    X = np.random.rand(10, 128,128, 3)
    y_pred = new_model.predict(X)
    print(y_pred)
    return new_model




model = build_age_model()
model.load_weights("checkpoints/best_last.h5")

num_layers = len(model.layers)
freeze_layers = num_layers-7
for i,layer in enumerate(model.layers):
    print(i, layer.name)
    if i<freeze_layers:
        layer.trainable=False
    else:
        layer.trainable=True


model.compile(loss=age_loss, metrics=["accuracy", age_mae], optimizer=Adam(lr=0.001)) #0.001   

print(len(model.layers))
print(model.summary())


model.fit(train_generator,
                validation_data=val_generator,
                verbose=1,  workers=7, epochs=50, callbacks=callbacks
                )
model.save("age_net_aug_60.hdf5")



