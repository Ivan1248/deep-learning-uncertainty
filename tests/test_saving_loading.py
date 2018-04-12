import os, sys

print("Loading and preparing data...")
from data_utils import Cifar10Loader
ds_test = Cifar10Loader.load_test()[:200]

print("Initializing model...")
from models import Dummy
from training import train

get_model = lambda: Dummy(
    input_shape=ds_test.image_shape,
    class_count=ds_test.class_count,
    batch_size=1,
    training_log_period=100)


saved_path = None

def load_save():
    import tensorflow as tf
    global saved_path

    model = get_model()

    if saved_path is not None:
        print("Loading model...")
        print(saved_path)
        model.load_state(saved_path)
        print(saved_path)

    print("Starting training and validation loop...")
    train(model, ds_test, ds_test, epoch_count=2)
    
    if saved_path is None:
        print("Saving model...")
        import datetime
        import dirs
        saved_path = dirs.SAVED_MODELS + '/dummy-' + datetime.datetime.now(
        ).strftime("%Y-%m-%d")+"-ckpt"
        saved_path = model.save_state(saved_path)
        print(saved_path)


for i in range(4):
    print([i]*50)
    load_save()
