import tensorflow as tf
import numpy as np
import cPickle

def unpickle(file):
    f = open(file, 'rb')
    data = cPickle.load(f)
    f.close()
    return data

test = unpickle("CIFAR-10-py/data_batch_1")

arr = np.array(test['data'], np.uint8)
arr = arr.reshape(10000, 3, 32, 32)


#print test['labels']
