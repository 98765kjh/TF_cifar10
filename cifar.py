import tensorflow as tf
import cPickle
import gzip

def unpickle(file):
    f = gzip.open(file, 'rb')
    data = cPickle.load(f)
    f.close()
    return data

test = unpickle("CIFAR-10/cifar-10-python.tar.gz")
print test

