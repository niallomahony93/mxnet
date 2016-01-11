import mxnet as mx
import mxnet.ndarray as nd


for i in range(1000):
    noise = mx.random.normal(0,10,(i,i),ctx=mx.gpu())