import mxnet as mx
mx.random.normal(0,10,(3,3), ctx=mx.gpu()).asnumpy()