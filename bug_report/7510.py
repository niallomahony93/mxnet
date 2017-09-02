import mxnet as mx
import numpy as np

value = mx.sym.Variable('data')
sorted_adj = mx.sym.argsort(value, axis = 2)

coord_data = np.random.rand(32, 2048, 2048)
coord_blob = mx.nd.array(coord_data, mx.gpu())
e = sorted_adj.bind(mx.gpu(), {'data':coord_blob})
y = e.forward()

result = y[0].asnumpy()
vis = np.zeros(2048)
print(result[4, 0, :][1024:])
print(coord_data[4, 0, :].argsort()[1024:])
#ch = input()
for i in range (2048):
    ind = int(round(result[4, 0, i]))
    print(ind, vis[ind])
    vis[ind] = 1
cnt = 0
for i in range (2048):
    if vis[i] == 0:
        cnt += 1
print(cnt)