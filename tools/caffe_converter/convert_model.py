from __future__ import print_function
import mxnet as mx
import numpy as np
import argparse
import re

import sys

from convert_symbol import proto2symbol

caffe_flag = True
try:
    import caffe
except ImportError:
    import caffe_parse.parse_from_protobuf as parse

    caffe_flag = False


def get_caffe_iter(layer_names, layers):
    for layer_idx, layer in enumerate(layers):
        layer_name = re.sub('[-/]', '_', layer_names[layer_idx])
        layer_type = layer.type
        layer_blobs = layer.blobs
        yield (layer_name, layer_type, layer_blobs)


def get_iter(layers):
    for layer in layers:
        layer_name = re.sub('[-/]', '_', layer.name)
        layer_type = layer.type
        layer_blobs = layer.blobs
        yield (layer_name, layer_type, layer_blobs)


def main():
    parser = argparse.ArgumentParser(description='Caffe prototxt to mxnet model parameter converter.\
                    Note that only basic functions are implemented. You are welcomed to contribute to this file.')
    parser.add_argument('caffe_prototxt', help='The prototxt file in Caffe format')
    parser.add_argument('caffe_model', help='The binary model parameter file in Caffe format')
    parser.add_argument('save_model_name', help='The name of the output model prefix')
    args = parser.parse_args()

    sym, arg_params, aux_params, input_dim = process_caffe_model(args.caffe_prototxt, args.caffe_model)
    model = mx.mod.Module(symbol=sym, label_names=['prob_label', ])
    model.bind(data_shapes=[('data', tuple(input_dim))])
    model.init_params(arg_params=arg_params, aux_params=aux_params)
    model.save_checkpoint(args.save_model_name, 1)

    print ('Saved model successfully to {}'.format(args.save_model_name))

def process_caffe_model(caffe_prototxt, caffe_model, output_file=None, data=None, data_shapes=None):
    prob, input_dim = proto2symbol(caffe_prototxt)

    layers = ''
    layer_names = ''

    if caffe_flag:
        caffe.set_mode_cpu()
        net_caffe = caffe.Net(caffe_prototxt, caffe_model, caffe.TEST)
        layer_names = net_caffe._layer_names
        layers = net_caffe.layers
    else:
        layers = parse.parse_caffemodel(caffe_model)

    arg_shapes, output_shapes, aux_shapes = prob.infer_shape(data=tuple(input_dim))
    arg_names = prob.list_arguments()
    aux_names = prob.list_auxiliary_states()
    arg_shape_dic = dict(zip(arg_names, arg_shapes))
    aux_shape_dic = dict(zip(aux_names, aux_shapes))
    arg_params = {}
    aux_params = {}
    iter = ''
    if caffe_flag:
        iter = get_caffe_iter(layer_names, layers)
    else:
        iter = get_iter(layers)
    first_conv = True

    for layer_name, layer_type, layer_blobs in iter:
        if layer_type == 'Convolution' or layer_type == 'InnerProduct' or layer_type == 4 or layer_type == 14 \
                or layer_type == 'PReLU':
            if layer_type == 'PReLU':
                assert (len(layer_blobs) == 1)
                wmat = layer_blobs[0].data
                weight_name = layer_name + '_gamma'
                arg_params[weight_name] = mx.nd.zeros(wmat.shape)
                arg_params[weight_name][:] = wmat
                continue
            wmat_dim = []
            if getattr(layer_blobs[0].shape, 'dim', None) is not None:
                if len(layer_blobs[0].shape.dim) > 0:
                    wmat_dim = layer_blobs[0].shape.dim
                else:
                    wmat_dim = [layer_blobs[0].num, layer_blobs[0].channels, layer_blobs[0].height, layer_blobs[0].width]
            else:
                wmat_dim = list(layer_blobs[0].shape)
            wmat = np.array(layer_blobs[0].data).reshape(wmat_dim)

            channels = wmat_dim[1]
            if channels == 3 or channels == 4:  # RGB or RGBA
                if first_conv:
                    print ('Swapping BGR of caffe into RGB in mxnet')
                    wmat[:, [0, 2], :, :] = wmat[:, [2, 0], :, :]

            assert(wmat.flags['C_CONTIGUOUS'] is True)
            sys.stdout.write('converting layer {0}, wmat shape = {1}'.format(layer_name, wmat.shape))
            if len(layer_blobs) == 2:
                bias = np.array(layer_blobs[1].data)
                bias = bias.reshape((bias.shape[0], 1))
                assert(bias.flags['C_CONTIGUOUS'] is True)
                bias_name = layer_name + "_bias"
                bias = bias.reshape(arg_shape_dic[bias_name])
                arg_params[bias_name] = mx.nd.zeros(bias.shape)
                arg_params[bias_name][:] = bias
                sys.stdout.write(', bias shape = {}'.format(bias.shape))

            sys.stdout.write('\n')
            sys.stdout.flush()
            wmat = wmat.reshape((wmat.shape[0], -1))
            weight_name = layer_name + "_weight"

            if weight_name not in arg_shape_dic:
                print(weight_name + ' not found in arg_shape_dic.')
                continue
            print(arg_shape_dic[weight_name])
            wmat = wmat.reshape(arg_shape_dic[weight_name])
            arg_params[weight_name] = mx.nd.zeros(wmat.shape)
            arg_params[weight_name][:] = wmat


            if first_conv and (layer_type == 'Convolution' or layer_type == 4):
                first_conv = False

        elif layer_type == 'Scale':
            bn_name = layer_name.replace('scale', 'bn')
            gamma = layer_blobs[0].data
            beta = layer_blobs[1].data
            # beta = np.expand_dims(beta, 1)
            beta_name = '{}_beta'.format(bn_name)
            gamma_name = '{}_gamma'.format(bn_name)

            beta = beta.reshape(arg_shape_dic[beta_name])
            gamma = gamma.reshape(arg_shape_dic[gamma_name])
            arg_params[beta_name] = mx.nd.zeros(beta.shape)
            arg_params[gamma_name] = mx.nd.zeros(gamma.shape)
            arg_params[beta_name][:] = beta
            arg_params[gamma_name][:] = gamma

            assert gamma.flags['C_CONTIGUOUS'] is True
            assert beta.flags['C_CONTIGUOUS'] is True
            print ('converting scale layer, beta shape = {}, gamma shape = {}'.format(beta.shape, gamma.shape))
        elif layer_type == 'BatchNorm':
            bn_name = layer_name
            mean = layer_blobs[0].data
            var = layer_blobs[1].data
            moving_average_factor = layer_blobs[2].data
            mean_name = '{}_moving_mean'.format(bn_name)
            var_name = '{}_moving_var'.format(bn_name)
            maf_name = '{}_momentum'.format(bn_name)
            mean = mean.reshape(aux_shape_dic[mean_name])
            var = var.reshape(aux_shape_dic[var_name])
            aux_params[mean_name] = mx.nd.zeros(mean.shape)
            aux_params[var_name] = mx.nd.zeros(var.shape)
            arg_params[maf_name] = mx.nd.zeros(moving_average_factor.shape)
            aux_params[mean_name][:] = mean
            aux_params[var_name][:] = var
            arg_params[maf_name][:] = moving_average_factor
            assert var.flags['C_CONTIGUOUS'] is True
            assert mean.flags['C_CONTIGUOUS'] is True
            print ('converting batchnorm layer, mean shape = {}, var shape = {}'.format(mean.shape, var.shape))
        else:
            assert len(layer_blobs) == 0
            print ('\tskipping layer {} of type {}'.format(layer_name, layer_type))
    return prob, arg_params, aux_params, input_dim




if __name__ == '__main__':
    main()
