import os
import subprocess

current_dir = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(current_dir + '/caffe_pb2.py'):
    ret = subprocess.call(['protoc', '-I=' + current_dir, '--python_out=' + current_dir, current_dir + '/caffe.proto'])
    if ret is not 0:
        raise NotImplementedError("`caffe_pb2.py` is missed in %s. We need to use `protoc -I=%s --python_out=%s %s/caffe.proto` "
                                  "to rebuild the protobuf. But we have failed to execute of this command. The google-protobuf may have not "
                                  "been installed properly" %(current_dir, current_dir, current_dir, current_dir))
