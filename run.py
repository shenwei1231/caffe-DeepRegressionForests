import sys, os, re, urllib
sys.path.insert(0, 'caffe-drf/python')
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.coord_map import crop
from caffe.proto import caffe_pb2
from os.path import join, splitext, abspath, exists, dirname, isdir, isfile
from datetime import datetime
from scipy.io import savemat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
tmp_dir = 'tmp'
tmp_dir = join(dirname(__file__), tmp_dir)
if not isdir(tmp_dir):
  os.makedirs(tmp_dir)

parser = argparse.ArgumentParser(description='DRF')
parser.add_argument('--gpu', type=int, required=False, default=3)
parser.add_argument('--data', type=str, required=False, default='Morph')
parser.add_argument('--tree', type=int, required=False, default=5)
parser.add_argument('--depth', type=int, required=False, default=6)
parser.add_argument('--drop', type=bool, required=False, default=False)
parser.add_argument('--nout', type=int, required=False, default=128)
parser.add_argument('--init', type=str, required=False, default='init')
parser.add_argument('--save', type=str, required=False, default='model')
parser.add_argument('--cs_l', type=int, required=False, default=5)
parser.add_argument('--model', type=str, required=False, default='./VGG_ILSVRC_16_layers.caffemodel')
parser.add_argument('--traintxt', type=str, required=False, default='./train.txt')
parser.add_argument('--testtxt', type=str, required=False, default='./test.txt')
args=parser.parse_args()

testdir = './data/morph/'  #directory for test dataset
traindir = './data/morph/' #directory for train dataset

with open(args.traintxt,'r') as f:
  nTrain = len(f.readlines())
with open(args.testtxt,'r') as f:
  nTest = len(f.readlines())

# some useful options ##
ntree = args.tree      #
treeDepth = args.depth #
maxIter = 30000        #
test_interval = 10000  #
test_batch_size = 15   #
train_batch_size = 16  #
########################

test_iter = int(np.ceil(nTest / test_batch_size))

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, mult=[1,1,2,0]):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, 
        weight_filler=dict(type='gaussian', std=0.005),
        param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def make_net(phase='train'):
  n = caffe.NetSpec()
  if phase=='train':
      batch_size = train_batch_size
      n.data, n.label = L.ImageData(ntop=2,image_data_param=dict(source=args.traintxt,root_folder=traindir,shuffle=True,batch_size=batch_size,new_height=256,new_width=256),
            transform_param=dict(mirror=True,mean_value=112,crop_size=224))
  elif phase=='test':
      batch_size = test_batch_size
      n.data, n.label = L.ImageData(ntop=2,image_data_param=dict(source=args.testtxt,root_folder=testdir,batch_size=batch_size),
            transform_param=dict(mean_value=112,crop_size=224))
  if phase == 'deploy':
      n.data = L.Input(shape=dict(dim=[1,3,256,256]))

  n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, mult=[10,10,20,0])
  n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, mult=[10,10,20,0])
  n.pool1 = max_pool(n.relu1_2)

  n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, mult=[10,1,20,0])
  n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, mult=[10,1,20,0])
  n.pool2 = max_pool(n.relu2_2)

  n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, mult=[10,1,20,0])
  n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, mult=[10,1,20,0])
  n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, mult=[10,1,20,0])
  n.pool3 = max_pool(n.relu3_3)

  n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
  n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
  n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
  n.pool4 = max_pool(n.relu4_3)
  
  n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
  n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
  n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
  n.pool5 = max_pool(n.relu5_3)

  n.fc6 = L.InnerProduct(n.pool5, num_output=4096, bias_term=True, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0),
            param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
  n.relu6 = L.ReLU(n.fc6, in_place=True)
  n.drop6 = L.Dropout(n.relu6, in_place=True, dropout_ratio=0.5)

  n.fc7 = L.InnerProduct(n.drop6, num_output=4096, bias_term=True, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0),
            param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
  n.relu7 = L.ReLU(n.fc7, in_place=True)
  n.drop7 = L.Dropout(n.relu7, in_place=True, dropout_ratio=0.5)

  if args.nout > 0:
    assert(args.nout >= int(pow(2, treeDepth - 1) - 1))
    nout = args.nout
  else:
    if ntree == 1:
      nout = int(pow(2, treeDepth - 1) - 1)
    else:
      nout = int((pow(2, treeDepth - 1) - 1) * ntree * 2 / 3)
  n.fc8 = L.InnerProduct(n.drop7, num_output=nout, bias_term=True, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0),
            param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], name='fc8-101')#name='fc8a')

  if phase=='train':
    all_data_vec_length = int(50)
    #all_data_vec_length = int(nTrain / train_batch_size)
    n.loss = L.NeuralDecisionRegForestWithLoss(n.fc8, n.label, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)], 
        neural_decision_forest_param=dict(depth=treeDepth, num_trees=ntree, num_classes=1, iter_times_class_label_distr=20, 
            iter_times_in_epoch=50, all_data_vec_length=all_data_vec_length, drop_out=args.drop, init_filename=args.init,record_filename='F_morph.record'), name='probloss1')
  elif phase=='test':
    n.pred = L.NeuralDecisionRegForest(n.fc8, n.label, neural_decision_forest_param=dict(depth=treeDepth, num_trees=ntree, num_classes=1), name='probloss1')
    n.MAE = L.MAE(n.pred, n.label)
    n.CS5 = L.CS(n.pred, n.label,cs_param = dict(lll = args.cs_l))
  elif phase=='deploy':
    n.pred = L.NeuralDecisionRegForest(n.fc8, neural_decision_forest_param=dict(depth=treeDepth, num_trees=ntree, num_classes=1), name='probloss1')
  return n.to_proto()

def make_solver():
  s = caffe_pb2.SolverParameter()
  s.type = 'SGD'
  s.display = 10
  s.base_lr = 0.05
  s.lr_policy = "step"
  s.gamma = 0.5
  s.momentum = 0.9
  s.stepsize = 10000
  s.max_iter = maxIter
  s.snapshot = 10000
  snapshot_prefix = join(dirname(__file__), args.save)
  if not isdir(snapshot_prefix):
    os.makedirs(snapshot_prefix)
  s.snapshot_prefix = join(snapshot_prefix, args.data)
  s.train_net = join(tmp_dir, args.data + '-train' + '.prototxt')
  s.test_net.append(join(tmp_dir, args.data + '-test' + '.prototxt'))
  s.test_interval = 100 # will test mannualy
  s.test_iter.append(test_iter)
  s.test_initialization = True
  return s

if __name__ == '__main__':
    print args
    # write training/testing nets and solver
    with open(join(tmp_dir, args.data + '-train'  + '.prototxt'), 'w') as f:
      f.write(str(make_net()))
    with open(join(tmp_dir, args.data + '-test'  + '.prototxt'), 'w') as f:
      f.write(str(make_net('test')))
    with open(join(tmp_dir, args.data + '-deploy'  + '.prototxt'), 'w') as f:
      f.write(str(make_net('deploy')))
    with open(join(tmp_dir, args.data + '-solver' + '.prototxt'), 'w') as f:
      f.write(str(make_solver()))
    if args.gpu<0:    
      caffe.set_mode_cpu()
    else:
      caffe.set_mode_gpu()
      caffe.set_device(args.gpu)
    iter = 0
    mae = []
    cs = []
    solver = caffe.SGDSolver(join(tmp_dir, args.data + '-solver' + '.prototxt'))
    base_weights = join(args.model)
    if not isfile(base_weights):
      print "There is not base model to %s"%(base_weights)
    solver.net.copy_from(base_weights)
    print "Summarize of net parameters:"
    for p in solver.net.params:
      param = solver.net.params[p][0].data[...]
      print "  layer \"%s\":, parameter[0] mean=%f, std=%f"%(p, param.mean(), param.std())
    print args
    raw_input("Press Enter to continue...")
    while iter < maxIter:
      solver.step(test_interval)
      solver.test_nets[0].share_with(solver.net)
      mae1 = np.float32(0.0)
      cs1 = np.float32(0.0)
      for t in range(test_iter):
          output= solver.test_nets[0].forward()
          mae1 += output['MAE']
          cs1 += output['CS5']
      mae1 /= test_iter
      cs1 /= test_iter
      mae.append(mae1)
      cs.append(cs1)
      iter = iter + test_interval
      print args
      print "Iter%d, currentMAE=%.4f, bestMAE=%.4f, currentCS=%.4f, bestCS=%.4f"%(iter, mae[-1], min(mae),cs[-1],max(cs))
    mae = np.array(mae, dtype=np.float32)
    cs = np.array(cs, dtype=np.float32)
    sav_fn = join(tmp_dir, "MAE-%dtree%ddepth%dtime%s"%(
            args.data, ntree, treeDepth, datetime.now().strftime("M%mD%d-H%HM%MS%S")))
    np.save(sav_fn+'.npy', mae)
    mat_dict = dict({'mae':mae,'cs':cs})
    mat_dict.update(vars(args))  # save args to .mat
    savemat(sav_fn+'.mat', mat_dict)
    print args
    print "Best MAE=%.4f, Best CS=%.4f."%(mae.min(),cs.max())
    print "Done! Results saved at \'"+sav_fn+"\'"
