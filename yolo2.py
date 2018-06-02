import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon import Block, HybridBlock
from mxnet.gluon.model_zoo import vision














pretrained = vision.get_model('resnet18_v1', pretrained=True).features