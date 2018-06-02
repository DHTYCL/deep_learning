from mxnet import gluon

root_url = ('https://apache-mxnet.s3-accelerate.amazonaws.com/'
            'gluon/dataset/pikachu/')
data_dir = '../data/pikachu/'
dataset = {'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
          'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
          'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'}
for k, v in dataset.items():
    gluon.utils.download(root_url+k, data_dir+k, sha1_hash=v)

from mxnet import image
from mxnet import nd

data_shape = 256
batch_size = 32
rgb_mean = nd.array([123, 117, 104])

def get_iterators(data_shape, batch_size):
    class_names = ['pikachu']
    num_class = len(class_names)
    train_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec=data_dir+'train.rec',
        path_imgidx=data_dir+'train.idx',
        shuffle=True,
        mean=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
    val_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec=data_dir+'val.rec',
        shuffle=False,
        mean=True)
    return train_iter, val_iter, class_names, num_class

train_data, test_data, class_names, num_class = get_iterators(data_shape, batch_size)
batch = train_data.next()
#print batch

import matplotlib as mpl
mpl.rcParams['figure.dpi']= 120
import matplotlib.pyplot as plt

def box_to_rect(box, color, linewidth=3):
    """convert an anchor box to a matplotlib rectangle"""
    box = box.asnumpy()
    return plt.Rectangle(
        (box[0], box[1]), box[2]-box[0], box[3]-box[1],
        fill=False, edgecolor=color, linewidth=linewidth)

_, figs = plt.subplots(3, 3, figsize=(6,6))
for i in range(3):
    for j in range(3):
        img, labels = batch.data[0][3*i+j], batch.label[0][3*i+j]
        # (3L, 256L, 256L) => (256L, 256L, 3L)
        img = img.transpose((1, 2, 0)) + rgb_mean
        img = img.clip(0,255).asnumpy()/255
        fig = figs[i][j]
        fig.imshow(img)
        for label in labels:
            rect = box_to_rect(label[1:5]*data_shape,'red',2)
            fig.add_patch(rect)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
#plt.show()


from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior

# shape: batch x channel x height x weight
n = 40
x = nd.random.uniform(shape=(1, 3, n, n))

y = MultiBoxPrior(x, sizes=[.5,.25,.1], ratios=[1,2,.5])

boxes = y.reshape((n, n, -1, 4))
#print(boxes.shape)

# # The first anchor box centered on (20, 20)
# # its format is (x_min, y_min, x_max, y_max)
#print boxes[20, 20, 0, :]


colors = ['blue', 'green', 'red', 'black', 'magenta']

#plt.imshow(nd.ones((n, n, 3)).asnumpy())
anchors = boxes[20, 20, :, :]
for i in range(anchors.shape[0]):
    plt.gca().add_patch(box_to_rect(anchors[i,:]*n, colors[i]))
#plt.show()

from mxnet.gluon import nn
def class_predictor(num_anchors, num_classes):
    """return a layer to predict classes"""
    return nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)

cls_pred = class_predictor(5, 10)
cls_pred.initialize()
x = nd.zeros((2, 3, 20, 20))
y = cls_pred(x)
#print y.shape


def box_predictor(num_anchors):
    """return a layer to predict delta locations"""
    return nn.Conv2D(num_anchors * 4, 3, padding=1)

box_pred = box_predictor(10)
box_pred.initialize()
x = nd.zeros((2, 3, 20, 20))
y = box_pred(x)
#print y.shape

def down_sample(num_filters):
    """stack two Conv-BatchNorm-Relu blocks and then a pooling layer
    to halve the feature size"""
    out = nn.HybridSequential()
    for _ in range(2):
        out.add(nn.Conv2D(num_filters, 3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filters))
        out.add(nn.Activation('relu'))
    out.add(nn.MaxPool2D(2))
    return out

blk = down_sample(10)
blk.initialize()
x = nd.zeros((2, 3, 20, 20))
y = blk(x)
#print y.shape
#
#
# x = nd.zeros((2, 8, 20, 20))
#
#
# cls_pred1 = class_predictor(5, 10)
# cls_pred1.initialize()
# y1 = cls_pred1(x)
#
#
# ds = down_sample(16)
# ds.initialize()
# x = ds(x)
#
#
# cls_pred2 = class_predictor(3, 10)
# cls_pred2.initialize()
# y2 = cls_pred2(x)
#
# def flatten_prediction(pred):
#     return pred.transpose(axes=(0,2,3,1)).flatten()
#
# def concat_predictions(preds):
#     return nd.concat(*preds, dim=1)
#
# flat_y1 = flatten_prediction(y1)
#
# flat_y2 = flatten_prediction(y2)
#
# y = concat_predictions([flat_y1, flat_y2])
#
# def body():
#     out = nn.HybridSequential()
#     for nfilters in [16, 32, 64]:
#         out.add(down_sample(nfilters))
#     return out
#
# bnet = body()
# bnet.initialize()
# x = nd.random.uniform(shape=(2,3,256,256))
# y = bnet(x)
#
#
# def toy_ssd_model(num_anchors, num_classes):
#     downsamplers = nn.Sequential()
#     for _ in range(3):
#         downsamplers.add(down_sample(128))
#
#     class_predictors = nn.Sequential()
#     box_predictors = nn.Sequential()
#     for _ in range(5):
#         class_predictors.add(class_predictor(num_anchors, num_classes))
#         box_predictors.add(box_predictor(num_anchors))
#
#     model = nn.Sequential()
#     model.add(body(), downsamplers, class_predictors, box_predictors)
#     return model
#
#
# def toy_ssd_forward(x, model, sizes, ratios, verbose=False):
#     body, downsamplers, class_predictors, box_predictors = model
#     anchors, class_preds, box_preds = [], [], []
#     # feature extraction
#     x = body(x)
#     for i in range(5):
#         # predict
#         anchors.append(MultiBoxPrior(
#             x, sizes=sizes[i], ratios=ratios[i]))
#         class_preds.append(
#             flatten_prediction(class_predictors[i](x)))
#         box_preds.append(
#             flatten_prediction(box_predictors[i](x)))
#         if verbose:
#             print('Predict scale', i, x.shape, 'with',
#                   anchors[-1].shape[1], 'anchors')
#         # down sample
#         if i < 3:
#             x = downsamplers[i](x)
#         elif i == 3:
#             x = nd.Pooling(
#                 x, global_pool=True, pool_type='max',
#                 kernel=(x.shape[2], x.shape[3]))
#     # concat data
#     return (concat_predictions(anchors),
#             concat_predictions(class_preds),
#             concat_predictions(box_preds))
#
#
# from mxnet import gluon
# class ToySSD(gluon.Block):
#     def __init__(self, num_classes, verbose=False, **kwargs):
#         super(ToySSD, self).__init__(**kwargs)
#         # anchor box sizes and ratios for 5 feature scales
#         self.sizes = [[.2,.272], [.37,.447], [.54,.619],
#                       [.71,.79], [.88,.961]]
#         self.ratios = [[1,2,.5]]*5
#         self.num_classes = num_classes
#         self.verbose = verbose
#         num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
#         # use name_scope to guard the names
#         with self.name_scope():
#             self.model = toy_ssd_model(num_anchors, num_classes)
#
#     def forward(self, x):
#         anchors, class_preds, box_preds = toy_ssd_forward(
#             x, self.model, self.sizes, self.ratios,
#             verbose=self.verbose)
#         # it is better to have class predictions reshaped for softmax computation
#         class_preds = class_preds.reshape(shape=(0, -1, self.num_classes+1))
#         return anchors, class_preds, box_preds
#
#
# net = ToySSD(num_classes=2, verbose=True)
# net.initialize()
# x = batch.data[0][0:1]
#
# anchors, class_preds, box_preds = net(x)
#
# from mxnet.contrib.ndarray import MultiBoxTarget
# def training_targets(anchors, class_preds, labels):
#     class_preds = class_preds.transpose(axes=(0,2,1))
#     return MultiBoxTarget(anchors, labels, class_preds)
#
# out = training_targets(anchors, class_preds, batch.label[0][0:1])
#
# import numpy as np
#
# def focal_loss(gamma, x):
#     return - (1-x)**gamma*np.log(x)
#
# x = np.arange(0.01, 1, .01)
# gammas = [0,.25,.5,1]
# for i,g in enumerate(gammas):
#     plt.plot(x, focal_loss(g,x), colors[i])
# class FocalLoss(gluon.loss.Loss):
#     def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
#         super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
#         self._axis = axis
#         self._alpha = alpha
#         self._gamma = gamma
#
#     def hybrid_forward(self, F, output, label):
#         output = F.softmax(output)
#         pj = output.pick(label, axis=self._axis, keepdims=True)
#         loss = - self._alpha * ((1 - pj) ** self._gamma) * pj.log()
#         return loss.mean(axis=self._batch_axis, exclude=True)
#
# cls_loss = FocalLoss()
#
# class SmoothL1Loss(gluon.loss.Loss):
#     def __init__(self, batch_axis=0, **kwargs):
#         super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)
#
#     def hybrid_forward(self, F, output, label, mask):
#         loss = F.smooth_l1((output - label) * mask, scalar=1.0)
#         return loss.mean(self._batch_axis, exclude=True)
#
# box_loss = SmoothL1Loss()
#
# from mxnet import init
# from mxnet import gpu
#
# ctx = gpu(0)
# # the CUDA implementation requres each image has at least 3 lables.
# # Padd two -1 labels for each instance
# train_data.reshape(label_shape=(3, 5))
# train_data = test_data.sync_label_shape(train_data)
#
# net = ToySSD(num_class)
# net.initialize(init.Xavier(magnitude=2), ctx=ctx)
# #trainer = gluon.Trainer(net.collect_params(),
# #                        'sgd', {'learning_rate': 0.1, 'wd': 5e-4})