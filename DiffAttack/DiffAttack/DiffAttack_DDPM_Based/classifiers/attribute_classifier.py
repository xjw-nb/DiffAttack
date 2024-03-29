# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import os
from . import attribute_net

softmax = torch.nn.Softmax(dim=1)

#对于图像进行下采样操作，减小图像分辨率以节省资源
def downsample(images, size=256):
    # Downsample to 256x256. The attribute classifiers were built for 256x256.
    # follows https://github.com/NVlabs/stylegan/blob/master/metrics/linear_separability.py#L127
    if images.shape[2] > size:
        factor = images.shape[2] // size #计算缩放的比例因子
        assert (factor * size == images.shape[2])#赋值缩放后的图像，确保图像尺寸为整数
        #images.view()常用来改变张量的形状，-1表示当前维度的大小根据其他维度大小自动计算
        images = images.view(
            [-1, images.shape[1], images.shape[2] // factor, factor, images.shape[3] // factor, factor])
        #沿着3，5两个维度进行均值计算
        images = images.mean(dim=[3, 5])
        return images
    else:
        assert (images.shape[-1] == 256)
        return images


def get_logit(net, im):
    im_256 = downsample(im)
    logit = net(im_256)
    return logit


def get_softmaxed(net, im):
    logit = get_logit(net, im)
    logits = torch.cat([logit, -logit], dim=1)
    softmaxed = softmax(torch.cat([logit, -logit], dim=1))[:, 1]
    return logits, softmaxed


def load_attribute_classifier(attribute, ckpt_path=None):
    if ckpt_path is None:
        base_path = 'pretrained/celebahq'
        attribute_pkl = os.path.join(base_path, attribute, 'net_best.pth')
        ckpt = torch.load(attribute_pkl)
    else:
        ckpt = torch.load(ckpt_path)
    print("Using classifier at epoch: %d" % ckpt['epoch'])
    if 'valacc' in ckpt.keys():
        print("Validation acc on raw images: %0.5f" % ckpt['valacc'])
    detector = attribute_net.from_state_dict(
        ckpt['state_dict'], fixed_size=True, use_mbstd=False).cuda().eval()
    return detector


class ClassifierWrapper(torch.nn.Module):
    def __init__(self, classifier_name, ckpt_path=None, device='cuda'):
        super(ClassifierWrapper, self).__init__()
        self.net = load_attribute_classifier(classifier_name, ckpt_path).eval().to(device)

    def forward(self, ims):
        out = (ims - 0.5) / 0.5
        return get_softmaxed(self.net, out)[0]
