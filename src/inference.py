import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from models.modnet import MODNet

if __name__ == '__main__':
    # define cmd arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help='path of input images', default="input")
    parser.add_argument('--output-path', type=str, help='path of output images', default="output")
    parser.add_argument('--ckpt-path', type=str, help='path of pre-trained MODNet',
                        default="pretrained\modnet_photographic_portrait_matting.ckpt")
    args = parser.parse_args()

    # check input arguments
    if not os.path.exists(args.input_path):
        print('Cannot find input path: {0}'.format(args.input_path))
        exit()
    if not os.path.exists(args.output_path):
        print('Cannot find output path: {0}'.format(args.output_path))
        exit()
    if not os.path.exists(args.ckpt_path):
        print('Cannot find ckpt path: {0}'.format(args.ckpt_path))
        exit()
    # define hyper-parameters
    ref_size = 512
    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet).cuda()
    modnet.load_state_dict(torch.load(args.ckpt_path))
    modnet.eval()
    # 注：程序中的数字仅表示某张输入图片尺寸，如1080x1440，此处只为记住其转换过程。
    # inference images
    im_names = os.listdir(args.input_path)
    for im_name in im_names:
        print('Process image: {0}'.format(im_name))
        # read image
        im = Image.open(os.path.join(args.input_path, im_name))
        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]
        im_org = im  # 保存numpy原始数组 (1080,1440,3)
        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)
        # add mini-batch dim
        im = im[None, :, :, :]
        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda(), True)  # 从模型获得的 matte ([1,1,512, 672])

        # resize and save matte，foreground picture
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')  # 内插，扩展到([1,1,1080,1440])  范围[0,1]
        matte = matte[0][0].data.cpu().numpy()  # torch 张量转换成numpy (1080, 1440)
        matte_name = im_name.split('.')[0] + '_matte.png'
        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(args.output_path, matte_name))
        matte_org = np.repeat(np.asarray(matte)[:, :, None], 3, axis=2)  # 扩展到 (1080, 1440, 3) 以便和im_org计算

        # img = np.zeros([400, 400, 3], np.uint8)
        img = np.zeros(im_org.shape, np.uint8)

        img[:, :, 0] = np.ones([img.shape[0],img.shape[1]]) * 255  # 0修改第一个通道值使第一个通道值全为255，出现一个蓝色的图片
        # foreground = im_org * matte_org + np.full(im_org.shape, 255) * (1 - matte_org)  # 计算前景，获得抠像
        foreground = im_org * matte_org + img* (1 - matte_org)  # 计算前景，获得抠像
        fg_name = im_name.split('.')[0] + '_fg.png'
        Image.fromarray(((foreground).astype('uint8')), mode='RGB').save(os.path.join(args.output_path, fg_name))
