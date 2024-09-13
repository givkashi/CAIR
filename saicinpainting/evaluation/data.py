import glob
import os

import cv2
import PIL.Image as Image
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F


def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


def pad_tensor_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


class RetargetingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.png', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image(self.img_filenames[i], mode='RGB')
        mask = load_image(self.mask_filenames[i], mode='RGB')# must be L for binary mask
        result = dict(input_image=image, mask=mask[None, ...])

        if self.scale_factor is not None:
            result['input_image'] = scale_image(result['input_image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['input_image'] = pad_img_to_modulo(result['input_image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)
        # print(result["mask"].shape, result["input_image"].shape)
        result['mask'] = result['mask'].squeeze()

        return result

class RetargetingImageMaskTestDataset(Dataset):
    def __init__(self, datadir, img_suffix='.png', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '*mask*.png'), recursive=True)))
        self.source_filenames = sorted(list(glob.glob(os.path.join(self.datadir, '**', '**.jpg'), recursive=True)))
        self.input_img_filenames = [fname.rsplit('_mask', 1)[0] + img_suffix for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        image = load_image(self.input_img_filenames[i], mode='RGB')
        mask = load_image(self.mask_filenames[i], mode='RGB')
        path = self.source_filenames[i]
        print(path)
        source_img = cv2.imread(path)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
        new_image_width = 512
        new_image_height = 512
        x_center_o = (new_image_width - source_img.shape[1]) // 2
        y_center_o = (new_image_height - source_img.shape[0]) // 2
        boarders = [y_center_o, y_center_o + source_img.shape[0], x_center_o, x_center_o + source_img.shape[1]]
        result = dict(input_image=image, mask=mask[None, ...], boarders=boarders)

        if self.scale_factor is not None:
            result['input_image'] = scale_image(result['input_image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['input_image'] = pad_img_to_modulo(result['input_image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)
        # print(result["mask"].shape, result["input_image"].shape)
        result['mask'] = result['mask'].squeeze()

        return result

class OurRetargetingDataset(Dataset):
    def __init__(self, datadir, img_suffix='.jpg', pad_out_to_modulo=None, scale_factor=None):
        self.datadir = datadir
        self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, 'mask', '**', '*mask*.png'), recursive=True)))
        self.img_filenames = [os.path.join(self.datadir, 'img', os.path.basename(fname.rsplit('-', 1)[0].rsplit('_', 1)[0]) + '.png') for fname in self.mask_filenames]
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, i):
        result = dict(image=load_image(self.img_filenames[i], mode='RGB'),
                      mask=load_image(self.mask_filenames[i], mode='L')[None, ...])

        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['mask'] = scale_image(result['mask'], self.scale_factor)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)

        return result

class PrecomputedRetargetingResultsDataset(RetargetingDataset):
    def __init__(self, datadir, predictdir, inpainted_suffix='_inpainted.jpg', **kwargs):
        super().__init__(datadir, **kwargs)
        if not datadir.endswith('/'):
            datadir += '/'
        self.predictdir = predictdir
        self.pred_filenames = [os.path.join(predictdir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
                               for fname in self.mask_filenames]

    def __getitem__(self, i):
        result = super().__getitem__(i)
        result['inpainted'] = load_image(self.pred_filenames[i])
        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['inpainted'] = pad_img_to_modulo(result['inpainted'], self.pad_out_to_modulo)
        return result

class OurPrecomputedInpaintingResultsDataset(OurRetargetingDataset):
    def __init__(self, datadir, predictdir, inpainted_suffix="png", **kwargs):
        super().__init__(datadir, **kwargs)
        if not datadir.endswith('/'):
            datadir += '/'
        self.predictdir = predictdir
        self.pred_filenames = [os.path.join(predictdir, os.path.basename(os.path.splitext(fname)[0]) + f'_inpainted.{inpainted_suffix}')
                               for fname in self.mask_filenames]
        # self.pred_filenames = [os.path.join(predictdir, os.path.splitext(fname[len(datadir):])[0] + inpainted_suffix)
        #                        for fname in self.mask_filenames]

    def __getitem__(self, i):
        result = super().__getitem__(i)
        result['inpainted'] = self.file_loader(self.pred_filenames[i])

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['inpainted'] = pad_img_to_modulo(result['inpainted'], self.pad_out_to_modulo)
        return result


