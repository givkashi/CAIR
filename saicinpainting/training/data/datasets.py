import glob
import logging
import os
import random
import pandas as pd
import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import webdataset
from omegaconf import open_dict, OmegaConf
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler, ConcatDataset
from torchvision import transforms
from saicinpainting.evaluation.data import RetargetingDataset as RetargetingEvaluationDataset, ceil_modulo, \
    RetargetingImageMaskTestDataset
from saicinpainting.training.data.aug import IAAAffine2, IAAPerspective2
from saicinpainting.training.data.masks import get_mask_generator

LOGGER = logging.getLogger(__name__)

class RetargetingTrainDataset(Dataset):
    def __init__(self, indir, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        print(indir)
        print(len(self.in_files))
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = img
        H_im = img.shape[0]
        W_im = img.shape[1]
        r = random.random()

        if random.random() > 0.5:
            c2 = np.random.rand(1)
            c2 = c2 * (0.5)
            c3 = np.random.rand(1)
            c3 = c3 * (0.5 - c2)
            g2 = np.random.rand(1)
            g2 = g2 * (0.5)
            g3 = np.random.rand(1)
            g3 = g3 * (0.5 - g2)
            a = H_im * (1 - c2)
            b = W_im * (1 - g2)
            img = center_crop(img, (b, a))

            a = (H_im / img.shape[0]) * (1 - c3 - c2)
            b = (W_im / img.shape[1]) * (1 - g3 - g2)
            img = scale_image(img, a, b)
        else:
            c2 = np.random.rand(1)
            c2 = c2 * (0.5)
            c3 = np.random.rand(1)
            c3 = c3 * (0.5 - c2)
            g2 = np.random.rand(1)
            g2 = g2 * (0.5)
            g3 = np.random.rand(1)
            g3 = g3 * (0.5 - g2)
            a = (H_im / img.shape[0]) * (1 - c2)
            b = (W_im / img.shape[1]) * (1 - g2)
            img = scale_image(img, a, b)
            a = H_im * (1 - c3 - c2)
            b = W_im * (1 - g3 - g2)
            img = center_crop(img, (b, a))

        old_image_height, old_image_width, channels = img.shape

        # create new image of desired size and color (blue) for padding
        new_image_width = 512
        new_image_height = 512
        color = (0, 0, 0)
        pad_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_original_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_mask = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        # compute center offset
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2
        x_center_o = (new_image_width - original_image.shape[1]) // 2
        y_center_o = (new_image_height - original_image.shape[0]) // 2

        # copy img image into center of result image
        pad_image[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img
        pad_original_image[y_center_o:y_center_o + original_image.shape[0], x_center_o:x_center_o + original_image.shape[1]] = original_image
        pad_mask[y_center_o:y_center_o + original_image.shape[0], x_center_o:x_center_o + original_image.shape[1]] = np.ones_like(original_image)
        boarders = [y_center_o, y_center_o + original_image.shape[0], x_center_o, x_center_o + original_image.shape[1]]

        pad_image = self.transform(image=pad_image)['image']
        pad_original_image = self.transform(image=pad_original_image)['image']
        pad_image = np.transpose(pad_image, (2, 0, 1))
        pad_original_image = np.transpose(pad_original_image, (2, 0, 1))
        mask = pad_mask[:, :, 0].astype("float32")
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        self.iter_i += 1

        return dict(input_image=pad_image,
                    GT=pad_original_image,
                    mask=mask,
                    boarders=boarders)

class RetargetingyoloDataset(Dataset):
    def __init__(self, indir, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        print(indir)
        print(len(self.in_files))
        self.transform = transform
        self.iter_i = 0
        self.yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=r'E:\yolov5s.pt').cpu()  # or yolov5m, yolov5l, yolov5x, custom


    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = img
        old_image_height, old_image_width, channels = original_image.shape
        hu = -1
        if random.random() > 0.5:
            with torch.no_grad():
                yolo = self.yolo(img).crop(save=False)
            if yolo:
                cord = yolo[0]["box"]
                a = np.random.randint(0, int(cord[1] - hu))
                b = np.random.randint(int(cord[3] + hu), old_image_height)
                c = np.random.randint(0, int(cord[0] - hu))
                d = np.random.randint(int(cord[2] + hu), old_image_width)
                img = img[a:b, c:d]
                r1 = (b - a) / old_image_height
                r2 = (d - c) / old_image_width
            else:
                r1 = 1
                r2 = 1
            crop_rate = max(r1, r2)
            if crop_rate > 0.5:
                resize_rate = 0.8 - random.random() * (0.5 - (crop_rate - 0.5) / crop_rate)
                img = scale_image(img, resize_rate, resize_rate)
        else:
            resize_rate = 0.8 - random.random() * (0.8 - 0.5)
            img = scale_image(img, resize_rate, resize_rate)
            if resize_rate > 0.5:
                with torch.no_grad():
                    yolo = self.yolo(img).crop(save=False)
                if yolo:
                    cord = yolo[0]["box"]
                    a = np.random.randint(0, int(cord[1] - hu))
                    b = np.random.randint(int(cord[3] + hu), img.shape[0])
                    c = np.random.randint(0, int(cord[0] - hu))
                    d = np.random.randint(int(cord[2] + hu), img.shape[1])
                    img = img[a:b, c:d]

        # create new image of desired size and color (blue) for padding
        new_image_width = 512
        new_image_height = 512
        color = (0, 0, 0)
        pad_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_original_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_mask = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        # compute center offset
        x_center = (new_image_width - img.shape[1]) // 2
        y_center = (new_image_height - img.shape[0]) // 2
        x_center_o = (new_image_width - original_image.shape[1]) // 2
        y_center_o = (new_image_height - original_image.shape[0]) // 2

        # copy img image into center of result image
        pad_image[y_center:y_center + img.shape[0], x_center:x_center + img.shape[1]] = img
        pad_original_image[y_center_o:y_center_o + original_image.shape[0], x_center_o:x_center_o + original_image.shape[1]] = original_image
        pad_mask[y_center_o:y_center_o + original_image.shape[0], x_center_o:x_center_o + original_image.shape[1]] = np.ones_like(original_image)
        boarders = [y_center_o, y_center_o + original_image.shape[0], x_center_o, x_center_o + original_image.shape[1]]

        pad_image = self.transform(image=pad_image)['image']
        pad_original_image = self.transform(image=pad_original_image)['image']
        pad_image = np.transpose(pad_image, (2, 0, 1))
        pad_original_image = np.transpose(pad_original_image, (2, 0, 1))
        mask = pad_mask[:, :, 0].astype("float32")
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        self.iter_i += 1
        return dict(input_image=pad_image,
                    GT=pad_original_image,
                    mask=mask,
                    boarders=boarders)

class RetargetingTrainDataset(Dataset):
    def __init__(self, indir, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        print(indir)
        print(len(self.in_files))
        self.transform = transform
        self.iter_i = 0
        self.yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=r'yolov5s.pt').cpu()

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = img
        # img = cv2.resize(img, (self.out_size, self.out_size))

        H_im = img.shape[0]
        W_im = img.shape[1]
        rr = max(H_im, W_im)
        if rr > 320:
            img = scale_image(img, 320 / rr, 320 / rr)
        hu = -1
        if random.random() > 0.5:
            with torch.no_grad():
                yolo = self.yolo(img).crop(save=False)
            if yolo:
                cord = yolo[0]["box"]
                a = np.random.randint(0, int(cord[1] - hu))
                b = np.random.randint(int(cord[3] + hu), H_im)
                c = np.random.randint(0, int(cord[0] - hu))
                d = np.random.randint(int(cord[2] + hu), W_im)
                img = img[a:b, c:d]
                r1 = (b - a) / H_im
                r2 = (d - c) / W_im
            else:
                r1 = 1
                r2 = 1
            crop_rate = max(r1, r2)
            if crop_rate > 0.5:
                resize_rate = 0.8 - random.random() * (0.5 - (crop_rate - 0.5) / crop_rate)
                img = scale_image(img, resize_rate, resize_rate)
        else:
            resize_rate = 0.8 - random.random() * (0.8 - 0.5)
            img = scale_image(img, resize_rate, resize_rate)
            if resize_rate > 0.5:
                with torch.no_grad():
                    yolo = self.yolo(img).crop(save=False)
                if yolo:
                    cord = yolo[0]["box"]
                    a = np.random.randint(0, int(cord[1] - hu))
                    b = np.random.randint(int(cord[3] + hu), img.shape[0])
                    c = np.random.randint(0, int(cord[0] - hu))
                    d = np.random.randint(int(cord[2] + hu), img.shape[1])
                    img = img[a:b, c:d]


        old_image_height, old_image_width, channels = img.shape
        # create new image of desired size and color (blue) for padding
        new_image_width = 512
        new_image_height = 512
        new_image_width_mask = 256
        new_image_height_mask = 256
        color = (0, 0, 0)
        pad_image = np.full((new_image_height_mask, new_image_width_mask, channels), color, dtype=np.uint8)
        pad_image_1024 = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_original_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_mask = np.full((new_image_height_mask, new_image_width_mask, channels), color, dtype=np.uint8)
        # compute center offset
        x_center = (new_image_width_mask - old_image_width) // 2
        y_center = (new_image_width_mask - old_image_height) // 2
        x_center_o = (new_image_width - original_image.shape[1]) // 2
        y_center_o = (new_image_height - original_image.shape[0]) // 2
        x_center_h = (new_image_width - old_image_width) // 2
        y_center_h = (new_image_height - old_image_height) // 2

        mask_width = original_image.shape[1] // 2
        mask_height = original_image.shape[0] // 2
        x_center_m = (new_image_width_mask - mask_width) // 2
        y_center_m = (new_image_height_mask - mask_height) // 2
        # copy img image into center of result image
        pad_image[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img
        pad_image_1024[y_center_h:y_center_h + old_image_height, x_center_h:x_center_h + old_image_width] = img
        pad_original_image[y_center_o:y_center_o + original_image.shape[0], x_center_o:x_center_o + original_image.shape[1]] = original_image
        pad_mask[y_center_m:y_center_m + mask_height, x_center_m:x_center_m + mask_width] = np.ones_like((mask_width, mask_height, 3))
        boarders = [y_center_m, y_center_m + mask_height, x_center_m, x_center_m + mask_width]

        pad_image = self.transform(image=pad_image)['image']
        pad_image_1024 = self.transform(image=pad_image_1024)['image']
        pad_original_image = self.transform(image=pad_original_image)['image']
        pad_image = np.transpose(pad_image, (2, 0, 1))
        pad_image_1024 = np.transpose(pad_image_1024, (2, 0, 1))
        pad_original_image = np.transpose(pad_original_image, (2, 0, 1))
        mask = pad_mask[:, :, 0].astype("float32")
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        self.iter_i += 1
        return dict(input_image=pad_image,
                    GT=pad_original_image,
                    mask=mask,
                    boarders=boarders,
                    pad_image_1024=pad_image_1024)

class RetargetingTrainDatasetMaskWithObject(Dataset):
    def __init__(self, indir, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        self.transform = transform
        self.iter_i = 0
        self.yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt').cpu()
        self.deepl = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True).cpu()
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])
    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = img

        H_im = img.shape[0]
        W_im = img.shape[1]
        rr = max(H_im, W_im)
        if rr > 320:
            img = scale_image(img, 320 / rr, 320 / rr)
        hu = -1
        # t1 = time.time()
        if random.random() > 0.5:
            with torch.no_grad():
                yolo = self.yolo(img).crop(save=False)
            if yolo:
                cord = yolo[0]["box"]
                a = np.random.randint(0, int(cord[1] - hu))
                b = np.random.randint(int(cord[3] + hu), H_im)
                c = np.random.randint(0, int(cord[0] - hu))
                d = np.random.randint(int(cord[2] + hu), W_im)
                img = img[a:b, c:d]
                r1 = (b - a) / H_im
                r2 = (d - c) / W_im
            else:
                r1 = 1
                r2 = 1
            crop_rate = max(r1, r2)
            if crop_rate > 0.5:
                resize_rate = 0.8 - random.random() * (0.5 - (crop_rate - 0.5) / crop_rate)
                img = scale_image(img, resize_rate, resize_rate)
        else:
            resize_rate = 0.8 - random.random() * (0.8 - 0.5)
            img = scale_image(img, resize_rate, resize_rate)
            if resize_rate > 0.5:
                with torch.no_grad():
                    yolo = self.yolo(img).crop(save=False)
                if yolo:
                    cord = yolo[0]["box"]
                    a = np.random.randint(0, int(cord[1] - hu))
                    b = np.random.randint(int(cord[3] + hu), img.shape[0])
                    c = np.random.randint(0, int(cord[0] - hu))
                    d = np.random.randint(int(cord[2] + hu), img.shape[1])
                    img = img[a:b, c:d]

        org_img = Image.fromarray((original_image).astype("uint8"))
        input_tensor = self.preprocess(org_img)

        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
        self.deepl.eval()
        with torch.no_grad():
            output = self.deepl(input_batch)['out'][0]
        output_predictions = output.argmax(0)
        output_predictions = (output_predictions == 3).float()
        mask_rev = np.repeat(np.expand_dims(((output_predictions == 0) * 255).numpy().astype("uint8"), axis=-1), 3, axis=-1)
        output_predictions = output_predictions * transforms.ToTensor()(original_image) * 255
        mask = output_predictions.permute(1, 2, 0).numpy().astype("uint8")
        mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
        mask = mask + mask_rev

        old_image_height, old_image_width, channels = img.shape
        # create new image of desired size and color (blue) for padding
        new_image_width = 512
        new_image_height = 512
        new_image_width_mask = 512
        new_image_height_mask = 512
        color = (0, 0, 0)
        pad_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_original_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_mask = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        # compute center offset
        x_center = (new_image_width_mask - old_image_width) // 2
        y_center = (new_image_width_mask - old_image_height) // 2
        x_center_o = (new_image_width - original_image.shape[1]) // 2
        y_center_o = (new_image_height - original_image.shape[0]) // 2
        x_center_h = (new_image_width - old_image_width) // 2
        y_center_h = (new_image_height - old_image_height) // 2

        mask_width = original_image.shape[1] // 2
        mask_height = original_image.shape[0] // 2
        x_center_m = (new_image_width_mask - mask_width) // 2
        y_center_m = (new_image_height_mask - mask_height) // 2
        # copy img image into center of result image
        pad_image[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img
        pad_original_image[y_center_o:y_center_o + original_image.shape[0], x_center_o:x_center_o + original_image.shape[1]] = original_image
        pad_mask[y_center_o:y_center_o + original_image.shape[0], x_center_o:x_center_o + original_image.shape[1]] = mask
        boarders = [y_center_m, y_center_m + mask_height, x_center_m, x_center_m + mask_width]

        pad_image = self.transform(image=pad_image)['image']
        pad_original_image = self.transform(image=pad_original_image)['image']
        pad_mask = self.transform(image=pad_mask)['image']
        pad_image = np.transpose(pad_image, (2, 0, 1))
        pad_original_image = np.transpose(pad_original_image, (2, 0, 1))
        pad_mask = np.transpose(pad_mask, (2, 0, 1))

        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        self.iter_i += 1
        return dict(input_image=pad_image,
                    GT=pad_original_image,
                    mask=pad_mask,
                    boarders=boarders)


class RetargetingTestDataset(Dataset):
    def __init__(self, indir, transform):
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        print(indir)
        print(len(self.in_files))
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = img
        H_im = img.shape[0]
        W_im = img.shape[1]

        if random.random() > 0.5:
            c2 = np.random.rand(1)
            c2 = c2 * (0.5)
            c3 = np.random.rand(1)
            c3 = c3 * (0.5 - c2)
            g2 = np.random.rand(1)
            g2 = g2 * (0.5)
            g3 = np.random.rand(1)
            g3 = g3 * (0.5 - g2)
            a = H_im * (1 - c2)
            b = W_im * (1 - g2)
            img = center_crop(img, (b, a))
            a = (H_im / img.shape[0]) * (1 - c3 - c2)
            b = (W_im / img.shape[1]) * (1 - g3 - g2)
            img = scale_image(img, a, b)
        else:
            c2 = np.random.rand(1)
            c2 = c2 * (0.5)
            c3 = np.random.rand(1)
            c3 = c3 * (0.5 - c2)
            g2 = np.random.rand(1)
            g2 = g2 * (0.5)
            g3 = np.random.rand(1)
            g3 = g3 * (0.5 - g2)
            a = (H_im / img.shape[0]) * (1 - c2)
            b = (W_im / img.shape[1]) * (1 - g2)
            img = scale_image(img, a, b)
            a = H_im * (1 - c3 - c2)
            b = W_im * (1 - g3 - g2)
            img = center_crop(img, (b, a))

        old_image_height, old_image_width, channels = img.shape

        # create new image of desired size and color (blue) for padding
        new_image_width = 512
        new_image_height = 512
        color = (0, 0, 0)
        pad_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_original_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_mask = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        # compute center offset
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2
        x_center_o = (new_image_width - original_image.shape[1]) // 2
        y_center_o = (new_image_height - original_image.shape[0]) // 2

        # copy img image into center of result image
        pad_image[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img
        pad_original_image[y_center_o:y_center_o + original_image.shape[0], x_center_o:x_center_o + original_image.shape[1]] = original_image
        pad_mask[y_center_o:y_center_o + original_image.shape[0],
        x_center_o:x_center_o + original_image.shape[1]] = np.ones_like(original_image)
        boarders = [y_center_o, y_center_o + original_image.shape[0], x_center_o, x_center_o + original_image.shape[1]]
        boarders_input = [y_center, y_center + old_image_height, x_center, x_center + old_image_width]
        pad_image = self.transform(image=pad_image)['image']
        pad_original_image = self.transform(image=pad_original_image)['image']
        pad_image = np.transpose(pad_image, (2, 0, 1))
        pad_original_image = np.transpose(pad_original_image, (2, 0, 1))
        mask = pad_mask[:, :, 0].astype("float32")
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        self.iter_i += 1
        return dict(input_image=pad_image,
                    GT=pad_original_image,
                    mask=mask,
                    boarders=boarders,
                    boarders_input=boarders_input)


class RetargetingTrainWebDataset(IterableDataset):
    def __init__(self, indir, mask_generator, transform, shuffle_buffer=200):
        self.impl = webdataset.Dataset(indir).shuffle(shuffle_buffer).decode('rgb').to_tuple('jpg')
        self.mask_generator = mask_generator
        self.transform = transform

    def __iter__(self):
        for iter_i, (img,) in enumerate(self.impl):
            img = np.clip(img * 255, 0, 255).astype('uint8')
            img = self.transform(image=img)['image']
            img = np.transpose(img, (2, 0, 1))
            mask = self.mask_generator(img, iter_i=iter_i)
            yield dict(image=img,
                       mask=mask)

class RetargetingTrainYoloDeeplabDataset(Dataset):
    def __init__(self, indir, transform):
        self.img_suffix = '.jpg'
        self.mask_path = r"E:\data\CUB_200_2011\CUB_200_2011\max 256 dataset\aug_train256\img_mask_all"
        self.in_files = sorted(list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True)))
        # self.mask_files = sorted(list(glob.glob(os.path.join(self.mask_path, '**', '*.jpg'), recursive=True)))
        # print(len(self.in_files))
        self.transform = transform
        self.iter_i = 0
        self.yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt').cpu()

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = img
        old_image_height, old_image_width, channels = original_image.shape
        path_mask = os.path.join(self.mask_path, self.in_files[item].split("\\")[-1])
        mask = cv2.imread(path_mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        hu = -1
        if random.random() > 0.5:
            with torch.no_grad():
                yolo = self.yolo(img).crop(save=False)
            if yolo:
                cord = yolo[0]["box"]
                a = np.random.randint(0, int(cord[1] - hu))
                b = np.random.randint(int(cord[3] + hu), old_image_height)
                c = np.random.randint(0, int(cord[0] - hu))
                d = np.random.randint(int(cord[2] + hu), old_image_width)
                img = img[a:b, c:d]
                r1 = (b - a) / old_image_height
                r2 = (d - c) / old_image_width
            else:
                r1 = 1
                r2 = 1
            crop_rate = max(r1, r2)
            if crop_rate > 0.5:
                resize_rate = 0.8 - random.random() * (0.5 - (crop_rate - 0.5) / crop_rate)
                img = scale_image(img, resize_rate, resize_rate)
        else:
            resize_rate = 0.8 - random.random() * (0.8 - 0.5)
            img = scale_image(img, resize_rate, resize_rate)
            if resize_rate > 0.5:
                with torch.no_grad():
                    yolo = self.yolo(img).crop(save=False)
                if yolo:
                    cord = yolo[0]["box"]
                    a = np.random.randint(0, int(cord[1] - hu))
                    b = np.random.randint(int(cord[3] + hu), img.shape[0])
                    c = np.random.randint(0, int(cord[0] - hu))
                    d = np.random.randint(int(cord[2] + hu), img.shape[1])
                    img = img[a:b, c:d]

        # create new image of desired size and color (blue) for padding
        new_image_width = 384
        new_image_height = 384
        color = (0, 0, 0)
        channels_mask = 3
        pad_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_original_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_mask = np.full((new_image_height, new_image_width, channels_mask), color, dtype=np.uint8)
        # compute center offset
        x_center = (new_image_width - img.shape[1]) // 2
        y_center = (new_image_height - img.shape[0]) // 2
        x_center_o = (new_image_width - original_image.shape[1]) // 2
        y_center_o = (new_image_height - original_image.shape[0]) // 2

        # copy img image into center of result image
        pad_image[y_center:y_center + img.shape[0], x_center:x_center + img.shape[1]] = img
        pad_original_image[y_center_o:y_center_o + original_image.shape[0], x_center_o:x_center_o + original_image.shape[1]] = original_image
        pad_mask[y_center_o:y_center_o + original_image.shape[0], x_center_o:x_center_o + original_image.shape[1]] = mask
        boarders = [y_center_o, y_center_o + original_image.shape[0], x_center_o, x_center_o + original_image.shape[1]]

        pad_image = self.transform(image=pad_image)['image']
        pad_original_image = self.transform(image=pad_original_image)['image']
        pad_image = np.transpose(pad_image, (2, 0, 1))
        pad_original_image = np.transpose(pad_original_image, (2, 0, 1))
        pad_mask = self.transform(image=pad_mask)['image']
        pad_mask = np.transpose(pad_mask, (2, 0, 1))
        # pad_mask = pad_mask[:, :, :1].astype("float32") #for binary mask
        # pad_mask = np.transpose(pad_mask, (2, 0, 1))
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        self.iter_i += 1
        return dict(input_image=pad_image,
                    GT=pad_original_image,
                    mask=pad_mask,
                    boarders=boarders)


class RetargetingTrainOfflineDataset(Dataset):
    def __init__(self, indir, transform):
        self.img_suffix ='.jpg'
        self.gt_filenames = sorted(list(glob.glob(os.path.join(indir, '**', '*gt*.jpg'), recursive=True)))
        self.in_files = [fname.rsplit('_gt', 1)[0] + self.img_suffix for fname in self.gt_filenames]
        print(indir)
        print(len(self.gt_filenames))
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.gt_filenames)

    def __getitem__(self, item):
        img = cv2.imread(self.in_files[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gt = cv2.imread(self.gt_filenames[item])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        original_image = gt

        old_image_height, old_image_width, channels = img.shape

        # create new image of desired size and color (blue) for padding
        new_image_width = 256
        new_image_height = 256
        color = (0, 0, 0)
        pad_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_gt = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_mask = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        # compute center offset
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2
        x_center_o = (new_image_width - original_image.shape[1]) // 2
        y_center_o = (new_image_height - original_image.shape[0]) // 2

        # copy img image into center of result image
        pad_image[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img
        pad_gt[y_center_o:y_center_o + original_image.shape[0], x_center_o:x_center_o + original_image.shape[1]] = original_image
        pad_mask[y_center_o:y_center_o + original_image.shape[0], x_center_o:x_center_o + original_image.shape[1]] = np.ones_like(original_image)
        boarders = [y_center_o, y_center_o + original_image.shape[0], x_center_o, x_center_o + original_image.shape[1]]

        pad_image = self.transform(image=pad_image)['image']
        pad_original_image = self.transform(image=pad_gt)['image']
        pad_image = np.transpose(pad_image, (2, 0, 1))
        pad_original_image = np.transpose(pad_original_image, (2, 0, 1))
        mask = pad_mask[:, :, 0].astype("float32")
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        self.iter_i += 1
        return dict(input_image=pad_image,
                    GT=pad_original_image,
                    mask=mask,
                    boarders=boarders)

def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    # process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

def center_crop2(img, factor_h=1, factor_w=1):
    width, height = img.shape[1], img.shape[0]
    dim0 = int(img.shape[1]*factor_w)
    dim1 = int(img.shape[0]*factor_h)
    # process crop width and height for max available dimension
    crop_width = dim0 if dim0<img.shape[1] else img.shape[1]
    crop_height = dim1 if dim1<img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img

def scale_image(img, factor_h=1, factor_w=1):
    return cv2.resize(img,(int(img.shape[1]*factor_w), int(img.shape[0]*factor_h)))

class ImgSegmentationDataset(Dataset):
    def __init__(self, indir, mask_generator, transform, out_size, segm_indir, semantic_seg_n_classes):
        self.indir = indir
        self.segm_indir = segm_indir
        self.mask_generator = mask_generator
        self.transform = transform
        self.out_size = out_size
        self.semantic_seg_n_classes = semantic_seg_n_classes
        self.in_files = list(glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = img
        # img = cv2.resize(img, (self.out_size, self.out_size))
        H_im = img.shape[0]
        W_im = img.shape[1]

        r = random.random()
        if r > 0.5:
            img = center_crop(img, (W_im/2, H_im/2))

        else:
            img = scale_image(img, 0.5, 0.5)

        old_image_height, old_image_width, channels = img.shape

        # create new image of desired size and color (blue) for padding
        new_image_width = 256
        new_image_height = 256
        color = (0, 0, 0)
        pad_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        pad_original_image = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)
        # compute center offset
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        # copy img image into center of result image
        pad_image[y_center:y_center + old_image_height, x_center:x_center + old_image_width] = img
        pad_original_image[y_center:y_center + original_image.shape[0], x_center:x_center + original_image.shape[1]] = original_image

        pad_image = self.transform(image=pad_image)['image']
        pad_original_image = self.transform(image=pad_original_image)['image']
        pad_image = np.transpose(pad_image, (2, 0, 1))
        pad_original_image = np.transpose(pad_original_image, (2, 0, 1))
        mask = (pad_original_image > 0).astype("float32")
        result = dict(image=pad_image,
                      mask=mask)
        return result

    def load_semantic_segm(self, img_path):
        segm_path = img_path.replace(self.indir, self.segm_indir).replace(".jpg", ".png")
        mask = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.out_size, self.out_size))
        tensor = torch.from_numpy(np.clip(mask.astype(int)-1, 0, None))
        ohe = F.one_hot(tensor.long(), num_classes=self.semantic_seg_n_classes) # w x h x n_classes
        return ohe.permute(2, 0, 1).float(), tensor.unsqueeze(0)


def get_transforms(transform_variant, out_size):
    if transform_variant == 'default':
        transform = A.Compose([
            A.RandomScale(scale_limit=0.2),  # +/- 20%
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.7, 1.3),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale05_1':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.5, 1.0),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_12':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 1.2),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_07':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 0.7),  # scale 512 to 256 in average
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_light':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.02)),
            IAAAffine2(scale=(0.8, 1.8),
                       rotate=(-20, 20),
                       shear=(-0.03, 0.03)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'non_space_transform':
        transform = A.Compose([
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'no_augs':
        transform = A.Compose([
            A.ToFloat()
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform


def make_default_train_dataloader(indir, kind='default', out_size=512, mask_gen_kwargs=None, transform_variant='default',
                                  mask_generator_kind="mixed", dataloader_kwargs=None, ddp_kwargs=None, **kwargs):
    LOGGER.info(f'Make train dataloader {kind} from {indir}. Using mask generator={mask_generator_kind}')

    mask_generator = get_mask_generator(kind=mask_generator_kind, kwargs=mask_gen_kwargs)
    transform = get_transforms(transform_variant, out_size)

    if kind == 'default':
        dataset = RetargetingTrainYoloDeeplabDataset(indir=indir,
                                         transform=transform,
                                         **kwargs)
    elif kind == 'default_web':
        dataset = RetargetingTrainWebDataset(indir=indir,
                                            mask_generator=mask_generator,
                                            transform=transform,
                                            **kwargs)
    elif kind == 'img_with_segm':
        dataset = ImgSegmentationDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         out_size=out_size,
                                         **kwargs)
    else:
        raise ValueError(f'Unknown train dataset kind {kind}')

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    is_dataset_only_iterable = kind in ('default_web',)

    if ddp_kwargs is not None and not is_dataset_only_iterable:
        dataloader_kwargs['shuffle'] = False
        dataloader_kwargs['sampler'] = DistributedSampler(dataset, **ddp_kwargs)

    if is_dataset_only_iterable and 'shuffle' in dataloader_kwargs:
        with open_dict(dataloader_kwargs):
            del dataloader_kwargs['shuffle']

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_default_val_dataset(indir, kind='default', out_size=512, transform_variant='default', **kwargs):
    out_size = 256
    transform_variant = 'no_augs'
    if OmegaConf.is_list(indir) or isinstance(indir, (tuple, list)):
        return ConcatDataset([
            make_default_val_dataset(idir, kind=kind, out_size=out_size, transform_variant=transform_variant, **kwargs) for idir in indir 
        ])

    LOGGER.info(f'Make val dataloader {kind} from {indir}')
    mask_generator = get_mask_generator(kind=kwargs.get("mask_generator_kind"), kwargs=kwargs.get("mask_gen_kwargs"))
    transform = get_transforms(transform_variant, out_size)
    if transform_variant is not None:
        transform = get_transforms(transform_variant, out_size)

    if kind == 'default':
        dataset = RetargetingImageMaskTestDataset(indir, **kwargs)

    elif kind == 'our_eval':
        dataset = RetargetingImageMaskTestDataset(indir, **kwargs)

    elif kind == 'img_with_mask':
        dataset = RetargetingEvaluationDataset(indir, **kwargs)

    elif kind == 'online':
        dataset = RetargetingEvaluationDataset(indir, **kwargs)
    else:
        raise ValueError(f'Unknown val dataset kind {kind}')

    return dataset


def make_default_val_dataloader(*args, dataloader_kwargs=None, **kwargs):
    dataset = make_default_val_dataset(*args, **kwargs)

    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_constant_area_crop_params(img_height, img_width, min_size=128, max_size=512, area=256*256, round_to_mod=16):
    min_size = min(img_height, img_width, min_size)
    max_size = min(img_height, img_width, max_size)
    if random.random() < 0.5:
        out_height = min(max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod))
        out_width = min(max_size, ceil_modulo(area // out_height, round_to_mod))
    else:
        out_width = min(max_size, ceil_modulo(random.randint(min_size, max_size), round_to_mod))
        out_height = min(max_size, ceil_modulo(area // out_width, round_to_mod))

    start_y = random.randint(0, img_height - out_height)
    start_x = random.randint(0, img_width - out_width)
    return (start_y, start_x, out_height, out_width)
