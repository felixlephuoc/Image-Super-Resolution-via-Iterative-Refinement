import os
import torch
import torchvision
import random
import numpy as np
from PIL import Image
from torchvision.transforms.functional import hflip

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1)):    
    imgs = [totensor(img) for img in img_list]
    if split == 'train':
        imgs = torch.stack(imgs, 0)
        imgs = hflip(imgs)
        imgs = torch.unbind(imgs, dim=0)
    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs]
    return ret_img

################## Process Frames ##############################

def get_paths_from_frames(path):
    """
    Get a list of file paths for all valid frames files in a directory and its subdirectories.
    """
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)

    frame_paths = []  

    for dirpath, _ , _ in sorted(os.walk(path)):
        for path, _, fnames in sorted(os.walk(dirpath)): 
            frames = []
            for fname in sorted(fnames):
                # Check if the file has an image file extension
                if is_image_file(fname):
                    img_path = os.path.join(path, fname)
                    frames.append(img_path)
        frame_paths.append(frames)
    return frame_paths


def read_concat(frames_paths, n_frames=3, high_resolution=False):
    """
    Reads and concatenate a list of frames.
    """

    if len(frames_paths) < n_frames:
        return None
    
    tensors = []

    # Select only middle frame in the case of High Resolution
    if high_resolution:
        middle_frame = frames_paths[len(frames_paths)//2]
        return totensor(Image.open(middle_frame).convert("RGB"))
              
    for frame_path in frames_paths:
        try:
            img = Image.open(frame_path).convert("RGB")
            tensor = totensor(img)
            tensors.append(tensor)
        except (OSError, IOError):
            return None

    concatenated_tensor = torch.cat(tensors, dim=0)

    return concatenated_tensor


def transform_augment_frames(frames_list, split='val', min_max=(0, 1), _hflip=True):
    """
    Applies data augmentation transformations to a list of frames.
    """
    
    # data augmentation for training set
    if split == 'train':
        if _hflip and random.random() < 0.5:
            frames_list = [hflip(frame) for frame in frames_list]
    # normalization
    ret_frames = [frame * (min_max[1] - min_max[0]) + min_max[0] for frame in frames_list]
    return ret_frames