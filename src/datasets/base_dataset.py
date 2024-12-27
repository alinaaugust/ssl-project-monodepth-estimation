from __future__ import absolute_import, division, print_function
import logging
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Takes dataset index (list[dict]), processes different monocular datasets
    for the same task in the identical manner.
    """

    def __init__(
        self,
        data_dir,
        filenames,
        height=192,
        width=640,
        frames=[0, -1, 1],
        scaling_factor=[0, 1, 2, 3],
        img_format=".jpg",
        split="train",
        colorjitter_params={
            "brightness": (0.8, 1.2),
            "contrast": (0.8, 1.2),
            "saturation": (0.8, 1.2),
            "hue": (-0.1, 0.1),
        },
    ):
        """
        Args:
            data_dir (str): path to the data folder.
            filenames (list[str]): list of filenames with the data.
            height (int): stores input image height.
            width (int): stores input image width.
            frames (list[int]): frames to load.
            scaling_factor (list[int]): scales to use in loss.
            img_format (str): extension of images in dataset.
            split (str): train/val/test data split.

            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.filenames = filenames

        self.height = height
        self.width = width
        self.frames = frames
        self.num_scaling_factors = len(scaling_factor)

        self.img_format = img_format
        self.split = split
        self.loader = pil_loader

        self.data_dir = data_dir
        if split == "train":
            self.data_dir = self.data_dir + "/train"
        elif split == "val":
            self.data_dir = self.data_dir + "/val"

        self.get_tensor = transforms.ToTensor()
        self.get_resize = {
            i: transforms.Resize(
                (self.height // 2 ** i, self.width // 2 ** i),
                interpolation=Image.ANTIALIAS,
            )
            for i in range(self.num_scaling_factors)
        }

        self.get_depth = self.check_depth()
        self.colorjitter_params = colorjitter_params

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance.
        """
        use_color_transform = (
            True if self.split == "train" and random.random() >= 0.5 else False
        )
        use_flip = True if self.split == "train" and random.random() >= 0.5 else False

        instance_data = {}
        instance_info = self.filenames[ind].split()
        path, idx, side = None, 0, None
        path = instance_info[0]
        path = path[path.find('/') + 1:]

        if len(instance_info) == 3:
            idx = instance_info[1]
            side = instance_info[2]

        for f in self.frames:
            if f == "s":
                second_side = "r" if side == "l" else "l"
                instance_data[("color", f, -1)] = self.get_color(
                    path, idx, second_side, use_flip
                )
            else:
                instance_data[("color", f, -1)] = self.get_color(
                    path, idx + str(f), side, use_flip
                )

        for scale_idx in range(self.num_scaling_factors):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale_idx)
            K[1, :] *= self.height // (2 ** scale_idx)

            inv_K = np.linalg.pinv(K)

            instance_data[("K", scale_idx)] = torch.from_numpy(K)
            instance_data[("inv_K", scale_idx)] = torch.from_numpy(inv_K)

        self.preprocess_data(instance_data, use_color_transform)

        for f in self.frames:
            del instance_data[("color", f, -1)]
            del instance_data[("color_transformed", f, -1)]

        if self.get_depth:
            target_depth = self.get_depth(path, idx, side, use_flip)
            instance_data["target_depth"] = np.expand_dims(target_depth, 0)
            instance_data["target_depth"] = torch.from_numpy(
                instance_data["target_depth"].astype(np.float32)
            )

        if "s" in self.frames:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if use_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1
            instance_data["stereo_T"] = torch.from_numpy(stereo_T)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self.filenames)

    def preprocess_data(self, instance_data, use_color):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance.
            use_color (bool): flag for using ColorJitter transform.
        Returns:
            instance_data (dict): dict, containing instance (possibly transformed via
                instance transform).
        """
        for key, _ in instance_data.items():
            if "color" in key:
                n, im, i = key
                for i in range(self.num_scaling_factors):
                    instance_data[(n, im, i)] = self.get_resize[i](
                        instance_data[(n, im, i - 1)]
                    )

        for key, frame in instance_data.items():
            if "color" in key:
                n, im, i = key
                instance_data[(n, im, i)] = self.get_tensor(frame)
                if use_color:
                    transform = transforms.ColorJitter.get_params(
                        self.colorjitter_params["brightness"],
                        self.colorjitter_params["contrast"],
                        self.colorjitter_params["saturation"],
                        self.colorjitter_params["hue"],
                    )
                    frame = transform(frame)
                instance_data[(n + "_transformed", im, i)] = self.get_tensor(frame)

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, path, idx, side, use_flip):
        raise NotImplementedError

    def get_color(self, path, idx, side, use_flip):
        raise NotImplementedError
