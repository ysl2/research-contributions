# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import os
import math
import numpy as np
import torch
import monai
from monai import transforms, data
import argparse
from typing import List, Tuple
import os

from thesmuggler import smuggle

dl = smuggle('/home/yusongli/Documents/shidaoai_new_project/networks/yusongli/dataloader.py')
pp = smuggle('/home/yusongli/Documents/shidaoai_new_project/data/pathparser.py')


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_loader(args: argparse.ArgumentParser) -> monai.data.DataLoader:
    data_dir = args.data_dir
    # datalist_json = os.path.join(data_dir, args.json_list)
    datalist_json = args.json_list
    train_transform = transforms.Compose(
        [
            dl.LoadImaged(keys=["image", "label"], func=yusongli_load_data),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Resized(keys=['image', 'label'], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            # ? Intensity
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            # ? Shape
            transforms.RandFlipd(keys=['image', 'label'], prob=args.RandFlipd_prob, spatial_axis=0),
            transforms.RandFlipd(keys=['image', 'label'], prob=args.RandFlipd_prob, spatial_axis=1),
            transforms.RandFlipd(keys=['image', 'label'], prob=args.RandFlipd_prob, spatial_axis=2),
            transforms.RandZoomd(keys=['image', 'label']),
            transforms.RandRotated(keys=['image', 'label']),
            # transforms.RandRotate90d(keys=['image', 'label']),
            transforms.RandAffined(keys=['image', 'label']),
            transforms.Rand3DElasticd(keys=['image', 'label'], sigma_range=(5, 7), magnitude_range=(50, 150)),
            transforms.RandGridDistortiond(keys=['image', 'label']),
            # ? Noise
            transforms.RandGibbsNoised(keys=['image']),
            transforms.RandGaussianNoised(keys=['image']),
            # transforms.RandKSpaceSpikeNoised(keys=['image']),
            transforms.RandAdjustContrastd(keys=['image']),
            transforms.RandGaussianSmoothd(keys=['image']),
            transforms.RandGaussianSharpend(keys=['image']),
            transforms.RandHistogramShiftd(keys=['image']),
            transforms.RandCoarseDropoutd(keys=['image'], holes=1, max_holes=10, spatial_size=1, max_spatial_size=4),
            transforms.RandCoarseShuffled(keys=['image'], holes=1, max_holes=10, spatial_size=1, max_spatial_size=4),
            # ? Size
            transforms.ToTensord(keys=["image", "label"], dtype=torch.float32),
        ]
    )
    val_transform = transforms.Compose(
        [
            dl.LoadImaged(keys=["image", "label"], func=yusongli_load_data),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.Resized(keys=['image', 'label'], spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            transforms.ToTensord(keys=["image", "label"], dtype=torch.float32),
        ]
    )

    if args.test_mode:
        test_files = dl.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_ds = (
            data.Dataset(data=test_files, transform=val_transform)
            if args.use_normal_dataset
            else data.CacheDataset(
                data=test_files,
                transform=val_transform,
                cache_num=args.testset_cache_num,
                cache_rate=1.0,
                num_workers=args.workers,
            )
        )
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        return [
            data.DataLoader(
                test_ds,
                batch_size=1,
                shuffle=False,
                num_workers=args.workers,
                sampler=test_sampler,
                pin_memory=True,
                persistent_workers=True,
            ),
            train_transform,
            val_transform,
        ]

    datalist = dl.load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    train_ds = (
        data.Dataset(data=datalist, transform=train_transform)
        if args.use_normal_dataset
        else data.CacheDataset(
            data=datalist,
            transform=train_transform,
            cache_num=args.trainset_cache_num,
            cache_rate=1.0,
            num_workers=args.workers,
        )
    )

    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
        persistent_workers=True,
    )
    val_files = dl.load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    val_ds = (
        data.Dataset(data=val_files, transform=val_transform)
        if args.use_normal_dataset
        else data.CacheDataset(
            data=val_files,
            transform=val_transform,
            cache_num=args.valset_cache_num,
            cache_rate=1.0,
            num_workers=args.workers,
        )
    )
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = data.DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        sampler=val_sampler,
        pin_memory=True,
        persistent_workers=True,
    )
    return [train_loader, val_loader, train_transform, val_transform]


# _DATAPATH = '/home/yusongli/Public/sda1/_dataset/shidaoai/img/_out'
_DATAPATH = '/home/yusongli/_dataset/shidaoai/img/_out'


def yusongli_rootdir(args: argparse.ArgumentParser) -> str:
    return os.path.join(f'{_DATAPATH}', f'wangqifeng-spacial-dilated-net_{args.model_name}_val',)


def yusongli_load_data(yamlmetadata: List) -> Tuple[str]:
    _key, _where, _who, _number, _name = yamlmetadata[1:6]
    if _key == 'image':
        objpath = f'{_DATAPATH}/wangqifeng-spacial/{_where}/{_who}/{_number}/{_name}'
    elif _key == 'label':
        objpath = f'{_DATAPATH}/wangqifeng-spacial-dilated_maskonly/{_where}/{_who}/{_number}/{_name}'
    return (objpath,)


def yusongli_save_pred(yamlmetadata: List, epoch: int, rootdir: str) -> str:
    for i in range(len(yamlmetadata)):
        if isinstance(yamlmetadata[i], (tuple, list)) and len(yamlmetadata[i]) == 1:
            yamlmetadata[i] = yamlmetadata[i][0]
    _where, _who, _number, _name = yamlmetadata[2:6]

    return os.path.join(f'{rootdir}', f'{epoch:02d}/{_where}/{_who}/{_number}/{_name}.nii.gz')


if __name__ == '__main__':
    yamlmetadata2 = '/home/yusongli/Documents/research-contributions/UNETR/BTCV/dataset/meta_data2.yaml'
    with open(yamlmetadata2, 'r') as j:
        yamlmetadata = pp.yaml.load(j)
    yusongli_load_data(yamlmetadata['training'][0]['label'])
