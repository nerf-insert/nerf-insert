# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Instruct-NeRF2NeRF Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

from rich.progress import Console

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
import torch
CONSOLE = Console(width=120)

@dataclass
class NeRFInsertDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the NeRFInsertDataManager."""

    _target: Type = field(default_factory=lambda: NeRFInsertDataManager)
    patch_size: int = 32
    """Size of patch to sample from. If >1, patch-based sampling will be used."""

class NeRFInsertDataManager(VanillaDataManager):
    """Data manager for NeRFInsert."""

    config: NeRFInsertDataManagerConfig

    def setup_train(self):
        """Sets up the data loaders for training"""

        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

        # pre-fetch the image batch (how images are replaced in dataset)
        self.image_batch = next(self.iter_train_image_dataloader)
        # keep a copy of the original image batch
        self.original_image_batch = {}
        self.original_image_batch['image'] = self.image_batch['image'].clone()
        self.original_image_batch['image_idx'] = self.image_batch['image_idx'].clone()
        # self.original_image_batch['inpaint_mask'] = self.image_batch['inpaint_mask'].clone()

        '''
        import numpy as np
        from scipy import ndimage

        size = 21
        y, x = np.ogrid[:size, :size]
        circle = (y - size / 2) ** 2 + (x - size / 2) ** 2 < (size / 2) ** 2

        self.original_image_batch['inpaint_mask_dilated'] = []
        for mask in self.original_image_batch['inpaint_mask']:
            mask = mask[...,0].cpu().numpy()
            mask = ndimage.binary_dilation(mask>0.5, circle)
            self.original_image_batch['inpaint_mask_dilated'].append(torch.tensor(mask))
        self.original_image_batch['inpaint_mask_dilated'] = torch.stack(self.original_image_batch['inpaint_mask_dilated'])
        self.original_image_batch['inpaint_mask_dilated'] = self.original_image_batch['inpaint_mask_dilated'].unsqueeze(-1).repeat(1, 1, 1, 3)
        '''

    def next_train(self, step: int, balancemask=False) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        assert self.train_pixel_sampler is not None
        '''print(self.image_batch.keys())
        print(self.image_batch['image_idx'])
        print(self.image_batch['image'].shape)
        print(self.image_batch['inpaint_mask'].shape)
        quit()'''
        batch = self.train_pixel_sampler.sample(self.image_batch)

        if balancemask:

            #print(batch.keys())
            n_topick = batch['image'].shape[0]
            n_picked = int(n_topick * 0.7)
            for x in batch.keys():
                batch[x] = batch[x][:n_picked]

            while n_picked < n_topick:
                batch_ = self.train_pixel_sampler.sample(self.image_batch)
                #print(batch.keys())
                #quit()

                mask_ = batch_['inpaint_mask'].mean(dim=1) > 0.5

                for x in batch.keys():

                    batch_[x] = batch_[x][mask_.cpu()]

                    batch[x] = torch.concatenate((batch[x], batch_[x]), dim=0)
                n_picked += batch_['image'].shape[0]

            for x in batch.keys():
                batch[x] = batch[x][:n_topick]

        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)

        return ray_bundle, batch
