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
Model for NeRFInsert
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.model_components.losses import (
    L1Loss,
    MSELoss,
    interlevel_loss,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig

@dataclass
class NeRFInsertModelConfig(NerfactoModelConfig):
    """Configuration for the NeRFInsertModel."""
    _target: Type = field(default_factory=lambda: NeRFInsertModel)
    use_lpips: bool = False
    """Whether to use LPIPS loss"""
    use_l1: bool = True
    """Whether to use L1 loss"""
    patch_size: int = 32
    """Patch size to use for LPIPS loss."""
    lpips_loss_mult: float = 1.0
    """Multiplier for LPIPS loss."""

class NeRFInsertModel(NerfactoModel):
    """Model for NeRFInsert."""

    config: NeRFInsertModelConfig

    def populate_modules(self):
        """Required to use L1 Loss."""
        super().populate_modules()

        if self.config.use_l1:
            self.rgb_loss = L1Loss()
        else:
            self.rgb_loss = MSELoss()
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_loss_dict(self, outputs, batch, field_outputs, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])

        loss_dict["outside_mask_loss"] = self.outside_mask_loss(field_outputs=field_outputs) * 10
        #loss_dict["outside_mask_loss"] = self.outside_mask_loss(field_outputs=field_outputs) * 0

        if self.config.use_lpips:
            out_patches = (outputs["rgb"].view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            gt_patches = (image.view(-1, self.config.patch_size,self.config.patch_size, 3).permute(0, 3, 1, 2) * 2 - 1).clamp(-1, 1)
            loss_dict["lpips_loss"] = self.config.lpips_loss_mult * self.lpips(out_patches, gt_patches)

        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"] * 10
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
        return loss_dict

    @staticmethod
    def outside_mask_loss(field_outputs):

        #TODO make weigths hyperparameters
        field_outputs['density'] = 1 - torch.exp(-field_outputs['density'])
        field_outputs['density_unedited'] = 1 - torch.exp(-field_outputs['density_unedited'])

        loss = torch.nn.functional.mse_loss(field_outputs['density'],
                                            field_outputs['density_unedited'],
                                            reduction='none') * 100


        loss = loss + torch.nn.functional.mse_loss(field_outputs['rgb'],
                                                   field_outputs['rgb_unedited'],
                                                   reduction='none') * 10
        loss = loss * (field_outputs['weights_unedited'] + field_outputs['weights'])
        loss = loss.mean()
        return loss
