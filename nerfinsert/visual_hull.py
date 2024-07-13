# Copyright 2017-2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.



import torch

import numpy as np
from PIL import Image

from nerfstudio.field_components.field_heads import FieldHeadNames
from collections import defaultdict


class EpipolarMaskField:

    def __init__(self,
                 model,
                 masks_transform,
                 dataparser_transform=None,
                 dataparser_scale=None,
                 ):
        self.model = model
        self.masks_transforms = masks_transform
        self.masks = []
        self.w2cs = []
        self.intrinsics = []

        fl_x_ = self.masks_transforms['fl_x'] if 'fl_x' in self.masks_transforms.keys() else None
        fl_y_ = self.masks_transforms['fl_y'] if 'fl_y' in self.masks_transforms.keys() else None
        cx_ = self.masks_transforms['cx'] if 'cx' in self.masks_transforms.keys() else None
        cy_ = self.masks_transforms['cy'] if 'cy' in self.masks_transforms.keys() else None

        for frame in self.masks_transforms['frames']:
            #mask_image = Image.open(data_path + '/' + frame['inpaint_mask_path'])
            mask_image = Image.open(frame['inpaint_mask_path'])
            mask_image = np.array(mask_image.convert('L')) / 255
            mask_image = torch.tensor(mask_image) > 0.5
            assert len(mask_image.shape) == 2
            self.masks.append(mask_image)

            c2w = torch.tensor(frame['transform_matrix'])

            if c2w.shape == (3, 4):
                c2w = torch.cat([c2w, torch.tensor([[0, 0, 0, 1]], dtype=c2w.dtype)], dim=0)
            assert c2w.shape == (4, 4)
            if dataparser_transform is not None:
                c2w = dataparser_transform @ c2w
                c2w[:3, 3] = c2w[:3, 3] * dataparser_scale
                c2w = torch.cat([c2w, torch.tensor([[0, 0, 0, 1]], dtype=c2w.dtype)], dim=0)
            assert c2w.shape == (4, 4)

            fl_x = fl_x_ if fl_x_ is not None else frame['fl_x']
            fl_y = fl_y_ if fl_y_ is not None else frame['fl_y']
            cx = cx_ if cx_ is not None else frame['cx']
            cy = cy_ if cy_ is not None else frame['cy']
            # self.c2ws.append(torch.tensor(frame['transform_matrix']))
            self.w2cs.append(c2w.inverse())
            self.intrinsics.append(
                torch.tensor(
                    [[fl_x, 0, cx],
                     [0, fl_y, cy],
                     [0, 0, 1]]
                )  # Switching Y axis according to camera ray generator. _generate_rays_from_coords(...)
            )

    @staticmethod
    def in_frustrum_test(xyz,
                         mask,
                         w2c,
                         intrinsic_matrix):
        assert xyz.shape[-1] == 3
        assert len(xyz.shape) == 2
        assert len(mask.shape) == 2
        ## We start by projecting into camera

        xyz = xyz.clone()
        xyz = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device=xyz.device)], dim=1)

        w2c = w2c.to(xyz.device)
        # xyz = (w2c @ xyz.T).T This is equivalent
        xyz = xyz @ w2c.T

        xyz = xyz[:, :-1].float()
        sv = xyz / -xyz[:, 2:]  # PROJECTIVE TRANSFORM. Project to plane z = -1
        sv[:, 2] = 1.0  # Let's fix the homogeneous coordinates after projection
        sv[:, 1] = -sv[:, 1]  # Now the y coordinate should be increasing downwards to match array or pixel convention
        # plane is projected onto the z=-1 plane not z=1
        intrinsic_matrix = intrinsic_matrix.to(xyz.device)
        sv = sv @ intrinsic_matrix.T[:, :2]
        sv = torch.round(sv).long()
        mask = mask.to(xyz.device)

        ## Now starts the mask comparison
        inmask = (sv[:, 0] > 0) * (sv[:, 0] < mask.shape[1])
        inmask = inmask * (sv[:, 1] > 0) * (sv[:, 1] < mask.shape[0])
        mask = mask > 0.5
        mask_ = mask.flatten()  # flatten mask to be indexed efficiently
        sv_ = sv[:, 0] + mask.shape[1] * sv[:, 1]  ## flatten sv positions
        sv_ = sv_.long()
        inmask_ = inmask.clone()
        inmask[inmask_] = inmask[inmask_] * torch.take(mask_, sv_[inmask_])

        return inmask


    def in_mask(self,
                xyz):
        in_mask = self.in_frustrum_test(xyz, self.masks[0], self.w2cs[0], self.intrinsics[0])
        for i in range(1, len(self.masks)):
            in_mask = in_mask * self.in_frustrum_test(xyz, self.masks[i], self.w2cs[i], self.intrinsics[i])

        return in_mask


    def get_outputs(self, ray_bundle):
        #raise NotImplementedError()
        # ray_samples, weights_list, ray_samples_list = self.model.proposal_sampler(ray_bundle, density_fns=self.model.density_fns)
        ray_samples, weights_list, ray_samples_list = self.model.proposal_sampler(ray_bundle, density_fns=self.model.modified_density_fns)

        field_outputs = self.model.field.forward(ray_samples, compute_normals=self.model.config.predict_normals)

        points = ray_samples.frustums.origins + ray_samples.frustums.directions * (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        #points = self.model.field.spatial_distortion(points)
        #points = (points + 2.0) / 4.0

        #points = (points+2)/4
        n_rays, n_across_ray, _ = points.shape
        assert _ == 3

        points = points.reshape(n_rays * n_across_ray, 3)

        in_mask = self.in_mask(points)

        #in_mask = self.sphere_test(xyz=points)
        in_mask = in_mask.reshape(n_rays, n_across_ray)
        field_outputs[FieldHeadNames.DENSITY][in_mask] = 400
        #field_outputs[FieldHeadNames.DENSITY][~in_mask] = 0.
        field_outputs[FieldHeadNames.RGB][in_mask] = torch.tensor([1., 1., 1.], device=field_outputs[FieldHeadNames.RGB].device)
        field_outputs[FieldHeadNames.RGB][~in_mask] = torch.tensor([0., 0., 0.], device=field_outputs[FieldHeadNames.RGB].device)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.model.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        depth = self.model.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.model.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }


        if self.model.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.model.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.model.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs


    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle):
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.model.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle)
            for output_name, output in outputs.items():  # type: ignore
                if not torch.is_tensor(output):
                    # TODO: handle lists of tensors as well
                    continue
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs


    def forward(self, ray_bundle):
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.model.collider is not None:
            ray_bundle = self.model.collider(ray_bundle)

        return self.get_outputs(ray_bundle)