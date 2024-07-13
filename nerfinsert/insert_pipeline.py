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

"""InstructPix2Pix Pipeline and trainer"""
import glob
import json
import os.path
import time
from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type
import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText

from nerfinsert.insert_datamanager import (
    NeRFInsertDataManagerConfig,
)

from nerfinsert.sd_example import PaintByExample
from nerfinsert.sd import StableDiffusion
from PIL import Image

import numpy as np
from scipy import ndimage

from .visual_hull import EpipolarMaskField
import copy

from nerfstudio.field_components.field_heads import FieldHeadNames

from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)


@dataclass
class NeRFInsertPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NeRFInsertPipeline)
    """target class to instantiate"""
    datamanager: NeRFInsertDataManagerConfig = NeRFInsertDataManagerConfig()
    """specifies the datamanager config"""
    pbe_used: bool = False
    """If true, we use PBE, if false we use Stable Diffusion impainting"""
    prompt: str = "man wearing blue headphones"
    """prompt for Stable Diffusion inpainting"""
    image_prompt_path: str = "data/inpaint_example_images/bluejbl.jpeg"
    """prompt for PaintByExample"""
    masks_path: str = "data/in2n-data/face/manualmasks/masks_headphones_2"
    """path to the masks that define the inpainting region"""
    edit_rate: int = 6000
    """how many NeRF steps before image edit"""
    edit_count: int = -1 # We set it equal to the number of training images
    """how many images to edit per NeRF step"""
    diffusion_steps: int = 20
    """Number of maximum diffusion steps to take"""
    lower_bound: float = 0.02
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 0.98
    """Upper bound for diffusion timesteps to use for image editing"""
    diffusion_device: Optional[str] = None
    """Second device to place InstructPix2Pix on. If None, will use the same device as the pipeline"""
    diffusion_use_full_precision: bool = True
    """Whether to use full precision for InstructPix2Pix"""

class NeRFInsertPipeline(VanillaPipeline):
    """NeRFInsert pipeline"""

    config: NeRFInsertPipelineConfig

    def __init__(
        self,
        config: NeRFInsertPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):

        super().__init__(config, device, test_mode, world_size, local_rank)

        # select device for InstructPix2Pix
        self.diffusion_device = (
            torch.device(device)
            if self.config.diffusion_device is None
            else torch.device(self.config.diffusion_device)
        )


        self.paint_by_example = False
        if self.paint_by_example:
            self.diffusion = PaintByExample(self.diffusion_device)
        else:
            self.diffusion = StableDiffusion(self.diffusion_device)


        # keep track of spot in dataset
        if self.datamanager.config.train_num_images_to_sample_from == -1:
            self.train_indices_order = cycle(range(len(self.datamanager.train_dataparser_outputs.image_filenames)))
        else:
            self.train_indices_order = cycle(range(self.datamanager.config.train_num_images_to_sample_from))

        # viewer elements
        self.prompt_box = ViewerText(name="Prompt", default_value=self.config.prompt, cb_hook=self.prompt_callback)
        #self.guidance_scale_box = ViewerNumber(name="Text Guidance Scale", default_value=self.config.guidance_scale, cb_hook=self.guidance_scale_callback)
        #self.image_guidance_scale_box = ViewerNumber(name="Image Guidance Scale", default_value=self.config.image_guidance_scale, cb_hook=self.image_guidance_scale_callback)


        self.config.masks_folder = 'data/in2n-data/face/manualmasks/masks_headphones_2'

        parse_masks_from_trainset = True
        if parse_masks_from_trainset:
            mask_transforms = self.parse_masks(self.config.masks_folder,
                                               self.datamanager.train_dataset._dataparser_outputs.metadata['json_path'])
            dataparser_transform = self.datamanager.train_dataset._dataparser_outputs.dataparser_transform
            dataparser_scale = self.datamanager.train_dataset._dataparser_outputs.dataparser_scale
        else:
            mask_transforms = json.load(open(self.config.masks_folder + '/transforms.json'))
            dataparser_transform = None
            dataparser_scale = None

        data_path = self.datamanager.train_dataset._dataparser_outputs.metadata['json_path'].split('/')[:-1]
        data_path = '/'.join(data_path)
        print('data path', data_path)
        self.maskfield_model = EpipolarMaskField(model=self.model,
                                                 masks_transform=mask_transforms,
                                                 dataparser_transform=dataparser_transform,
                                                 dataparser_scale=dataparser_scale)

        self.example_image = None
        if self.paint_by_example:
            self.example_image = Image.open(this.config.example_image_path)


        self.model_unedited = None

        self.config.edit_count = len(self.datamanager.train_dataparser_outputs.image_filenames)
        self.init_step = None
        self.first_iteration = True


    def guidance_scale_callback(self, handle: ViewerText) -> None:
        """Callback for guidance scale slider"""
        self.config.guidance_scale = handle.value

    def image_guidance_scale_callback(self, handle: ViewerText) -> None:
        """Callback for text guidance scale slider"""
        self.config.image_guidance_scale = handle.value

    def prompt_callback(self, handle: ViewerText) -> None:
        """Callback for prompt box, change prompt in config and update text embedding"""
        self.config.prompt = handle.value
        
        '''self.text_embedding = self.ip2p.pipe._encode_prompt(
            self.config.prompt, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        )'''

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """


        '''if self.model.modified_density_fns is None:
            densities = [2, 10]
            self.model.modified_density_fns = [self.create_density_fn(self.model.density_fns[i], densities[i]) for i in range(len(self.model.density_fns))]'''

        if self.first_iteration:
            self.first_iteration = False

            self.model_unedited = copy.deepcopy(self.model)

            self.propagate_masks()

        ray_bundle, batch = self.datamanager.next_train(step, balancemask=True)
        field_outputs = {}

        model_outputs, field_outputs['weights'], field_outputs['density'], field_outputs['rgb'], ray_samples = forward(self.model, ray_bundle, only_field_outputs=False)
        with torch.no_grad():
            field_outputs['weights_unedited'], field_outputs['density_unedited'], field_outputs['rgb_unedited'] = forward(self.model_unedited, ray_bundle, ray_samples=ray_samples, only_field_outputs=True)

        points = ray_samples.frustums.origins + ray_samples.frustums.directions * (ray_samples.frustums.starts + ray_samples.frustums.ends) / 2
        n_rays, n_across_ray, _ = points.shape
        assert _ == 3
        points = points.reshape(n_rays * n_across_ray, 3)
        in_mask = self.maskfield_model.in_mask(points)
        in_mask = in_mask.reshape(n_rays, n_across_ray, 1)
        in_mask = in_mask.expand(-1, -1, 3)

        field_outputs['weights_unedited'] = field_outputs['weights_unedited'].expand(-1, -1, 3)
        field_outputs['weights'] = field_outputs['weights'].expand(-1, -1, 3)
        field_outputs['density_unedited'] = field_outputs['density_unedited'].expand(-1, -1, 3)
        field_outputs['density'] = field_outputs['density'].expand(-1, -1, 3)


        for x in field_outputs.keys():
            field_outputs[x] = field_outputs[x][~in_mask]

        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        #self.render_images()

        # edit an image every ``edit_rate`` steps
        if step % self.config.edit_rate == 0:

            t = time.time()
            if self.init_step is None:
                self.init_step = step

            n_steps = self.n_steps
            strength = 1 - ((step - self.init_step) / n_steps)**0.5 * 0.8
            #strength = 0.5 - ((step - self.init_step) / n_steps) ** 0.5 * 0.3


            # edit ``edit_count`` images in a row
            n_edited = 0
            batch_size = 5
            while n_edited < self.config.edit_count:
                indices = []
                originals = []
                masks = []
                toedit = []
                i = 0
                while (i < batch_size) and (n_edited < self.config.edit_count):

                    n_edited += 1

                    # iterate through "spot in dataset"
                    current_spot = next(self.train_indices_order)

                    # get original image from dataset
                    original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device)
                    mask = self.datamanager.original_image_batch['inpaint_mask'][current_spot].to(self.device)
                    mask_dilated = self.datamanager.original_image_batch['inpaint_mask_dilated'][current_spot].to(self.device).float()
                    # generate current index in datamanger

                    if mask.mean() < 0.005:
                        continue

                    i += 1

                    indices.append(current_spot)
                    current_index = self.datamanager.image_batch["image_idx"][current_spot]

                    # get current camera, include camera transforms from original optimizer
                    camera_transforms = self.datamanager.train_camera_optimizer(current_index.unsqueeze(dim=0))
                    current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
                    current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)

                    # get current render of nerf
                    original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2)
                    mask = mask.unsqueeze(dim=0).permute(0, 3, 1, 2)[:, :1]
                    mask_dilated = mask_dilated.unsqueeze(dim=0).permute(0, 3, 1, 2)[:, :1]

                    masks.append(mask_dilated.cpu())

                    originals.append(original_image.cpu())

                    current_ray_bundle = current_ray_bundle.flatten()
                    mask_ = (mask > 0.5).flatten()

                    current_ray_bundle = current_ray_bundle[mask_].reshape((1, -1))

                    camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)

                    rendered_image = original_image.clone()
                    b, c, h, w = rendered_image.shape
                    assert b == 1
                    assert c == 3

                    rendered_image = rendered_image.squeeze(0).permute(1, 2, 0).reshape(h*w, 3)
                    rendered_image[mask_>0.5] = camera_outputs["rgb"]

                    rendered_image = rendered_image.reshape(h, w, 3)
                    rendered_image = rendered_image.permute(2, 0, 1).unsqueeze(0)

                    toedit.append(rendered_image.detach())

                    # delete to free up memory
                    del camera_outputs
                    del current_camera
                    del current_ray_bundle
                    del camera_transforms
                    torch.cuda.empty_cache()

                toedit = torch.concatenate(toedit, dim=0).cpu()
                masks = torch.concatenate(masks, dim=0).cpu()

                if self.paint_by_example:
                    edited_images = self.diffusion.edit_image(
                                example_image=[self.example_image] * toedit.shape[0],
                                image_toedit=toedit,
                                mask=masks,
                                diffusion_steps=int(self.config.diffusion_steps),
                                strength=strength,
                            )
                else:
                    edited_images = self.diffusion.edit_image(
                        text_prompt=[self.config.prompt] * toedit.shape[0],
                        image_toedit=toedit,
                        mask=masks,
                        diffusion_steps=int(self.config.diffusion_steps),
                        strength=strength,
                    )

                indices = torch.tensor(indices).long()


                for b in range(i):

                    current_spot = indices[b]
                    mask = masks[b]
                    original_image = originals[b]
                    edited_image = edited_images[b]

                    edited_image = torch.where(mask.to(edited_image.device)>0.5, edited_image, original_image.to(edited_image.device))
                    # write edited image to dataloader
                    self.datamanager.image_batch["image"][current_spot] = edited_image.squeeze().permute(1,2,0)

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict=metrics_dict, field_outputs=field_outputs)
        #print(loss_dict.keys())

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError



    def render_images(self):

        indices = [next(self.train_indices_order)]
        render = self.render(torch.tensor(indices[0]))
        render = render['rgb']
        filename = str(self.datamanager.train_dataparser_outputs.image_filenames[indices[-1]])
        filename = filename.split('/')[-1]
        filename = 'manual_renders/' + filename
        self.save_image(render, filename)

        indices.append(next(self.train_indices_order))
        while indices[-1] != indices[0]:
            render = self.render(torch.tensor(indices[-1]))
            render = render['rgb']
            filename = str(self.datamanager.train_dataparser_outputs.image_filenames[indices[-1]])
            filename = filename.split('/')[-1]
            filename = 'manual_renders/' + filename
            self.save_image(render, filename)
            print(filename)
            print(render.min(), render.max())
            indices.append(next(self.train_indices_order))
        quit()


    def render(self,
               index):
        camera_transforms = self.datamanager.train_camera_optimizer(index.unsqueeze(dim=0))
        current_camera = self.datamanager.train_dataparser_outputs.cameras[index].to(self.device)
        current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1),
                                                          camera_opt_to_camera=camera_transforms)
        return self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)


    def save_image(self, image, path):
        image = image.cpu().numpy()
        image = (image*255).astype(np.uint8)
        image = Image.fromarray(image)
        image.save(path)


    def create_density_fn(self, original_density_fn, mask_density):
        def f(xyz):
            assert len(xyz.shape) == 3
            assert xyz.shape[-1] == 3

            original_density = original_density_fn(xyz)

            n, z, _ = xyz.shape
            xyz = xyz.reshape(-1, 3)
            in_mask = self.maskfield_model.in_mask(xyz)
            in_mask = in_mask.reshape(n, z, 1)

            original_density[in_mask] = mask_density
            return original_density

        return f

    def parse_masks(self,
                    masks_folder,
                    json_path):

        mask_paths = glob.glob(masks_folder + '/*jpg')
        mask_ids = []
        assert len(mask_paths) > 0
        for m in mask_paths:
            print(m)
            mask_id = m.split('/')[-1].split('.')[0]
            print(mask_id)
            mask_ids.append(mask_id)
        original_data = json.load(open(json_path))

        frames_found = []

        for f in original_data['frames']:
            file_id = f['file_path']
            file_id = file_id.split('/')[-1].split('.')[0]
            if file_id in mask_ids:
                print('found', file_id)
                i = mask_ids.index(file_id)
                f['inpaint_mask_path'] = mask_paths[i]
                frames_found.append(f)

        assert len(frames_found) == len(mask_ids)

        return_data = dict(original_data)
        return_data['frames'] = frames_found

        return return_data



    def propagate_masks(self):

        print('mask folder:', self.config.masks_folder)
        propagated_masks_folder = self.config.masks_folder + '/propagated'
        dilated_masks_folder = self.config.masks_folder + '/propagated_dilated'
        if os.path.isdir(propagated_masks_folder):
            print('Propagated mask folder found!')
            for current_spot in range(len(self.datamanager.original_image_batch["image"])):

                index = self.datamanager.original_image_batch['image_idx'][current_spot]
                filename = str(self.datamanager.train_dataparser_outputs.image_filenames[index])
                filename = filename.split('/')[-1]
                filename = propagated_masks_folder + '/' + filename
                filename = filename.replace('.png', '.jpg')
                assert os.path.isfile(filename), filename

        else:

            print('Propagated mask folder not found: proceeding to mask propagation')
            os.mkdir(propagated_masks_folder)


            densities = [500, 500]
            self.model.modified_density_fns = [self.create_density_fn(self.model.density_fns[i], densities[i]) for i in
                                               range(len(self.model.density_fns))]

            for current_spot in range(len(self.datamanager.original_image_batch["image"])):

                current_index = self.datamanager.image_batch["image_idx"][current_spot]
                # get current camera, include camera transforms from original optimizer
                camera_transforms = self.datamanager.train_camera_optimizer(current_index.unsqueeze(dim=0))
                current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
                current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)

                mask_outputs = self.maskfield_model.get_outputs_for_camera_ray_bundle(current_ray_bundle)

                mask_rgb = np.array(mask_outputs['rgb'].cpu())

                mask_rgb_Image = Image.fromarray((mask_rgb * 255).astype(np.uint8))
                index = self.datamanager.original_image_batch['image_idx'][current_spot]
                filename = str(self.datamanager.train_dataparser_outputs.image_filenames[index])
                filename = filename.split('/')[-1]
                filename = propagated_masks_folder + '/' + filename
                filename = filename.replace('.png', '.jpg')
                mask_rgb_Image.save(filename)
                print(filename)

        masks = []
        masks_dilated = []

        dilation_size = 20
        y, x = np.ogrid[:dilation_size, :dilation_size]
        circle = (y - dilation_size / 2) ** 2 + (x - dilation_size / 2) ** 2 < (dilation_size / 2) ** 2


        if not os.path.isdir(dilated_masks_folder):
            os.mkdir(dilated_masks_folder)
            print('Dilated masks folder not found!, Creating one and starting dilating the masks')

        else:
            print('Dilated masks folder found!')

        for current_spot in range(len(self.datamanager.original_image_batch["image"])):
            index = self.datamanager.original_image_batch['image_idx'][current_spot]
            filename = str(self.datamanager.train_dataparser_outputs.image_filenames[index])
            filename = filename.split('/')[-1]
            filename = filename.replace('.png', '.jpg')
            total_filename = propagated_masks_folder + '/' + filename
            assert os.path.isfile(total_filename), total_filename

            mask = Image.open(total_filename)
            mask = np.array(mask)

            mask = torch.tensor(mask)
            mask = mask / 255
            mask = torch.where(mask > 0.5, 1.0, 0.0)
            masks.append(mask)

            filename_dilated = dilated_masks_folder + '/' + filename
            if not os.path.isfile(filename_dilated):

                mask = np.array(mask[...,0])
                mask = mask > 0.5
                mask = ndimage.binary_dilation(mask > 0.5, circle)
                mask = np.stack([mask] * 3, axis=2)
                masks_dilated.append(torch.tensor(mask))
                mask = Image.fromarray((mask * 255).astype(np.uint8))
                mask.save(filename_dilated)

            else:
                mask = Image.open(filename_dilated)
                mask = np.array(mask)
                mask = torch.tensor(mask)
                mask = mask / 255
                mask = torch.where(mask > 0.5, 1.0, 0.0)
                masks_dilated.append(mask)

        masks = torch.stack(masks)

        #self.datamanager.train_dataset.masks = torch.zeros_like(masks)
        #for i, j in enumerate(self.datamanager.image_batch["image_idx"]):
        #    self.datamanager.train_dataset.masks[j] = masks[i]
        #import pdb
        #pdb.set_trace()

        self.datamanager.image_batch['inpaint_mask'] = masks

        #self.datamanager.setup_train()

        self.datamanager.original_image_batch['inpaint_mask'] = masks
        masks_dilated = torch.stack(masks_dilated)
        self.datamanager.original_image_batch['inpaint_mask_dilated'] = masks_dilated.float()






def forward(self, ray_bundle, only_field_outputs=True, ray_samples = None):
    """Run forward starting with a ray bundle. This outputs different things depending on the configuration
    of the model and whether or not the batch is provided (whether we are training basically)

    Args:
        ray_bundle: containing all the information needed to render that ray latents included
    """
    # self has to be the model.
    if self.collider is not None:
        ray_bundle = self.collider(ray_bundle)

    return get_outputs(self, ray_bundle, only_field_outputs=only_field_outputs, ray_samples_=ray_samples)

def get_outputs(self, ray_bundle, only_field_outputs, ray_samples_):

    ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
    #ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.modified_density_fns)
    if ray_samples_ is not None:
        ray_samples = ray_samples_
    field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
    if self.config.use_gradient_scaling:
        field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

    weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
    weights_list.append(weights)

    if only_field_outputs:
        return weights, field_outputs[FieldHeadNames.DENSITY], field_outputs[FieldHeadNames.RGB]

    ray_samples_list.append(ray_samples)

    rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
    depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
    accumulation = self.renderer_accumulation(weights=weights)

    outputs = {
        "rgb": rgb,
        "accumulation": accumulation,
        "depth": depth,
    }

    if self.config.predict_normals:
        normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
        outputs["normals"] = self.normals_shader(normals)
        outputs["pred_normals"] = self.normals_shader(pred_normals)
    # These use a lot of GPU memory, so we avoid storing them for eval.
    if self.training:
        outputs["weights_list"] = weights_list
        outputs["ray_samples_list"] = ray_samples_list

    if self.training and self.config.predict_normals:
        outputs["rendered_orientation_loss"] = orientation_loss(
            weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
        )

        outputs["rendered_pred_normal_loss"] = pred_normal_loss(
            weights.detach(),
            field_outputs[FieldHeadNames.NORMALS].detach(),
            field_outputs[FieldHeadNames.PRED_NORMALS],
        )

    for i in range(self.config.num_proposal_iterations):
        outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

    return outputs, weights, field_outputs[FieldHeadNames.DENSITY], field_outputs[FieldHeadNames.RGB],  ray_samples

