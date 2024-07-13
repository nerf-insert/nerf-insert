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


import sys
import time
from dataclasses import dataclass

import torch
from rich.console import Console
from torch import Tensor, nn
from jaxtyping import Float

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL

import numpy as np
import torchvision.transforms.functional as f

CONSOLE = Console(width=120)

import random


from diffusers import StableDiffusionInpaintPipeline

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers import DDIMScheduler

from transformers import logging

from diffusers import AsymmetricAutoencoderKL



logging.set_verbosity_error()


@dataclass
class UNet2DConditionOutput:
    sample: torch.FloatTensor


class StableDiffusion(nn.Module):
    """ControlNet implementation
    Args:
        device: device to use
        num_train_timesteps: number of training timesteps
    """

    def __init__(self, device: Union[torch.device, str]) -> None:
        super().__init__()

        self.device = device

        self.generator = torch.Generator(device="cpu").manual_seed(1)

        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            #"runwayml/stable-diffusion-inpainting",
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
            #revision='fp16'
        )

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)

        # improve memory performance
        #self.pipe.enable_attention_slicing()

        #self.scheduler = self.pipe.scheduler

        self.pipe.unet.eval()
        self.pipe.vae.eval()

        # use for improved quality at cost of higher memory
        '''if self.ip2p_use_full_precision:
            self.pipe.unet.float()
            self.pipe.vae.float()
        else:
            if self.device.index:
                self.pipe.enable_model_cpu_offload(self.device.index)
            else:
                self.pipe.enable_model_cpu_offload(0)'''
        CONSOLE.print("ControlNet loaded!")

    def edit_image(
        self,
        text_prompt: Float[Tensor, "N max_length embed_dim"],
        image_toedit: Float[Tensor, "BS 3 H W"],
        mask: Float[Tensor, "BS 1 H W"],
        diffusion_steps: int = 20,
        strength: float = 1.0,
        negative_prompt=None
    ) -> torch.Tensor:

        assert len(mask.shape) == 4
        assert mask.shape[0] == image_toedit.shape[0]
        assert mask.shape[1] == 1
        assert len(image_toedit.shape) == 4
        assert image_toedit.shape[1] == 3

        assert mask.min() == 0.0
        assert mask.max() == 1.0

        assert image_toedit.min() >= -0.01, image_toedit.min()
        assert image_toedit.max() <= 1.01, image_toedit.max()
        image_toedit = torch.clamp(image_toedit, 0, 1)


        assert strength > 0.0
        assert strength <= 1.0

        masks_cropped = []
        images_toedit_cropped = []
        edges = []

        cropsizes = []

        for i in range(mask.shape[0]):
            cropsize = random.randint(300, 300)
            cropsizes.append(cropsize)
            minh, minw, maxh, maxw = self.crop_around_mask(mask[i, 0])
            minh, maxh = self.check_borders((minh + maxh) // 2, cropsize, mask[i, 0].shape[0])
            minw, maxw = self.check_borders((minw + maxw) // 2, cropsize, mask[i, 0].shape[1])

            mask_crop = mask[i, :, minh:maxh, minw:maxw]
            image_crop = image_toedit[i, :, minh:maxh, minw:maxw]
            mask_crop_upsampled = f.resize(mask_crop.unsqueeze(0), 512)
            image_crop_upsampled = f.resize(image_crop.unsqueeze(0), 512)

            masks_cropped.append(mask_crop_upsampled)
            images_toedit_cropped.append(image_crop_upsampled)
            edges.append((minh, minw, maxh, maxw))

        masks_cropped = torch.cat(masks_cropped, dim=0)
        images_toedit_cropped = torch.cat(images_toedit_cropped, dim=0)

        # masks_cropped = f.resize(masks_cropped, 512)
        # images_toedit_cropped = f.resize(images_toedit_cropped, 512)

        masks_cropped = masks_cropped.to(self.device)
        images_toedit_cropped = images_toedit_cropped.to(self.device)

        # control_image = torch.where(mask > 0.5, -1, cropped_unedited.clone() / 2 + 0.5)
        # control_image = torch.where(mask > 0.5, -1, cropped_unedited.clone())
        t = time.time()
        with torch.no_grad():
            '''out = self.call_controlnet(
                prompt=text_prompt,
                num_inference_steps=diffusion_steps,
                generator=self.generator,
                eta=1.0,
                image=cropped_unedited,
                mask_image=mask,
                control_image=control_image,
                init_latent_image=image_toedit,
                strength=strength,
                output_type='pt',
            )'''
            out = self.call_sdinpaint(
                self.pipe,
                prompt=text_prompt,
                negative_prompt=negative_prompt,
                image=images_toedit_cropped * 2 - 1,
                mask_image=masks_cropped,
                num_inference_steps=diffusion_steps,
                strength=strength,
                eta=1.0,
                output_type='pt',
                guidance_scale=40,
                generator=self.generator)

            edited_images = image_toedit.clone()
            for i in range(mask.shape[0]):
                edited_crop = out.images[i]
                edited_crop_resized = f.resize(edited_crop.unsqueeze(0), cropsizes[i])
                minh, minw, maxh, maxw = edges[i]
                edited_images[i, :, minh:maxh, minw:maxw] = edited_crop_resized[0]

        edited_images = torch.clamp(edited_images, 0.0, 1.0)
        return edited_images

    @staticmethod
    def crop_around_mask(mask_image):

        mask_crop = torch.nonzero(mask_image.float())
        mins, _ = mask_crop.min(dim=0)
        minh, minw = mins
        maxs, _ = mask_crop.max(dim=0)
        maxh, maxw = maxs
        return minh, minw, maxh, maxw

    @staticmethod
    def check_borders(center, cropsize, max):

        assert cropsize < max, f'{cropsize}, {max}'

        if (center - cropsize // 2) < 0:
            return 0, cropsize
        elif (center + cropsize // 2) > max:
            return max - cropsize, max
        else:
            return center - cropsize // 2, center - cropsize // 2 + cropsize


    @staticmethod
    def crop_and_resize(image, resize=512):

        _, _, h, w = image.shape
        cropsize = min(h, w)

        top = (h-cropsize) // 2
        left = (w-cropsize) // 2

        image = image[...,top:top+cropsize, left:left+cropsize]

        assert image.shape[-1] == cropsize, f'crop went wrong {cropsize}'
        assert image.shape[-2] == cropsize, f'crop went wrong {cropsize}'

        image = f.resize(image, size=resize)

        return image, top, left, cropsize

    @staticmethod
    def paste_crop_back(image, crop, left, top, size):
        assert crop.shape[-2] == crop.shape[-1]
        crop = f.resize(crop, size=size)
        image[..., top:top+size, left:left+size] = crop
        return image

    @staticmethod
    def call_sdinpaint(
            self,
            prompt: Union[str, List[str]] = None,
            image: Union[torch.FloatTensor, PIL.Image.Image] = None,
            mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            strength: float = 1.0,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs
        self.check_inputs(
            prompt,
            height,
            width,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength, device=device
        )
        # check that number of inference steps is not < 1 - as this doesn't make sense
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipeline"
                f"steps is {num_inference_steps} which is < 1 and not appropriate for this pipeline."
            )
        # at which timestep to set the initial noise (n.b. 50% if strength is 0.5)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        # create a boolean to check if the strength is set to 1. if so then initialise the latents with pure noise
        is_strength_max = strength == 1.0

        # 5. Preprocess mask and image
        mask, masked_image, init_image = prepare_mask_and_masked_image(
            image, mask_image, height, width, return_image=True
        )
        mask_condition = mask.clone()

        # 6. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 4

        latents_outputs = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            image=init_image,
            timestep=latent_timestep,
            is_strength_max=is_strength_max,
            return_noise=True,
            return_image_latents=return_image_latents,
        )

        if return_image_latents:
            latents, noise, image_latents = latents_outputs
            #image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = self.vae.encode(image).latent_dist.mode

        else:
            latents, noise = latents_outputs

        # 7. Prepare mask latent variables
        mask, masked_image_latents = self.prepare_mask_latents(
            mask,
            masked_image,
            batch_size * num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            do_classifier_free_guidance,
        )

        # 8. Check that sizes of mask, masked image and latents match
        if num_channels_unet == 9:
            # default case for runwayml/stable-diffusion-inpainting
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image}"
                    f" = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of"
                    " `pipeline.unet` or your `mask_image` or `image` input."
                )
        elif num_channels_unet != 4:
            raise ValueError(
                f"The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}."
            )

        # 9. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 10. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            # concat latents, mask, masked_image_latents in the channel dimension
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            if num_channels_unet == 9:
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if num_channels_unet == 4:
                init_latents_proper = image_latents[:1]
                init_mask = mask[:1]

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = self.scheduler.add_noise(
                        init_latents_proper, noise, torch.tensor([noise_timestep])
                    )

                latents = (1 - init_mask) * init_latents_proper + init_mask * latents

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        if not output_type == "latent":
            condition_kwargs = {}
            if isinstance(self.vae, AsymmetricAutoencoderKL):
                init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
                init_image_condition = init_image.clone()
                init_image = self._encode_vae_image(init_image, generator=generator)
                mask_condition = mask_condition.to(device=device, dtype=masked_image_latents.dtype)
                condition_kwargs = {"image": init_image_condition, "mask": mask_condition}
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, **condition_kwargs)[0]
            #image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            has_nsfw_concept = [False] * image.shape[0]
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


def prepare_mask_and_masked_image(image, mask, height, width, return_image: bool = False):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """

    if image is None:
        raise ValueError("`image` input cannot be undefined.")

    if mask is None:
        raise ValueError("`mask_image` input cannot be undefined.")

    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError("Image should be in [-1, 1] range")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]
        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            # resize all images w.r.t passed height an width
            image = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = [i.resize((width, height), resample=PIL.Image.LANCZOS) for i in mask]
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    # n.b. ensure backwards compatibility as old function does not return image
    if return_image:
        return mask, masked_image, image

    return mask, masked_image