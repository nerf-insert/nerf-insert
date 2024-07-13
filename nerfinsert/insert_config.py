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
Instruct-NeRF2NeRF configuration file.
"""

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
#from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from .insert_dataparser import NeRFInsertDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.plugins.types import MethodSpecification

from nerfinsert.insert_datamanager import NeRFInsertDataManagerConfig
from nerfinsert.insert import NeRFInsertModelConfig
from nerfinsert.insert_pipeline import NeRFInsertPipelineConfig
from nerfinsert.insert_trainer import NeRFInsertTrainerConfig

insert_method = MethodSpecification(
    config=NeRFInsertTrainerConfig(
        method_name="insert",
        steps_per_eval_batch=1000,
        steps_per_eval_image=200,
        steps_per_save=250,
        max_num_iterations=90000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=NeRFInsertPipelineConfig(
            datamanager=NeRFInsertDataManagerConfig(
                dataparser=NeRFInsertDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
                patch_size=32,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=NeRFInsertModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=True,
            ),
            diffusion_use_full_precision=True
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-NeRF2NeRF primary method: uses LPIPS, IP2P at full precision",
)

insert_method_small = MethodSpecification(
    config=NeRFInsertTrainerConfig(
        method_name="insert-small",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=30000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=NeRFInsertPipelineConfig(
            datamanager=NeRFInsertDataManagerConfig(
                dataparser=NeRFInsertDataParserConfig(),
                train_num_rays_per_batch=16384,
                eval_num_rays_per_batch=4096,
                patch_size=32,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=NeRFInsertModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=True,
            ),
            diffusion_use_full_precision=False,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-NeRF2NeRF small method, uses LPIPs, IP2P at half precision",
)

insert_method_tiny = MethodSpecification(
    config=NeRFInsertTrainerConfig(
        method_name="insert-tiny",
        steps_per_eval_batch=1000,
        steps_per_eval_image=100,
        steps_per_save=250,
        max_num_iterations=30000,
        save_only_latest_checkpoint=True,
        mixed_precision=True,
        pipeline=NeRFInsertPipelineConfig(
            datamanager=NeRFInsertDataManagerConfig(
                dataparser=NeRFInsertDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
                patch_size=1,
                camera_optimizer=CameraOptimizerConfig(
                    mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=1e-30, eps=1e-8, weight_decay=1e-2)
                ),
            ),
            model=NeRFInsertModelConfig(
                eval_num_rays_per_chunk=1 << 15,
                use_lpips=False,
            ),
            diffusion_use_full_precision=False,
        ),
        optimizers={
            "proposal_networks": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Instruct-NeRF2NeRF tiny method, does not use LPIPs, IP2P at half precision",
)