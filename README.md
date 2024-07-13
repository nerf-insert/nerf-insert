

 # NeRF-Insert: Local 3D editing with multimodal control signals.
 
# Installation

## 1. Install Nerfstudio 0.3.2 dependencies

Follow the instructions [at this link](https://docs.nerf.studio/quickstart/installation.html) to install Nerfstudio, pytorch, tiny-cuda-nn, etc... Recommended to use conda environment. 

IMPORTANT: Currently NeRF-Insert works with Nerfstudio 0.3.2, so make sure to downgrade after installing:
`pip install --upgrade nerfstudio==0.3.2`

## 2. Installing NeRF-Insert


```bash
git clone https://github.com/benoriol/nerf-insert.git
cd nerf-insert
pip install --upgrade pip setuptools
pip install -e .
```

## 3. Checking the install

The following command should include `insert` as one of the options:
```bash
ns-train -h
```

# Using NeRF-Insert

First you must train a Nerfacto model using Nerfstudio instructions. It is important to deactivate the view-dependency in nerfacto model file in nerfstudio.
You can access more options in [NerfStudio](https://docs.nerf.studio/quickstart/first_nerf.html)
```bash
ns-train nerfacto --data {DATA_DIR}
```
We provide some example data from Instruct-NeRF2NeRF's "face" scene in "data/in2n-data/face"

All trained model are saved in folder ```outputs```. Within each experiment folder, the model checkpoints are going to be inside a folder called ```nerfstudio_models```

To start training for editing the NeRF, run the following command:

```bash
ns-train insert --data {DATA_DIR} --load-dir {outputs/.../nerfstudio_models}
```

You can use ns-train insert --help to see the additional commands. Look at the ones under Pipeline configuration for those specific to NeRF-insert, including:
```
--pipeline.pbe-used {True,False}                                        
    If true, we use PBE, if false we use Stable Diffusion impainting   
     (default: False)                                                  
 --pipeline.prompt STR                                                  
     prompt for Stable Diffusion inpainting (default: 'man wearing blue
     headphones')                                                      
 --pipeline.image-prompt-path STR                                      
     prompt for PaintByExample (default:                                
     data/inpaint_example_images/bluejbl.jpeg)                         
 --pipeline.masks-path STR                                             
     prompt for PaintByExample (default: 'man wearing blue headphones')
 --pipeline.edit-rate INT                                              
     how many NeRF steps before image edit (default: 6000)                      
 --pipeline.diffusion-steps INT                                         
     Number of maximum diffusion steps to take (default: 20)                                                            
 --pipeline.diffusion-device {None}|STR                                 
     Second device to place InstructPix2Pix on. If None, will use the  
     same device as the pipeline (default: None)
 --pipeline.diffusion-use-full-precision {True,False}                 
     Whether to use full precision for InstructPix2Pix (default: True)
```




After editing the NeRF, a camera trajectory can be rendered using ```ns-render```, look up nerfstudio documentation for more details.
