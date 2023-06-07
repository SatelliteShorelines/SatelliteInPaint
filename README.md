# 📦 SatelliteInPaint

Inpaint satellite imagery using Stable Diffusion. It takes an input image, and fills in the zeros. It does this by dilating and filtering a mask of the zeros in the input image using morphological operators, then using the Stable Diffusion model to carry out inpainting in that region. The original and inpainted images are merged such that only the previously zero pixels are modified. 

It uses a cuda-enabled GPU and runs using pytorch

User variables:

* `prop_black_upper` to set an upper limit on the allowable proportion of black pixels, e.g. 0.3 for 30% black pixels. This tool is not suitable for imagery with very large gaps. 
* `size_disk` size of dilation applied to mask
* `num_inference_steps`number of inference steps (larger = longer time, more accurate)
* `guidance_scale` guidance scale
* `method` method 1 only replaces missing pixels. method 2 replaces all pixels
* `option`  option 1 uses the "runwayml/stable-diffusion-inpainting" model, whereas option 2 uses the "stabilityai/stable-diffusion-2-inpainting" pretrained checkpoint
* `num_images_per_prompt` number of images over which to average

### ✍️ Authors

* [@dbuscombe-usgs](https://github.com/dbuscombe-usgs)


## ⬇️ Installation

Optionally, install a conda env for your dependencies

```
conda create -n inpaint python -y
conda activate inpaint
```

Install the requirements using pip:

```
pip install -r requirements.txt
```

## Usage

```
python SatelliteInPaint.py  
```

## Example outputs

![2006-12-09-15-22-08_RGB_L5_workflow](https://github.com/dbuscombe-usgs/SatelliteInPaint/assets/3596509/9619e517-f34b-4c29-8502-7fd8c87f96e0)
![2006-12-24-15-23-53_RGB_L7_workflow](https://github.com/dbuscombe-usgs/SatelliteInPaint/assets/3596509/224f87ba-179c-43b5-9ca6-a0a958adf7a0)
![2018-08-31-15-43-33_RGB_S2_workflow](https://github.com/dbuscombe-usgs/SatelliteInPaint/assets/3596509/d6f57193-91e8-49b6-b588-fb7f5be442bf)
![2018-07-22-15-47-20_RGB_S2_workflow](https://github.com/dbuscombe-usgs/SatelliteInPaint/assets/3596509/b9b66d88-5785-4b00-a1e4-9e7ea051f39c)
