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