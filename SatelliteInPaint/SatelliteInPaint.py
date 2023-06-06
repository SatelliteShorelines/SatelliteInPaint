# Written by Dr Daniel Buscombe, Marda Science LLC
# for  the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2023, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE zSOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image as pil_image
from skimage.morphology import binary_dilation, disk, remove_small_objects, binary_erosion
import numpy as np
from glob import glob
import os
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import *

def make_model_pipeline(option=1):
    """
    Save transect data, including raw timeseries, intersection data, and cross distances.

    Args:
        option (str): The ID of the ROI.

    Returns:
        pipe (obj): 
    """
    if option==1:
        ## set up the model pipeline
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        ).to("cuda")

    else:
        # alternative
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        ).to("cuda")
    return pipe

def get_files(folder):
    """
    Save transect data, including raw timeseries, intersection data, and cross distances.

    Args:
        folder (str): The ID of the ROI.

    Returns:
        files
        filetypes
    """
    files = glob(folder+os.sep+"*.jpg")+glob(folder+os.sep+"*.png")+glob(folder+os.sep+"*.tif")
    files = [f for f in files if "inpaint" not in f]
    files = [f for f in files if "workflow" not in f]

    filetypes = [] 
    for f in files:
        filetypes.append(os.path.basename(f).split('.')[-1])
    return files, filetypes

def open_image_get_mask(file, N = 512, thres=0.01):
    """
    Save transect data, including raw timeseries, intersection data, and cross distances.

    Args:
        file (str): The ID of the ROI.
        N (int): The directory path to save the transect data.
        thres (float): Dictionary containing cross distance transects data.

    Returns:
        image_orig
        mask
    """
    # open file and get size
    image_orig = pil_image.open(file)
    nx, ny = image_orig.size
    ## resize to NxN
    image_orig = image_orig.convert("RGB").resize((N,N))
    ## create mask from from channel zeros
    mask = ((np.array(image_orig)[:,:,0]<thres) * 255).astype(np.uint8)
    return image_orig, mask

def dilate_mask(mask, size_disk=9):
    """
    Save transect data, including raw timeseries, intersection data, and cross distances.

    Args:
        mask (pil_image.array): The ID of the ROI.
        size_disk (int): The directory path to save the transect data.

    Returns:
        mask_image
    """
    mask = binary_dilation(mask, disk(size_disk))
    ## convert back to PIL format
    mask_image = pil_image.fromarray(mask)
    return mask_image

def get_inpainted(pipe, image_orig, mask_image, prompt="", num_images_per_prompt=3, num_inference_steps=200, guidance_scale=18):
    """
    Save transect data, including raw timeseries, intersection data, and cross distances.

    Args:
        pipe (str): The ID of the ROI.
        image_orig (str): The directory path to save the transect data.
        mask_image (dict): Dictionary containing cross distance transects data.
        prompt (dict): Dictionary containing extracted shorelines data.
        num_images_per_prompt
        num_inference_steps
        guidance_scale

    Returns:
        outputs (list)
    """
    ## run model, ask for 3 outputs, and resize the outputs
    output = pipe(prompt = prompt, 
                  image=image_orig, 
                  mask_image=mask_image, 
                  num_inference_steps=num_inference_steps, 
                  guidance_scale=guidance_scale, 
                  num_images_per_prompt=num_images_per_prompt)

    outputs = []
    for k in range(num_images_per_prompt):
        outputs.append(output.images[0])
    return outputs

def merge_outputs(outputs, file, filetype, method, mask_image, debug=False):
    """
    Save transect data, including raw timeseries, intersection data, and cross distances.

    Args:
        outputs (str): The ID of the ROI.
        file (str): The directory path to save the transect data.
        filetype (dict): Dictionary containing cross distance transects data.
        method (dict): Dictionary containing extracted shorelines data.
        mask_image
        debug

    Returns:
        output.
    """
    rx, ry = outputs[0].size

    if method==1:
        # reopen orig image
        image_orig = pil_image.open(file)
        nx, ny = image_orig.size
        image_orig = image_orig.resize((rx,ry))

    # converet to arrays, take medians of each channel, and recombine
    outputs_arrays = []
    for k in outputs:
        outputs_arrays.append(np.array(k))

    output_arrayR = np.median(np.dstack([o[:,:,0] for o in outputs_arrays]),axis=-1)
    output_arrayG = np.median(np.dstack([o[:,:,1] for o in outputs_arrays]),axis=-1)
    output_arrayB = np.median(np.dstack([o[:,:,2] for o in outputs_arrays]),axis=-1)
    output_array = np.dstack((output_arrayR,output_arrayG,output_arrayB))/255.

    if method==2:
        # reopen orig image
        image_orig = pil_image.open(file)
        nx, ny = image_orig.size        
        output_array = (output_array*255).astype('uint8')
        output = pil_image.fromarray(output_array).resize((nx,ny))

    if method==1:
        image_array = np.array(image_orig)
        # mask out arrays so they can be added
        try:
            mask = ((np.array(image_array)[:,:,0]>0))
        except:
            mask = ((np.array(image_array)>0))
    
        ## fill in hole in mask
        mask2 = remove_small_objects(mask,min_size=16)
        ## mask the mask sligjtly smaller to deal with edge effects
        mask2 = binary_erosion(mask2, disk(2))

        output_array[mask2==True]=0
        image_array[mask2==False]=0    
        output_array = (output_array*255).astype('uint8')
        out_max = np.maximum(image_array,output_array)

        # this ensures that only the original missing portion is made up!
        output = pil_image.fromarray(out_max).resize((nx,ny))

    if debug:
        fig=plt.figure(figsize=(8,8))
        plt.subplot(221); plt.imshow(image_orig); plt.axis('off'); plt.title('a)',loc='left')
        plt.subplot(222); plt.imshow(mask_image, cmap='gray'); plt.axis('off'); plt.title('b)',loc='left')
        plt.subplot(223); plt.imshow(output_array); plt.axis('off'); plt.title('c)',loc='left')
        plt.subplot(224); plt.imshow(output); plt.axis('off'); plt.title('d)',loc='left')
        plt.savefig(file.replace("."+filetype,"_workflow.jpg"), dpi=300, bbox_inches="tight")
        plt.close(); del fig

    return output

def output_per_file_in_folder(pipe,files,filetypes,thres,N,prop_black_upper,size_disk,prompt,num_images_per_prompt,num_inference_steps,guidance_scale,method,debug):
    """
    Save transect data, including raw timeseries, intersection data, and cross distances.

    Args:
        pipe (str): The ID of the ROI.
        files (str): The directory path to save the transect data.
        filetypes (dict): Dictionary containing cross distance transects data.
        thres (dict): Dictionary containing extracted shorelines data.
        N
        prop_black_upper
        size_disk,prompt
        num_images_per_prompt
        num_inference_steps
        guidance_scale
        method
        debug

    Returns:
        None.
    """
    for file,filetype in zip(files,filetypes):
        ##
        image_orig, mask = open_image_get_mask(file, N = N, thres=thres)

        ## only return data if proportion black pixels between 0 and prop_black_upper
        if np.sum(mask==255)/(N*N)>0 and np.sum(mask==255)/(N*N)<prop_black_upper:
            ##
            mask_image = dilate_mask(mask,
                                     size_disk=size_disk)
            ##
            outputs = get_inpainted(pipe, 
                                    image_orig, 
                                    mask_image, 
                                    prompt=prompt, 
                                    num_images_per_prompt=num_images_per_prompt, 
                                    num_inference_steps=num_inference_steps, 
                                    guidance_scale=guidance_scale)
            ##
            output = merge_outputs(outputs, 
                                   file, 
                                   filetype, 
                                   method, 
                                   mask_image, 
                                   debug=debug)

            ## save to file
            output.save(file.replace("."+filetype,"_inpaint."+filetype))    
########

if __name__ == '__main__':
# def main():
    ## internal size of imagery to use with model
    N = 512
    ## for making mask. values less than thres are inpainted
    thres=0.01
    ## upper limit of black pixel proportion to be considered
    prop_black_upper = .3
    ## size of dilation applied to mask
    size_disk = 9
    ## number of images over which to average
    num_images_per_prompt = 3
    ## option 1 = "runwayml/stable-diffusion-inpainting",
    ## option 2 = "stabilityai/stable-diffusion-2-inpainting"
    option=1
    ## method 1 only replaces missing pixels. method 2 replaces all pixels
    method=1
    ## provide no text prompt
    prompt = ""

    ## number of inference steps (larger = longer time, more accurate)
    num_inference_steps=200
    ## guidance scale
    guidance_scale=18
    ## if True, create one "workflow" figure per input for debugging/presentation 
    debug = True #False

    ## user navigate to folder of files
    root = Tk()
    root.filename =  filedialog.askdirectory(initialdir = os.getcwd(),title = "Select directory of files")
    folder = root.filename
    print(folder)
    root.withdraw()

    ## make the model pipeline
    pipe = make_model_pipeline(option=option)
    ## get lists of file names and types
    files, filetypes = get_files(folder)
    ## cycle through each file
    output_per_file_in_folder(pipe,
                              files,
                              filetypes,
                              thres,
                              N,
                              prop_black_upper,
                              size_disk,
                              prompt,
                              num_images_per_prompt,
                              num_inference_steps,
                              guidance_scale,
                              method,
                              debug)

