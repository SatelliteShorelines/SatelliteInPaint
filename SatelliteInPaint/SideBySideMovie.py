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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.io import imread 
from glob import glob 
import os 
import matplotlib.ticker as ticker
from tkinter import filedialog
from tkinter import *

def make_ani_sidebyside(files1, files2):
    fig, ax = plt.subplots(1,2)
    ims = []
    for f1, f2 in zip(files1, files2):
        im1 = ax[0].imshow(imread(f1), animated=True)
        ax[0].xaxis.set_major_locator(ticker.NullLocator())
        ax[0].yaxis.set_major_locator(ticker.NullLocator())
        t1=ax[0].text(20,20,f1.split(os.sep)[-1].split('.jpg')[0], color='w', animated=True)

        im2 = ax[1].imshow(imread(f2), animated=True)
        ax[1].xaxis.set_major_locator(ticker.NullLocator())
        ax[1].yaxis.set_major_locator(ticker.NullLocator())
        t2=ax[1].text(20,20,f2.split(os.sep)[-1].split('.jpg')[0], color='w', animated=True)

        ims.append([im1,t1, im2, t2])
        # ax.clear()

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                    repeat_delay=0)
    return ani


#====================================================================
## user navigate to folder of files
root = Tk()
root.filename =  filedialog.askdirectory(initialdir = os.getcwd(),title = "Select directory of files")
folder = root.filename
print(folder)
root.withdraw()

fps = 1

files = sorted(glob(folder+os.sep+"*L7.jpg"))
inpaint_files = [f.replace(".jpg","_inpaint.jpg") for f in files]

#====================================================================
ani = make_ani_sidebyside(files,inpaint_files)
ani.save(folder+os.sep+"L7_orig_inpaint.gif", writer='imagemagick', fps=fps)
del ani



# ani = make_ani(files)
# ani.save(folder+os.sep+"L7_orig.gif", writer='imagemagick', fps=2)
# del ani

# 
# ani = make_ani(inpaint_files)
# ani.save(folder+os.sep+"L7_inpaint.gif", writer='imagemagick', fps=2)
# del ani

# plt.show()
# ani.save(folder+os.sep+"L7_inpainted.mp4")

# writer = animation.FFMpegWriter(
#     fps=15, metadata=dict(artist='Dan Buscombe, Marda Science LLC'), bitrate=1800)
# ani.save("L7_inpainted.mp4", writer=writer)

# def make_ani(files):
#     fig, ax = plt.subplots()
#     ims = []
#     for f in files:
#         im = ax.imshow(imread(f), animated=True)
#         ax.xaxis.set_major_locator(ticker.NullLocator())
#         ax.yaxis.set_major_locator(ticker.NullLocator())
#         t=ax.text(10,10,f.split(os.sep)[-1].split('.jpg')[0], color='w', animated=True)
#         ims.append([im,t])
#         # ax.clear()

#     ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
#                                     repeat_delay=0)
#     return ani