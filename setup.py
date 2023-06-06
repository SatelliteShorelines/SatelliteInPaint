import setuptools

long_description = """\

"""

setuptools.setup(
    name="SatelliteInPaint",
    version="0.0.1",
    author="Daniel Buscombe",
    author_email="dbuscombe@gmail.com",
    description="A Stable Diffusion-based way to inpaint satellite imagery",
    keywords = 'image inpainting, Earth observation, satellite imagery, Stable Diffusion model',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dbuscombe-usgs/SatelliteInPaint",
    packages=['SatelliteInPaint'],
    install_requires=['numpy','diffusers','rasterio','matplotlib',
        'torch','pillow','scikit-image','tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
