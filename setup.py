'''
code for computing and visualizing drone racelines
'''
from setuptools import setup

setup(
    name='drone3d',
    version='0.1',
    packages=['drone3d'],
    install_requires=[
        'numpy',
        'casadi',
        'imgui',
        'glfw',
        'PyOpenGL',
        'pygltflib',
        'pillow',
        'scipy',
        'trimesh',
        'rtree',
        ]
)
