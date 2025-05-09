''' utilities for loading assets '''
import os

def get_assets_folder() -> str:
    ''' return os path to assets folder for this module'''
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets',)

def get_assets_file(file: str) -> str:
    ''' return os path to assets file for this module'''
    return os.path.join(get_assets_folder(), file)
