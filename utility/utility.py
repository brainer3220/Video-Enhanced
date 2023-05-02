import os
from glob import glob


def get_video_file(path):
    return glob(path + "/*.mp4")


def get_directory(path):
    return glob(path + "/*")


def make_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)
