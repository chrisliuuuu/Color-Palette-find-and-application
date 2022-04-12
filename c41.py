import sys

from PIL import Image
import numpy as np
import os
from typing import List, Tuple, Dict


class TrainingData:
    temperature: float

    def __init__(self):
        self.temperature = -1

    def train_from_folder(self, image_directory="./sample-image"):
        self.temperature = -1
        if not os.path.isdir(image_directory):
            print("Image Root folder does not exist")
            sys.exit(1)

        for filename in os.listdir(image_directory):
            if not filename.endswith(".jpeg"):
                continue

if __name__ == "__main__":
    t = TrainingData()
    t.train_from_folder()
