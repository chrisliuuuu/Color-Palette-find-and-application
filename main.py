#!/usr/bin/env python3
from dataclasses import dataclass
import sys

from PIL import Image
import numpy as np
import os
from typing import List, Tuple, Dict
import logging
import argparse
import pathlib

#################
#     Types     #
#################
hueValue: float


@dataclass
class HueData:
    """Class for keeping track of data related to a particular Hue"""
    confidence: float
    saturation: float
    brightness: float

    def recalculate_average(self, saturation: float, brightness: float, samples: int) -> None:
        self.saturation = (samples * self.saturation + saturation) / (samples + 1)
        self.brightness = (samples * self.brightness + brightness) / (samples + 1)


class Train:
    temperature: float
    training_file: str
    confidence_level: float
    dataPoint: Dict[hueValue, HueData]

    def __init__(self, training_file: pathlib.Path, confidence_level: float):
        self.temperature = -1
        self.training_file = training_file.as_posix()
        self.dataPoint = {}
        self.confidence_level = confidence_level

    def train(self):
        """Loops through the directory with training data and collects samples and adds it to dataPoint"""
        self.temperature = -1

        if not os.path.isdir(self.training_file):
            print("Image Root folder does not exist")
            sys.exit(1)

        for filename in os.listdir(self.training_file):
            if not filename.endswith(".jpg"):
                continue

            logging.info(f"Training with: {filename}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Image Processor")
    p.add_argument('-f', dest="training_file", type=pathlib.Path, help="Location of directory containing training "
                                                                       "images", required=True)
    p.add_argument('-c', dest="confidence", type=float, help="Minimum confidence level required to use a colour",
                   default=1)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    logging.info("Beginning training")
    t = Train(args.training_file, args.confidence)
    t.train()
