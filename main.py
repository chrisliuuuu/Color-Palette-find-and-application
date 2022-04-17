#!/usr/bin/env python3
import time
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
    sample: int = 0

    def recalculate_average(self, saturation: float, brightness: float) -> None:
        self.saturation = (self.sample * self.saturation + saturation) / (self.sample + 1)
        self.brightness = (self.sample * self.brightness + brightness) / (self.sample + 1)
        self.sample += 1

    def recalculate_confidence(self, total_pixels: int, frequency_in_new_image: int,
                               resolution_of_new_image: int) -> None:
        self.confidence = ((self.confidence / 100 * total_pixels + frequency_in_new_image) /
                           (total_pixels + resolution_of_new_image)) * 100


class Train:
    temperature: float
    training_file: str
    confidence_level: float
    dataPoint: Dict[float, HueData]

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

        logging.info("[STARTING TRAINING]")
        start_time = time.perf_counter()

        total_pixels = 0

        for filename in os.listdir(self.training_file):
            if not filename.endswith(".jpg"):
                continue

            logging.info(f"Training with: {filename}")
            image = Image.open(f"{self.training_file}/{filename}")
            hsv_image = image.convert("HSV")
            data = np.array(hsv_image)

            hue_frequency = {}
            for i, line in enumerate(data):
                # skip every other line for performance
                if i % 2 == 1:
                    continue

                for pix in line:
                    hsv_data = HueData(0, 0, 0)

                    if not pix[0] in self.dataPoint.keys():
                        hsv_data.sample = 1
                        hsv_data.saturation = pix[1]
                        hsv_data.brightness = pix[2]
                        self.dataPoint[pix[0]] = hsv_data

                    else:
                        self.dataPoint[pix[0]].recalculate_average(pix[1], pix[2])

                    if pix[0] in hue_frequency:
                        hue_frequency[pix[0]] += 1

                    else:
                        hue_frequency[pix[0]] = 1

            x, y, z = data.shape
            resolution = (x * y) / 2

            for key, value in self.dataPoint.items():
                if key not in hue_frequency:
                    continue
                value.recalculate_confidence(total_pixels, hue_frequency[key], resolution)

            total_pixels += resolution

            logging.info(f"Finished training with {filename}")

        # update the confidnence

        self.dataPoint = dict(filter(lambda x: x[1].confidence > self.confidence_level, self.dataPoint.items()))
        end_time = time.perf_counter()
        logging.info("[DONE TRAINING]")
        logging.info(f"[TRAINING DONE IN]: {end_time - start_time} seconds")

        print("TRAINED DATA")
        print(self.dataPoint)


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