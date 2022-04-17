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

##################
#   Constants    #
##################

"""
Each of these constant define the ranges for various HSV classifications.
Example HUE_RANGE_SIZE = 10 will lead to divisions in increments of 10. (0, 10), (11,20) .... (351, 360)
Increase counter to create more colours and reduce counters to create lesser counters.
"""
HUE_RANGE_SIZE = 72
SAT_RANGE_SIZE = 72
VAL_RANGE_SIZE = 72


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


@dataclass
class Node:
    start: int
    end: int
    next: List
    frequency: int

    def __init__(self, start, end, nxt=None):
        self.start = start
        self.end = end
        self.frequency = 0
        self.next = nxt

    def contains(self, value: float) -> bool:
        return self.start <= value <= self.end

    def record_sample(self) -> None:
        self.frequency += 1


class HSVTree:
    tree: List[Node]

    def __init__(self):
        self.tree = []
        self.create_tree()

    def create_tree(self) -> None:

        for i in range(360 // HUE_RANGE_SIZE):
            if i == 0:
                s1 = 0
            else:
                s1 = (i * HUE_RANGE_SIZE) + 1
            e1 = (i + 1) * HUE_RANGE_SIZE

            sat_list = []
            for j in range(360 // SAT_RANGE_SIZE):
                if j == 0:
                    s2 = 0
                else:
                    s2 = (j * SAT_RANGE_SIZE) + 1
                e2 = (j + 1) * SAT_RANGE_SIZE

                value_list = []
                for k in range(360 // VAL_RANGE_SIZE):
                    if k == 0:
                        s3 = 0
                    else:
                        s3 = (k * VAL_RANGE_SIZE) + 1
                    e3 = (k + 1) * VAL_RANGE_SIZE

                    value_list.append(Node(s3, e3))

                sat_list.append(Node(s2, e2, value_list))

            self.tree.append(Node(s1, e1, sat_list))

    def add_sample(self, hue: float, sat: float, val: float) -> None:
        """
        Traverses down the tree and adds a sample. Yes I hate the number of for loops as well
        :param hue: Hue value
        :param sat: Saturation value
        :param val: Brightness value
        :return: None
        """
        for hue_node in self.tree:
            if not hue_node.contains(hue):
                continue

            for sat_node in hue_node.next:
                if not sat_node.contains(sat):
                    continue

                for val_node in sat_node.next:
                    if val_node.contains(val):
                        val_node.record_sample()


class Train:
    temperature: float
    training_file: str
    confidence_level: float
    dataPoints: HSVTree

    def __init__(self, training_file: pathlib.Path, confidence_level: float):
        self.temperature = -1
        self.training_file = training_file.as_posix()
        self.dataPoints = HSVTree()
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
                    self.dataPoints.add_sample(pix[0], pix[1], pix[2])

            logging.info("[TRAINED]")
            print(self.dataPoints.tree)

        #             hsv_data = HueData(0, 0, 0)
        #
        #             if not pix[0] in self.dataPoint.keys():
        #                 hsv_data.sample = 1
        #                 hsv_data.saturation = pix[1]
        #                 hsv_data.brightness = pix[2]
        #                 self.dataPoint[pix[0]] = hsv_data
        #
        #             else:
        #                 self.dataPoint[pix[0]].recalculate_average(pix[1], pix[2])
        #
        #             if pix[0] in hue_frequency:
        #                 hue_frequency[pix[0]] += 1
        #
        #             else:
        #                 hue_frequency[pix[0]] = 1
        #
        #     x, y, z = data.shape
        #     resolution = (x * y) / 2
        #
        #     for key, value in self.dataPoint.items():
        #         if key not in hue_frequency:
        #             continue
        #         value.recalculate_confidence(total_pixels, hue_frequency[key], resolution)
        #
        #     total_pixels += resolution
        #
        #     logging.info(f"Finished training with {filename}")
        #
        # # update the confidnence
        #
        # self.dataPoint = dict(filter(lambda x: x[1].confidence > self.confidence_level, self.dataPoint.items()))
        # end_time = time.perf_counter()
        # logging.info("[DONE TRAINING]")
        # logging.info(f"[TRAINING DONE IN]: {end_time - start_time} seconds")
        #
        # print("TRAINED DATA")
        # print(self.dataPoint)


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
