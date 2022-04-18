#!/usr/bin/env python3
import math
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
from util import PriorityQueue as PQ

#################
#     Types     #
#################
hueValue: float

##################
#   Constants    #
##################

HUE_SCALE_MAX_VALUE = 360
SAT_SCALE_MAX_VALUE = 255
VAL_SCALE_MAX_VALUE = 255

# numbers of colours to use to create palette for the image.
COLOR_PALETTE_SIZE = 6
"""
Each of these constant define the ranges for various HSV classifications.
Example HUE_RANGE_SIZE = 10 will lead to divisions in increments of 10. (0, 10), (11,20) .... (351, 360)
Increase counter to create more colours and reduce counters to create lesser counters.
"""
HUE_RANGE_SIZE = 10
SAT_RANGE_SIZE = 10
VAL_RANGE_SIZE = 10


@dataclass
class HSVData:
    """Class for keeping track of data related to a particular Hue"""
    hue: float
    saturation: float
    brightness: float

    def buildColor(self, hue: float, sat: float, val: float) -> None:
        self.hue = hue
        self.saturation = sat
        self.brightness = val

    def getNormalizedPercentages(self) -> str:
        return f"Hue: {self.hue}, Sat: {self.saturation / 255 * 100}%, Brightness: {self.brightness / 255 * 100}%"

    # def recalculate_average(self, saturation: float, brightness: float) -> None:
    #     self.saturation = (self.sample * self.saturation + saturation) / (self.sample + 1)
    #     self.brightness = (self.sample * self.brightness + brightness) / (self.sample + 1)

    # def recalculate_confidence(self, total_pixels: int, frequency_in_new_image: int,
    #                            resolution_of_new_image: int) -> None:
    #     self.confidence = ((self.confidence / 100 * total_pixels + frequency_in_new_image) /
    #                        (total_pixels + resolution_of_new_image)) * 100


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
        self.HSVData = None

    def contains(self, value: float) -> bool:
        return self.start <= value <= self.end

    def record_sample(self, sample) -> None:
        if not self.HSVData:
            self.HSVData = HSVData(*sample)

        self.frequency += 1


class HSVTree:
    tree: List[Node]
    heap: PQ

    def __init__(self):
        self.tree = []
        self.heap = PQ()
        self.create_tree()

    def create_tree(self) -> None:
        logging.info("[CREATING COLOUR TREE]")

        for i in range(HUE_SCALE_MAX_VALUE // HUE_RANGE_SIZE):
            if i == 0:
                s1 = 0
            else:
                s1 = (i * HUE_RANGE_SIZE) + 1
            e1 = (i + 1) * HUE_RANGE_SIZE

            sat_list = []
            for j in range(math.ceil(SAT_SCALE_MAX_VALUE / SAT_RANGE_SIZE)):
                if j == 0:
                    s2 = 0
                else:
                    s2 = (j * SAT_RANGE_SIZE) + 1
                e2 = (j + 1) * SAT_RANGE_SIZE

                value_list = []
                for k in range(math.ceil(VAL_SCALE_MAX_VALUE / VAL_RANGE_SIZE)):
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
                        print(f"Long Method: {val_node}")
                        # TODO: Make this converge to a value based on sample distribution
                        val_node.record_sample(
                            (hue_node.start + (hue_node.end - hue_node.start) / 2,
                             sat_node.start + (sat_node.end - sat_node.start) / 2,
                             val_node.start + (val_node.end - val_node.start) / 2))
                        self.heap.update(val_node.HSVData, -val_node.frequency)


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

        for filename in os.listdir(self.training_file):
            if not filename.endswith(".jpg"):
                continue

            logging.info(f"Training with: {filename}")
            image = Image.open(f"{self.training_file}/{filename}")
            hsv_image = image.convert("HSV")
            data = np.array(hsv_image)

            for i, line in enumerate(data):
                # skip every other line for performance
                if i % 3 != 0:
                    continue

                for j, pix in enumerate(line):
                    if j % 3 != 0:
                        continue

                    self.dataPoints.add_sample(pix[0], pix[1], pix[2])

        end_time = time.perf_counter()
        logging.info("[DONE TRAINING]")
        logging.info(f"[TRAINING DONE IN]: {end_time - start_time} seconds")
        logging.info("GENERATING COLOUR PALETTE")

        for i in range(COLOR_PALETTE_SIZE):
            if self.dataPoints.heap.isEmpty():
                break
            print(f"{i + 1} : {self.dataPoints.heap.pop().getNormalizedPercentages()}")


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
