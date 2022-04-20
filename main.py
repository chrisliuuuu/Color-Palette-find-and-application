#!/usr/bin/env python3
import colorsys
import math
from dataclasses import dataclass
import sys
from xmlrpc.client import boolean

import cv2
from PIL import Image
import numpy as np
import os
from typing import List, Tuple, Dict
import logging
import argparse
import pathlib
from util import PriorityQueue as PQ

from sklearn.cluster import KMeans
import time

#################
#     Types     #
#################
hueValue: float

##################
#   Constants    #
##################

HUE_SCALE_MAX_VALUE = 255
SAT_SCALE_MAX_VALUE = 255
VAL_SCALE_MAX_VALUE = 255

# numbers of colours to use to create palette for the image.
DATA_SET_SIZE = 50
COLOUR_PALETTE_SIZE = 5
"""
Each of these constant define the ranges for various HSV classifications.
Example HUE_RANGE_SIZE = 10 will lead to divisions in increments of 10. (0, 10), (11,20) .... (351, 360)
Increase counter to create more colours and reduce counters to create lesser counters.
"""
HUE_RANGE_SIZE = 10
SAT_RANGE_SIZE = 30
VAL_RANGE_SIZE = 30


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


@dataclass
class Node:
    start: int
    end: int
    next: List
    sAverage: float
    vAverage: float
    frequency: int
    endNode: boolean

    def __init__(self, start, end, nxt=None, ending=False):
        self.start = start
        self.end = end
        self.frequency = 0
        self.next = nxt
        self.HSVData = None
        self.sAverage = -1
        self.vAverage = -1
        self.endNode = ending

    def contains(self, value: float) -> bool:
        return self.start <= value < self.end

    # for value node specifically
    def record_sample(self, hueKey, hue, satKey, sat, valKey, val) -> None:
        # first sample
        if self.sAverage == -1 and self.vAverage == -1:
            self.sAverage = satKey
            self.vAverage = valKey

        self.frequency += 1


class HSVTree:
    tree: List[Node]
    heap: PQ
    rolling: dict

    def __init__(self):
        self.rolling = {}
        self.tree = []
        self.heap = PQ()
        self.create_tree()

    def create_tree(self) -> None:
        """Creates a 3-level colour tree using the range and size constant as branching factors. Top level nodes are
        Hue nodes, level 2 are saturation nodes and level 3 are brightness nodes"""

        logging.info("[CREATING COLOUR TREE]")

        for i in range(math.ceil(HUE_SCALE_MAX_VALUE / HUE_RANGE_SIZE)):
            if i == 0:
                s1 = 0
            else:
                s1 = (i * HUE_RANGE_SIZE)
            e1 = (i + 1) * HUE_RANGE_SIZE

            sat_list = []
            for j in range(math.ceil(SAT_SCALE_MAX_VALUE / SAT_RANGE_SIZE)):
                if j == 0:
                    s2 = 0
                else:
                    s2 = (j * SAT_RANGE_SIZE)
                e2 = (j + 1) * SAT_RANGE_SIZE

                value_list = []
                for k in range(math.ceil(VAL_SCALE_MAX_VALUE / VAL_RANGE_SIZE)):
                    if k == 0:
                        s3 = 0
                    else:
                        s3 = (k * VAL_RANGE_SIZE)
                    e3 = (k + 1) * VAL_RANGE_SIZE

                    value_list.append(Node(s3, e3, None, True))
                    self.rolling[str((s1, s2, s3))] = []

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

        # calculate indices of nodes that represent this HSV colour
        hue_idx = int(hue // HUE_RANGE_SIZE)
        sat_idx = int(sat // SAT_RANGE_SIZE)
        val_idx = int(val // VAL_RANGE_SIZE)

        # extract nodes
        hue_node = self.tree[hue_idx]
        sat_node = hue_node.next[sat_idx]
        val_node = sat_node.next[val_idx]

        # sanity check, may not be necessary
        if val_node.contains(val):

            # update rolling average of node
            colorKey = str((hue_node.start, sat_node.start, val_node.start))
            if not self.rolling[colorKey]:
                self.rolling[colorKey] = [hue, sat, val, 1]
            else:
                curHue = float(self.rolling[colorKey][0])
                curSat = float(self.rolling[colorKey][1])
                curVal = float(self.rolling[colorKey][2])
                curFreq = int(self.rolling[colorKey][3])

                self.rolling[colorKey][0] = (
                                                    curHue * curFreq + hue) / (curFreq + 1)
                self.rolling[colorKey][1] = (
                                                    curSat * curFreq + sat) / (curFreq + 1)
                self.rolling[colorKey][2] = (
                                                    curVal * curFreq + val) / (curFreq + 1)
                self.rolling[colorKey][3] = curFreq + 1

    def updateHeap(self, resolution):
        for key, value in self.rolling.items():
            if value:
                self.rolling[key][0] = int(self.rolling[key][0] / 255 * 360)
                self.rolling[key][1] = int(self.rolling[key][1] / 255 * 100)
                self.rolling[key][2] = int(self.rolling[key][2] / 255 * 100)
                self.rolling[key][3] = float(self.rolling[key][3] / resolution * 900)
                # filter out nearly white/black color
                if self.rolling[key][1] <= 5:
                    if self.rolling[key][2] <= 5 or self.rolling[key][2] >= 5:
                        self.rolling[key][3] = 0.0
                self.heap.update(key, - value[3])


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

        size = 0
        for filename in os.listdir(self.training_file):
            if not filename.endswith(".jpg"):
                continue

            logging.info(f"Training with: {filename}")
            image = Image.open(f"{self.training_file}/{filename}")
            hsv_image = image.convert("HSV")
            data = np.array(hsv_image)
            w, h, z = data.shape
            size = w * h

            for i, line in enumerate(data):
                # skip every other line for performance
                if i % 6 != 0:
                    continue

                for j, pix in enumerate(line):
                    if j % 6 != 0:
                        continue

                    self.dataPoints.add_sample(pix[0], pix[1], pix[2])

        end_time = time.perf_counter()
        logging.info("[DONE TRAINING]")
        logging.info(f"[TRAINING DONE IN]: {end_time - start_time} seconds")

        self.dataPoints.updateHeap(size)

        data_points = []

        # performing K-Means on the data-set
        for i in range(DATA_SET_SIZE):
            if self.dataPoints.heap.isEmpty():
                break
            data_points.append(self.dataPoints.rolling[self.dataPoints.heap.pop()][:3])
        data_array = np.asarray(data_points)
        colours, _ = data_array.shape
        kmeans = KMeans(n_clusters=min(colours, COLOUR_PALETTE_SIZE))
        kmeans.fit(data_array)
        generate_palette_image(kmeans.cluster_centers_)


def generate_palette_image(palette_as_hsv: np.array):
    """
    Generates the image of the palette
    :param palette_as_hsv: np array containing hsv values of colours
    :return: None
    """
    logging.info("[GENERATING PALETTE IMAGE]")
    palette_img = np.zeros((500, 720, 3), np.uint8)

    hsv_colour_tuples = list(map(tuple, palette_as_hsv))
    hsv_colour_tuples.sort(key=lambda x: x[0], reverse=True)

    width = 720 // len(hsv_colour_tuples)
    for i, hsv_tuple in enumerate(hsv_colour_tuples):
        # some weird shit I had to do to generate RGB values that work
        colour = tuple(round(i * 255) for i in
                       colorsys.hsv_to_rgb(hsv_tuple[0] / 359, hsv_tuple[1] / 100, hsv_tuple[2] / 100))

        top_left = (i * width, 0)
        bottom_right = ((i + 1) * width, 500)

        # had to reverse RGB cause cv2 accepts BRG for some reason
        cv2.rectangle(palette_img, top_left, bottom_right, colour[::-1], -1)

    cv2.imwrite("palette.png", palette_img)

    logging.info("[DONE]")


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
