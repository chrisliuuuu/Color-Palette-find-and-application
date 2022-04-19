#!/usr/bin/env python3
import math
import time
from dataclasses import dataclass
import sys
from xmlrpc.client import boolean

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

HUE_SCALE_MAX_VALUE = 255
SAT_SCALE_MAX_VALUE = 255
VAL_SCALE_MAX_VALUE = 255

# numbers of colours to use to create palette for the image.
COLOR_PALETTE_SIZE = 5
"""
Each of these constant define the ranges for various HSV classifications.
Example HUE_RANGE_SIZE = 10 will lead to divisions in increments of 10. (0, 10), (11,20) .... (351, 360)
Increase counter to create more colours and reduce counters to create lesser counters.
"""
HUE_RANGE_SIZE = 60
SAT_RANGE_SIZE = 25
VAL_RANGE_SIZE = 50


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
    sAverage: float
    vAverage: float
    frequency: int
    endNode: boolean

    def __init__(self, start, end, nxt=None, ending = False ):
        self.start = start
        self.end = end
        self.frequency = 0
        self.next = nxt
        self.HSVData = None
        self.sAverage = None
        self.vAverage = None
        self.endNode = ending

    def contains(self, value: float) -> bool:
        return self.start <= value < self.end

    # for value node specifically
    def record_sample(self, hueKey, hue, satKey, sat, valKey, val) -> None:
        # first sample
        if self.sAverage == None and self.vAverage == None:
            self.sAverage = satKey
            self.vAverage = valKey


        #if not self.HSVData:
        #    self.HSVData = HSVData(*sample)

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
        logging.info("[CREATING COLOUR TREE]")

        for i in range(HUE_SCALE_MAX_VALUE // HUE_RANGE_SIZE):
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

        for hue_node in self.tree:
            if not hue_node.contains(hue):
                continue

            for sat_node in hue_node.next:
                if not sat_node.contains(sat):
                    continue

                for val_node in sat_node.next:
                    if val_node.contains(val):
                        # print(f"Long Method: {val_node}")
                        # TODO: Make this converge to a value based on sample distribution
                        colorKey = str((hue_node.start, sat_node.start, val_node.start))
                        # print(self.rolling.keys())
                        if (self.rolling[colorKey] == []):
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
                            
                        """
                        val_node.record_sample(
                            
                            (hue_node.start + (hue_node.end - hue_node.start) / 2,
                             sat_node.start + (sat_node.end - sat_node.start) / 2,
                             val_node.start + (val_node.end - val_node.start) / 2)
                             
                            hue_node.start, hue, sat_node.start, sat, val_node.start, val)
                         self.heap.update(val_node.HSVData, -val_node.frequency)
                        """

    def updateHeap(self, resolution) :
        for key, value in self.rolling.items():
            if value != []:
                self.rolling[key][0] = int(self.rolling[key][0]/ 255 * 360)
                self.rolling[key][1] = int(self.rolling[key][1] / 255 * 100)
                self.rolling[key][2] = int(self.rolling[key][2] / 255 * 100)
                self.rolling[key][3] = float(self.rolling[key][3] / resolution * 900)
                # filter out nearly white/black color
                if (self.rolling[key][1] <= 5):
                    if (self.rolling[key][2] <= 5 or self.rolling[key][2] >= 5):
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
            size = w*h

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

        self.dataPoints.updateHeap(size)

        for i in range(COLOR_PALETTE_SIZE):
            if self.dataPoints.heap.isEmpty():
                break
            print(
                f"{i + 1} : {self.dataPoints.rolling[self.dataPoints.heap.pop()]}")

 
# K-means
def do_cluster2(hsv_array, K):
    # gets height, width and the number of channes from the image shape
    h,w,c = hsv_array.shape
    # prepares data for clustering by reshaping the image matrix into a (h*w) x c matrix of pixels
    cluster_data = hsv_array.reshape( (h*w,c) )
    # grabs the initial time
    t0 = t.time()
    # performs clustering
    codebook, distortion = kmeans(np.array(cluster_data, dtype=np.float), K)
    print(codebook)
    # takes the final time
    t1 = t.time()
    print("Clusterization took %0.5f seconds" % (t1-t0))
    
    
    # calculates the total amount of pixels
    tot_pixels = h*w
    # generates clusters
    data, dist = vq(cluster_data, codebook)
    # calculates the number of elements for each cluster
    weights = [len(data[data == i]) for i in range(0,K)]
    
    # creates a 4 column matrix in which the first element is the weight and the other three
    # represent the h, s and v values for each cluster
    color_rank = np.column_stack((weights, codebook))
    # sorts by cluster weight
    color_rank = color_rank[np.argsort(color_rank[:,0])]
    # print(color_rank)

    # creates a new blank image
    new_image =  np.array([0,0,255], dtype=np.uint8) * np.ones( (500, 500, 3), dtype=np.uint8)
    img_height = new_image.shape[0]
    img_width  = new_image.shape[1]

    # for each cluster
    for i,c in enumerate(color_rank[::-1]):
        
        # gets the weight of the cluster
        weight = c[0]
        
        # calculates the height and width of the bins
        height = int(weight/float(tot_pixels) *img_height )
        width = int(img_width/len(color_rank))

        # calculates the position of the bin
        x_pos = int(i*width)


        
        # defines a color so that if less than three channels have been used
        # for clustering, the color has average saturation and luminosity value
        color = np.array( [0,0,0], dtype=np.uint8)
        # print(color)
        
        # substitutes the known HSV components in the default color
        for j in range(len(c[1:])):
            color[j] = c[j+1]
        # print('===========')
        color_plt.append(color)
        
        # # draws the bin to the image
        new_image[img_height-height:img_height, x_pos:x_pos+width] = [color[0], color[1], color[2]]
        
    # returns the cluster representation
    return new_image, color_plt



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
