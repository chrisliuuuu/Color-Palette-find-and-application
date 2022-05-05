# game.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
# Modified by Krishna Kothandaraman

from typing import List
import numpy as np
import action
from util import *


#######################
# Parts worth reading #
#######################


class Agent:
    """
    An agent must define a getAction method, but may also define the
    following methods which will be called if they exist:
    def registerInitialState(self, state): # inspects the starting state
    """

    def __init__(self, index=0):
        self.index = index

    def getAction(self, state):
        """
        The Agent will receive a GameState (from either {pacman, capture, sonar}.py) and
        must return an action from Directions.{North, South, East, West, Stop}
        """
        raiseNotDefined()


class Configuration:
    """
    A Configuration holds the (x,y) coordinate of a character, along with its
    traveling direction.
    The convention for positions, like a graph, is that (0,0) is the lower left corner, x increases
    horizontally and y increases vertically.  Therefore, north is the direction of increasing y, or (0,1).
    """

    def __init__(self, image_array: np.array, actionType: action.ActionType):
        self.image = image_array
        self.actionType = actionType

    def getPosition(self) -> np.array:
        return self.image

    def getDirection(self) -> action.ActionType:
        return self.actionType

    def __eq__(self, other):
        if other is None:
            return False
        return self.image == other.image and self.actionType == other.actionType

    def __hash__(self):
        x = hash(self.image)
        y = hash(self.actionType)
        return hash(x + 13 * y)

    def __str__(self):
        return "(image)=" + str(self.image) + ", (action)=" + str(self.actionType)

    def generateSuccessor(self, newAction: action.ActionType):
        """
        Generates a new configuration reached by translating the current
        configuration by the action vector.  This is a low-level call and does
        not attempt to respect the legality of the movement.
        Actions are movement vectors.
        """
        return Configuration(action.take_action(self.image, newAction), newAction)


class Actions:
    """
    A collection of static methods for manipulating move actions.
    """

    def getPossibleActions() -> List[action.ActionType]:
        return [nextAction for nextAction in action.ActionType]

    getPossibleActions = staticmethod(getPossibleActions)

    def getSuccessor(image_np: np.array, nextAction: action.ActionType) -> np.array:
        """Takes an image as an np.arrow and returns a new image after the action has been performed"""

        return action.take_action(image_np, nextAction)

    getSuccessor = staticmethod(getSuccessor)

