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
# Modified by: Krishna Kothandaraman

import random
from typing import List

import numpy as np

import action
import mdp
import Reward
from PIL import Image
import featureExtractor

TERMINAL_DISTANCE = 100


def getPaletteFromImage(image) -> np.array:
    t = featureExtractor.Train(image)
    return t.train()


class ImageWorld(mdp.MarkovDecisionProcess):
    """
      Image world
    """

    def __init__(self, image: Image):
        # target image and palette in this world that the agent wants to reach
        self.image = image
        self.imagePalette = getPaletteFromImage(self.image)

        # parameters
        self.livingReward = 0.0
        self.noise = 0.2

    def setLivingReward(self, reward):
        """
        The (negative) reward for exiting "normal" states.
        Note that in the R+N text, this reward is on entering
        a state and therefore is not clearly part of the state's
        future rewards.
        """
        self.livingReward = reward

    def getStartState(self):
        """
        Return the start state of the MDP.
        """
        return self.image

    def setNoise(self, noise):
        """
        The probability of moving in an unintended direction.
        """
        self.noise = noise

    def getPossibleActions(self, state) -> List[action.ActionType]:
        """
        Returns list of valid actions for 'state'.
        Note that you can request moves into walls and
        that "exit" states transition to the terminal
        state under the special action "done".
        """
        # TODO: add terminal state
        return [nextAction for nextAction in action.ActionType]

    def getReward(self, state, action: action.ActionType, nextState: np.array):
        """
        Get reward for state, action, nextState transition.
        Note that the reward depends only on the state being
        departed (as in the R+N book examples, which more or
        less use this convention).
        """
        # initialize image object for feature extraction
        nextImage = Image.fromarray(nextState)
        # extract palette of new image
        nextPalette = getPaletteFromImage(nextImage)
        # get reward for new palette
        return Reward.getRewardFromPalettes(self.imagePalette, nextPalette)

    def isTerminal(self, nextImage: np.array):
        """
        Only the TERMINAL_STATE state is *actually* a terminal state.
        The other "exit" states are technically non-terminals with
        a single action "exit" which leads to the true terminal state.
        This convention is to make the grids line up with the examples
        in the R+N textbook.
        """
        # Terminate if Euclidean distance is <= certain amount
        return Reward.getRewardFromPalettes(self.imagePalette, getPaletteFromImage(nextImage)) <= TERMINAL_DISTANCE

    def getTransitionStatesAndProbs(self, state: np.array, imgAction: action.ActionType):
        """
        Returns list of (nextState, prob) pairs
        representing the states reachable
        from 'state' by taking 'action' along
        with their transition probabilities.
        """
        # only one next state possible with probability 1
        return [(action.take_action(state, imgAction), 1)]


class WorldEnvironment:

    def __init__(self, world: ImageWorld):
        self.world = world
        self.state = self.world.getStartState()

    def getCurrentState(self):
        return self.state

    def getPossibleActions(self, state):
        return self.world.getPossibleActions(state)

    def doAction(self, action):
        state = self.getCurrentState()
        (nextState, reward) = self.getRandomNextState(state, action)
        self.state = nextState
        return nextState, reward

    def getRandomNextState(self, state, action, randObj=None):
        if randObj is None:
            rand = random.random()
        else:
            rand = randObj.random()
        sum = 0.0
        successors = self.world.getTransitionStatesAndProbs(state, action)
        for nextState, prob in successors:
            sum += prob
            if sum > 1.0:
                raise Exception('Total transition probability more than one; sample failure.')
            if rand < sum:
                reward = self.world.getReward(state, action, nextState)
                return nextState, reward
        raise Exception('Total transition probability less than one; sample failure.')


def runEpisode(agent, environment, discount, decision, display, message, pause, episode):
    returns = 0
    totalDiscount = 1.0
    environment.reset()
    if 'startEpisode' in dir(agent): agent.startEpisode()
    message("BEGINNING EPISODE: " + str(episode) + "\n")
    while True:

        # DISPLAY CURRENT STATE
        state = environment.getCurrentState()
        display(state)
        pause()

        # END IF IN A TERMINAL STATE
        actions = environment.getPossibleActions(state)
        if len(actions) == 0:
            message("EPISODE " + str(episode) + " COMPLETE: RETURN WAS " + str(returns) + "\n")
            return returns

        # GET ACTION (USUALLY FROM AGENT)
        action = decision(state)
        if action == None:
            raise Exception('Error: Agent returned None action')

        # EXECUTE ACTION
        nextState, reward = environment.doAction(action)
        message("Started in state: " + str(state) +
                "\nTook action: " + str(action) +
                "\nEnded in state: " + str(nextState) +
                "\nGot reward: " + str(reward) + "\n")
        # UPDATE LEARNER
        if 'observeTransition' in dir(agent):
            agent.observeTransition(state, action, nextState, reward)

        returns += reward * totalDiscount
        totalDiscount *= discount
