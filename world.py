#!/usr/bin/env python3
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
import argparse
import logging
import pathlib
import random
import sys
from typing import List

import cv2
import numpy as np

import action
import mdp
import reward
from PIL import Image
import featureExtractor
import QlearningAgent

TERMINAL_DISTANCE = 100


def getPaletteFromImage(image: np.array) -> np.array:
    return featureExtractor.Train(Image.fromarray(image)).train()


class ImageWorld(mdp.MarkovDecisionProcess):
    """
      Image world
    """

    def __init__(self, image: np.array, goalPalette):
        # target image and palette in this world that the agent wants to reach
        self.image = image
        self.goalPalette = goalPalette

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
        # return 0 at terminal state
        if self.isTerminal(state):
            return []
        return [nextAction for nextAction in action.ActionType]

    def getReward(self, state, action: action.ActionType, nextState: np.array) -> float:
        """
        Get reward for state, action, nextState transition.
        Note that the reward depends only on the state being
        departed (as in the R+N book examples, which more or
        less use this convention).
        """
        # initialize image object for feature extraction
        nextImage = Image.fromarray(nextState)
        # extract palette of new image
        nextPalette = getPaletteFromImage(np.asarray(nextImage))
        # get reward for new palette. Negative because we want to actually minimize the euclidean distance
        return -reward.getRewardFromPalettes(self.goalPalette, nextPalette)

    def isTerminal(self, nextImage: np.array):
        """
        Only the TERMINAL_STATE state is *actually* a terminal state.
        The other "exit" states are technically non-terminals with
        a single action "exit" which leads to the true terminal state.
        This convention is to make the grids line up with the examples
        in the R+N textbook.
        """
        # Terminate if Euclidean distance is <= certain amount
        r = reward.getRewardFromPalettes(self.goalPalette, getPaletteFromImage(nextImage))
        print(f"Got reward {r}")
        return r <= TERMINAL_DISTANCE

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

    def reset(self):
        self.state = self.world.getStartState()


def runEpisode(agent, environment, discount, decision, message, episode):
    returns = 0
    totalDiscount = 1.0
    environment.reset()
    if 'startEpisode' in dir(agent): agent.startEpisode()
    message("BEGINNING EPISODE: " + str(episode) + "\n")
    i = 0
    while i < 5:

        # DISPLAY CURRENT STATE
        state = environment.getCurrentState()

        # END IF IN A TERMINAL STATE
        actions = environment.getPossibleActions(state)
        if len(actions) == 0:
            message("EPISODE " + str(episode) + " COMPLETE: RETURN WAS " + str(returns) + "\n")
            return returns

        # GET ACTION (USUALLY FROM AGENT)
        action = decision(state)
        if action is None:
            raise Exception('Error: Agent returned None action')

        # EXECUTE ACTION
        nextState, reward = environment.doAction(action)
        message("Took action: " + str(action) +
                "\nGot reward: " + str(reward) + "\n")
        # UPDATE LEARNER
        if 'observeTransition' in dir(agent):
            agent.observeTransition(state, action, nextState, reward)

        returns += reward * totalDiscount
        totalDiscount *= discount

        i += 1

    message("EPISODE " + str(episode) + " COMPLETE: RETURN WAS " + str(returns) + "\n")
    return returns


if __name__ == '__main__':
    # get goal image file
    p = argparse.ArgumentParser(description="Image Processor")
    p.add_argument('-g', dest="goal_image", type=pathlib.Path, help="Path to goal image", required=True)
    p.add_argument('-e', dest="edit_image", type=pathlib.Path, help="Path to image to edit", required=True)
    p.add_argument('-d', '--discount', action='store',
                   type=float, dest='discount', default=0.9,
                   help='Discount on future (default %default)')
    p.add_argument('-r', '--livingReward', action='store',
                   type=float, dest='livingReward', default=0.0,
                   metavar="R", help='Reward for living for a time step (default %default)')
    p.add_argument('-n', '--noise', action='store',
                   type=float, dest='noise', default=0.2,
                   metavar="P", help='How often action results in ' +
                                     'unintended direction (default %default)')
    p.add_argument('--epsilon', action='store',
                   type=float, dest='epsilon', default=0.3,
                   metavar="E", help='Chance of taking a random action in q-learning (default %default)')
    p.add_argument('-l', '--learningRate', action='store',
                   type=float, dest='learningRate', default=0.5,
                   metavar="P", help='TD learning rate (default %default)')
    p.add_argument('-i', '--iterations', action='store',
                   type=int, dest='iters', default=10,
                   metavar="K", help='Number of rounds of value iteration (default %default)')
    p.add_argument('-k', '--episodes', action='store',
                   type=int, dest='episodes', default=1,
                   metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    goalImagePath = pathlib.Path(args.goal_image)
    editImagePath = pathlib.Path(args.edit_image)

    if not goalImagePath.exists() or goalImagePath.suffix != ".jpg":
        print("ERROR: Invalid goal Image path or format")
        sys.exit(1)
    if not editImagePath.exists() or editImagePath.suffix != ".jpg":
        print("ERROR: Invalid edit Image path or format")
        sys.exit(1)
    # load images
    goalImageObj = Image.open(goalImagePath)
    editImageObj = Image.open(editImagePath)

    logging.info("Beginning training")
    # train feature extractor
    t = featureExtractor.Train(goalImageObj)
    # get palette of goal image
    cluster_centers = t.train()
    logging.info("Extracted features")

    imgWorld = ImageWorld(np.asarray(editImageObj), cluster_centers)
    env = WorldEnvironment(imgWorld)

    ###########################
    # GET THE AGENT
    ###########################

    # env.getPossibleActions, opts.discount, opts.learningRate, opts.epsilon
    # simulationFn = lambda agent, state: simulation.GridworldSimulation(agent,state,mdp)
    actionFn = lambda state: env.getPossibleActions(state)
    qLearnOpts = {'gamma': args.discount,
                  'alpha': args.learningRate,
                  'epsilon': args.epsilon,
                  'actionFn': actionFn}
    a = QlearningAgent.QLearningAgent(**qLearnOpts)

    message = lambda x: logging.info(x)
    ###########################
    # RUN EPISODES
    ###########################
    # DISPLAY Q/V VALUES BEFORE SIMULATION OF EPISODES

    # RUN EPISODES
    if args.episodes > 0:
        print()
        print("RUNNING", args.episodes, "EPISODES")
        print()
    returns = 0
    for episode in range(1, args.episodes + 1):
        returns += runEpisode(a, env, args.discount, a.getAction, message, episode)
    if args.episodes > 0:
        print(f"\nAVERAGE RETURNS FROM START STATE: {str((returns + 0.0) / args.episodes)}\n\n")

    print("Iterations over, palette image as edited.jpg")
    imgObj = Image.fromarray(env.getCurrentState(), "RGB")
    t = featureExtractor.Train(Image.fromarray(env.getCurrentState()))
    palette = t.train()
    featureExtractor.generate_palette_image(palette, "edited_palette")
    imgObj.save("edited_image.jpg")
