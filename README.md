# CS4100-Final-Project

Report on the logic behind this project <https://docs.google.com/document/d/10nPyV4zQj_lkJnf0TAUWtkKHMO-LboquivMf2nv-hII/edit?usp=sharing>

### Contributor: Krishna Kothandaraman, Haobo Liu, Qi Li

## To run the application in command lines (with python 3):

run world.py
mandotory argument:
    -g: (file path)the sample image you want to extract the color palette and be your reference later
    -e: (file path)the image you intend to edit
    note: those two file must be in .jpg format for this application

optional argument:
    -d: Discount on future (default %default)
    -r: livingReward, the Reward for living for a time step
    -n: How often action results in unintended direction
    --epsilon: Chance of taking a random action in q-learning
    -l: TD learning rate 
    -i: Number of rounds of value iteration
    -k: Number of epsiodes of the MDP to run

Input: you should have two jpg images ready for running this application. one is the targeted image you intend to copy its style, the other is the picture you want the algorithm to edit for you.


