# CS4100-Final-Project

Report on the logic behind this project can be found [here](<https://docs.google.com/document/d/10nPyV4zQj_lkJnf0TAUWtkKHMO-LboquivMf2nv-hII/edit?usp=sharing>)

### Contributor: Krishna Kothandaraman, Haobo Liu, Qi Li

## To run the application in command lines (with python 3):

- Make the program executable
```
chmod +x world.py
```

- run the editing process from command line
```
./world -g <path/to/exemplar/image> -e <path/to/edit/image>
```
CLI arguments
- Mandatory 
  - -g: (file path)the sample image you want to extract the color palette and be your reference later 
  - -e: (file path)the image you intend to edit
    __note: those two file must be in .jpg format for this application__

- optional arguments:
  - -d: Discount on future (default %default)
  - -r: livingReward, the Reward for living for a time step
  - -n: How often action results in unintended direction
  - --epsilon: Chance of taking a random action in q-learning
  - -l: TD learning rate 
  - -i: Number of rounds of value iteration
  - -k: Number of epsiodes of the MDP to run

Edited image will be saved as edited.jpg in the same directory that the program runs in.

