'''
MONTE CARLO TREE SEARCH
Nodes are game states.

1. Selection: traverse the tree down to a leaf, picking the best child based on Estimated Reward 
2. Expansion: produce a child node by selecting an arbitrary legal move from current node
3. Simulation: perform a number of playouts from the expanded node. Can be heavy or light playouts (random or pseudorandom moves until completion).
4. Backpropagation: update wins and visit counts in all nodes in the expanded path in the tree.
'''

import numpy as np
from collections import defaultdict

class MCTSNode():
    def __init__(self, state, parent=None, parentAction=None) -> None:
        """ Create a node everytime we expand our existing MCTS """
        board, pos0, pos1, max_steps = state
        self.state = StateSimulator(board, pos0, pos1, max_steps=max_steps)
        
        self.parent = parent                # pointer to parent Node
        self.parentAction = parentAction    # client code should use .parentAction of best child

        self.children = []      # subsequent game states
        self.n_visits = 0       # times visited or ni

        _outcomes = defaultdict(int)
        _outcomes[1] = 0     # wins
        _outcomes[-1] = 0    # losses
        self._outcomes = _outcomes
        self.triedActions = set()

    def getUntriedAction(self):
        action = self.state.getAction()
        while action in self.triedActions:
            action = self.state.getAction(0)
        self.triedActions.add(action)
        return action

    def quality(self):
        """(Wins - Losses)/Ni"""
        return (self._outcomes[1] - self._outcomes[-1]) / self.n_visits

    def expandOne(self):
        """
            From present state, generate child state by taking
            ONE possible action. Return the new node
        """
        action = self.getUntriedAction()        # action for player 0
        next_state = self.state.move(action)    # board, pos0, pos1
        child_node = MCTSNode(
            next_state, parent=self, parentAction=action)
        self.children.append(child_node)
        return child_node

    def isFinalState(self):
        return self.state.isEndGame()

    def playout(self):
        """
        Simulate an entire game until there is an outcome.
        Win:    1
        Loss:   -1
        Tie:    0
        """
        current_playout_state = self.state

        # Play until game ends
        while not current_playout_state.isEndGame():
            action = current_playout_state.getAction()
            current_playout_state = current_playout_state.move(action)

        return current_playout_state.result()

    def backpropagate(self, result):
        """
        Recursively update statistics for all nodes from this node
        to the root
        """
        # Result is either 1, -1 or 0
        self.n_visits += 1
        self._outcomes[result] += 1
        if self.parent:
            self.parent.backpropagate(result)

    def isExpandedEnough(self):
        # "Cutting actions by half can give us incredible savings"
        # return len(self._untried_actions) == 0
        return self.triedActions == self.legalMoves # TODO: important,on init, create all legal moves

    def selectBestChild(self, C=0.1):
        # Assumes there is at least one child in self.children
        # Exploitation vs Exploration tradeoff controlled by C
        # score is computed:  vi + C*sqrt[ln(N) / ni]
        scores = [
            child.quality() + \
            C * np.sqrt((2 * np.log(self.n_visits) / child.n_visits)) # why 2*?
            for child in self.children
        ]
        return self.children[np.argmax(scores)]
    
    def _treePolicy(self, fully_expand=False):
        """
        Selection: return the node to run simulations from
        """
        current = self
        if fully_expand: # BFS-like: Visits all children before going deeper
            
            while not current.isFinalState():
                if not current.isExpandedEnough(): 
                    return current.expandOne()
                else:
                    # Pick one or more 
                    current = current.selectBestChild()                
        
        return current
    
    def bestMove(self):
        iterations = 100         # tune for time performance

        for i in range(iterations):
            node = self._treePolicy()   # Selection and Expansion
            result = node.playout()     # Simulation
            node.backpropagate(result)  # Backpropagation

        return self.selectBestChild(C=0.)
