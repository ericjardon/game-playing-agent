# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import time
import numpy as np
from copy import deepcopy
from collections import deque

@register_agent("eric_agent")
class EricAgent(Agent):

    def __init__(self):
        super(EricAgent, self).__init__()
        self.name = "EricAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def step(self, chess_board, my_pos, adv_pos, max_step):
        current_state = BoardState(chess_board, my_pos, adv_pos, max_steps=max_step)

        root = MCTSNode(current_state)
        pos, wall = root.bestMove(iterations=100)

        return pos, wall

class BoardState():
    '''Used both as an information container and a simulator for now'''
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))  # u, r, d, l

    def __init__(self, board, my_pos, adv_pos, max_step, heavy_playout=False):
        # board is a 3d numpy array
        self.chess_board = deepcopy(board)
        self.board_size = board.shape[0]
        self.my_pos = my_pos
        self.adv_pos = adv_pos

        self.max_step = max_step
        self._result = None            # computed at endgame
        self.turn = 0                  # p0 turn
        if self.heavy_playout:      # TODO: review
            self.getAction = self.pseudoRandomAction

    
    def validatePos(self, pos):
        return 0 <= pos[0] < self.board_size and 0 <= pos[1] < self.board_size
    
    def getAllPossibleMoves(self):
        '''
        BFS
        Find all placeable walls at all reachable positions 
        within max_steps from current state.
        Returns a Deque of triplets (r,c,d) of available legal moves for p0.
        '''

        actions = deque()
        start = self.my_pos
        queue = deque()
        visited = set()
        queue.append(start)
        visited.add(start)
        
        level = 0

        while len(queue)>0:
            if level > self.max_steps:
                break
            
            # Avoid additional computation if no additional step allowed
            skipNeighbors = level == self.max_steps

            # For all nodes in current level
            for _ in range(len(queue)):     
                curr = queue.popleft()
                for dir in range(4):
                    if self.chess_board[curr[0], curr[1], dir]: # Wall
                        continue
                    
                    # Valid wall
                    actions.append((curr[0], curr[1], dir))

                    if skipNeighbors:
                        continue

                    # Next position
                    move = self.moves[dir]
                    nextt = (curr[0] + move[0], curr[1] + move[1])

                    if self.validatePos(nextt) and nextt not in visited \
                        and self.adv_pos != nextt:
                        queue.append(nextt)
                        visited.add(nextt)

            level += 1

        print("actions #",len(actions)) 
        print(actions)
        return actions


    def getAction(self, turn):
        '''Return  valid action for specified player turn (0 | 1)'''
        if turn:    # p1 turn
            temp_pos = deepcopy(self.adv_pos)
            temp_adv_pos = self.my_pos
        else:       # p0 turn
            temp_pos = deepcopy(self.my_pos)
            temp_adv_pos = self.adv_pos
        
        n_steps = np.random.randint(0, self.max_step+1)
        # Random n_steps walk
        for _ in range(n_steps):
            r, c = temp_pos
            dir = np.random.randint(0,4)
            m_r, m_c = self.moves[dir]
            temp_pos = (r + m_r, c + m_c) # TODO boundary conditions?

            # Special Case if enclosed by Adversary
            k = 0
            while self.chess_board[r, c, dir] or temp_pos == temp_adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = self.moves[dir]
                temp_pos = (r + m_r, c + m_c)

            if k > 300:
                temp_pos = self.adv_pos if turn == 1 else self.my_pos
                break
        
        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = temp_pos
        while self.chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)
        
        return r, c, dir

    def move(self, action, turn=0):
        '''Return a new BoardState from making the specified move'''
        new_board = deepcopy(self.chess_board)
        new_board[action[0], action[1], action[2]] = True

        if turn == 0:
            my_pos = (action[0], action[1])
            adv_pos = tuple(self.adv_pos)
        else:
            adv_pos = (action[0], action[1])
            my_pos = tuple(self.my_pos)

        return BoardState(new_board, my_pos, adv_pos, self.max_step)

    def updateBoard(self, action, turn):
        '''Update own state from making the specified move'''
        self.chess_board[action[0], action[1], action[2]] = True
        if turn == 0:
            self.my_pos = (action[0], action[1])
        else:
            self.adv_pos = (action[0], action[1])
        
    def pseudoRandomAction(self):
        raise NotImplementedError()

    def isEndGame(self):
        # Union-Find, O(n lg*n), better than O(nlogn)
        father = dict()
        
        for r in range(self.board_size):
            for c in range(self.board_size):
                # Every cell is its own parent in the beginning
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.board_size):
            for c in range(self.board_size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if self.chess_board[r, c, dir + 1]:
                        # wall, i.e. cell is disjoint
                        continue
                    # Find representative of both positions
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        # can reach pos_b from pos_a
                        union(pos_a, pos_b)

        for r in range(self.board_size):
            for c in range(self.board_size):
                find((r, c))

        # Find representatives of players' positions
        p0_r = find(tuple(self.p0_pos))
        p1_r = find(tuple(self.p1_pos))

        if p0_r == p1_r:
            return False

        # Tally up scores
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)

        if p0_score > p1_score:
            res = 1          # we won
        elif p0_score < p1_score:
            res = -1         # adv won
        else:
            res = 0          # tie
        
        self._result = res
        return True
        
    def runToCompletion(self, start_turn = 0):
        turn = start_turn
        while not self.isEndGame():
            action = self.getAction(turn)   # random walk
            self.updateBoard(action, turn)  
            turn ^= 1
        
        return self._result


'''
MONTE CARLO TREE SEARCH
Nodes are game states.

1. Selection: traverse the tree down to a leaf, picking the best child based on Estimated Reward 
2. Expansion: produce a child node by selecting an arbitrary legal move from current node
3. Simulation: perform a number of playouts from the expanded node. Can be heavy or light playouts (random or pseudorandom moves until completion).
4. Backpropagation: update wins and visit counts in all nodes in the expanded path in the tree.
'''

import numpy as np
from collections import defaultdict, deque

class MCTSNode():
    def __init__(self, state, parent=None, parentAction=None) -> None:
        """ Create a node everytime we expand our existing MCTS """
        self.state = state # BoardState
        
        self.parent = parent                # pointer to parent Node
        self.parentAction = parentAction    # client code should use .parentAction of best child

        self.children = deque()     # subsequent game states
        self.n_visits = 0           # times visited or ni

        _outcomes = defaultdict(int)
        _outcomes[1] = 0     # wins
        _outcomes[-1] = 0    # losses
        self._outcomes = _outcomes
        
        # Compute all possible moves
        self._untriedMoves = self.state.getAllPossibleMoves()

    def quality(self):
        return (self._outcomes[1] - self._outcomes[-1]) / self.n_visits

    def expandOne(self):
        """
            From present state, generate child state by taking
            ONE possible action.
            Instantiates a new MCTSNode and returns it.
        """
        # Pick a legal unexplored move
        action = self._untriedMoves.pop()       # pop moves with longer steps first
        # Compute next state
        next_state = self.state.move(action)    # BoardState
        child_node = MCTSNode(
            next_state, parent=self, parentAction=action)
        
        self.children.append(child_node)
        return child_node

    def isFinalState(self):
        return self.state.isEndGame()

    def playout(self):
        """
        Simulate an game from here until endgame.
        Win:    1
        Loss:   -1
        Tie:    0
        """
        boardState = self.state

        # Play until game ends
        result = boardState.runToCompletion()
        return result

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

    def isFullyExpanded(self):
        # "Cutting actions by half can give us incredible savings"
        return len(self._untried_actions) == 0

    def selectBestChild(self, C=0.1):
        # Exploitation vs Exploration score is computed:  vi + C*sqrt[ln(N) / ni]
        scores = [
            child.quality() + \
            C * np.sqrt((2 * np.log(self.n_visits) / child.n_visits)) # why 2*?
            for child in self.children
        ]
        return self.children[np.argmax(scores)]
    
    def treePolicy(self, fully_expand=False):
        """
        Selection: return the node to run simulations from
        """
        current = self
        if fully_expand: # Visits all children before going deeper
            while not current.isFinalState():
                if current.isFullyExpanded(): 
                    current = current.selectBestChild()      
                else:
                    return current.expandOne()          
        
        return current
    
    def bestMove(self, iterations):

        for _ in range(iterations):
            node = self.treePolicy()   # Selection and Expansion
            result = node.playout()     # Simulation
            node.backpropagate(result)  # Backpropagation

        return self.selectBestChild(C=0.)
