
import numpy as np
from copy import deepcopy
from agents.agent import Agent
from store import register_agent
from collections import deque, defaultdict
import time

@register_agent("try_agent")
class TryAgent(Agent):

    def __init__(self):
        super(TryAgent, self).__init__()
        self.name = "TryAgent"
        self.autoplay = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        tic = time.perf_counter()
        state = BoardState(
            chess_board,
            my_pos, 
            adv_pos,
            max_step,
            turn=0
        )
        tree = MCTSNode(state)
        pos_r, pos_c, wall = tree.bestMove(iterations=100)
        # print("Try agent moved in {:.2f}".format(time.perf_counter() - tic))
        return (pos_r, pos_c), wall

class BoardState():
    '''Used both as an information container and a simulator for now'''
    # Moves (Up, Right, Down, Left)
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    # Opposite Directions
    opposites = {0: 2, 1: 3, 2: 0, 3: 1}

    def __init__(self, board, p0_pos, p1_pos, max_step, turn, heavy_playout=False):
        self.chess_board = board.copy()
        self.board_size = board.shape[0]
        self.p0_pos = p0_pos                # us
        self.p1_pos = p1_pos                # adversary

        self.max_step = max_step
        self._result = None
        self.turn = turn
    
    def isValidPos(self, pos):
        return 0 <= pos[0] < self.board_size and 0 <= pos[1] < self.board_size
    
    def getPossibleMoves(self):
        '''
        BFS
        Finds all placeable walls at all reachable positions 
        within max_steps from current state.
        Returns a Deque of triplets (r,c,d) of available legal moves for p0.
        '''
        our_pos, adv_pos = self.p0_pos, self.p1_pos
        if self.turn:
            our_pos, adv_pos = adv_pos, our_pos
        actions = deque()
        start = our_pos
        queue = deque()
        visited = set()
        queue.append(start)
        visited.add(start)
        
        level = 0

        while len(queue)>0:
            if level > self.max_step:
                break
            # Avoid additional computation if no additional step allowed
            skipNeighbors = level == self.max_step

            # For all nodes in current level
            for _ in range(len(queue)):     
                curr = queue.popleft()
                for dir in range(4):
                    if self.chess_board[curr[0], curr[1], dir]: # Wall
                        continue
                    
                    # Valid wall
                    actions.append((curr[0], curr[1], dir)) # we could save every other action only

                    if skipNeighbors:
                        continue

                    # Next position
                    move = self.moves[dir]
                    nextt = (curr[0] + move[0], curr[1] + move[1])

                    if self.isValidPos(nextt) and nextt not in visited \
                        and adv_pos != nextt:
                        queue.append(nextt)
                        visited.add(nextt)
            level += 1
        return actions  # deque of (r,c,d)

    def randomAction(self):
        '''Produce Random Walk for specified player (0 | 1)'''
        if self.turn:    # p1 turn
            temp_pos = deepcopy(self.p1_pos)
            adv_pos = self.p0_pos
        else:            # p0 turn
            temp_pos = deepcopy(self.p0_pos)
            adv_pos = self.p1_pos
        
        n_steps = np.random.randint(0, self.max_step+1)
        # Random n_steps walk
        for _ in range(n_steps):
            r, c = temp_pos
            dir = np.random.randint(0,4)
            m_r, m_c = self.moves[dir]
            temp_pos = (r + m_r, c + m_c) # TODO boundary conditions?

            # Special Case if enclosed by Adversary
            k = 0
            while self.chess_board[r, c, dir] or temp_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = self.moves[dir]
                temp_pos = (r + m_r, c + m_c)

            if k > 300:
                temp_pos = self.p1_pos if self.turn == 1 else self.p0_pos
                break
        
        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = temp_pos
        cell = self.chess_board[r, c, :]
        if np.all(cell):
            print("loop at", r, c, f"p{self.turn} ({temp_pos})", f"p{1 - self.turn} ({adv_pos})")
        while self.chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)
        
        return r, c, dir

    def place_wall(self, r, c, dir):
            # Set the wall to True
            self.chess_board[r, c, dir] = True
            # Set the opposite wall to True
            move = self.moves[dir]
            self.chess_board[r + move[0], c + move[1], self.opposites[dir]] = True

    def updateBoard(self, action):
        '''Update own state (board, positions, turn) from making the specified move'''
        self.place_wall(action[0], action[1], action[2])
        
        if self.turn == 0:
            self.p0_pos = (action[0], action[1])
        else:
            self.p1_pos = (action[0], action[1])
        # Switch turns
        self.turn ^= 1


    def nextBoardState(self, action):
        '''
        Return a new BoardState from making the specified move
        WARNING: updates internal state of Board
        '''
        copy = BoardState(self.chess_board, self.p0_pos, self.p1_pos, self.max_step, self.turn)
        copy.updateBoard(action)

        return copy
        
    def pseudoRandomAction(self):
        raise NotImplementedError()

    def isGameOver(self):
        # Union-Find, O(n lg*n), better than O(nlogn)
        father = dict()
        for r in range(self.board_size):
            for c in range(self.board_size): # Every cell is its own parent in the beginning
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
        
    def runToCompletion(self, start_turn = 1):
        """Create a copy of the Board and update until endgame"""
        simul = BoardState(self.chess_board, self.p0_pos, self.p1_pos, self.max_step, turn=start_turn)
        while not simul.isGameOver():
            action = simul.randomAction()   # random walk
            simul.updateBoard(action)

        return simul._result

    def printState(self):
        print("p0", self.p0_pos)
        print("p1", self.p1_pos)
        print("to move:", self.turn)
        print("board")
        print(self.chess_board)
        print(",")

# Use nodes for both our turn and the other player's turn. Like MiniMax!

'''
MONTE CARLO TREE SEARCH
Nodes are game states.
1. Selection: traverse the tree down to a leaf, picking the best child based on Estimated Reward 
2. Expansion: produce a child node by selecting an arbitrary legal move from current node
3. Simulation: perform a number of playouts from the expanded node. Can be heavy or light playouts (random or pseudorandom moves until completion).
4. Backpropagation: update wins and visit counts in all nodes in the expanded path in the tree.
'''
class MCTSNode():
    def __init__(self, state, parent=None, parentAction=None) -> None:
        self.state = state                  # BoardState
        self.turn = state.turn
        self.parent = parent                # pointer to parent Node
        self.parentAction = parentAction    # (r,c,d)

        self.children = deque()     # Deque[MCTSNode]
        self.n_visits = 0           # times visited or ni

        _outcomes = defaultdict(int)
        _outcomes[1] = 0     # wins
        _outcomes[-1] = 0    # losses
        self._outcomes = _outcomes
        
        # Compute all possible moves
        self._untriedMoves = self.state.getPossibleMoves()

    def q(self):
        return (self._outcomes[1] - self._outcomes[-1]) / self.n_visits

    def expandOne(self):
        """
            From present state, generate child state by taking
            ONE possible action. Assumes untriedMoves len > 1
            Instantiates and returns a new MCTSNode.
        """
        # Pick a legal unexplored move
        action = self._untriedMoves.pop()       # pop moves with longer steps first

        next_board_state = self.state.nextBoardState(action)    # returns BoardState after action
        child_node = MCTSNode(
            next_board_state, parent=self, parentAction=action)

        self.children.append(child_node)
        return child_node

    def isTerminalNode(self):
        return self.state.isGameOver()

    def playout(self):
        """
        Simulate an game from here until endgame.
        Win:    1
        Loss:   -1
        Tie:    0
        """
        simul = BoardState(
            self.state.chess_board, 
            self.state.p0_pos, self.state.p1_pos, 
            self.state.max_step, 
            self.state.turn)
        
        while not simul.isGameOver():
            action = simul.randomAction()   # random walk
            simul.updateBoard(action)       # update state and toggle turn

        return simul._result

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

    def hasTriedAllMoves(self):
        # "Cutting actions by half can give us incredible savings"
        return len(self._untriedMoves) == 0

    def selectBestChild(self, C=0.1):
        # Exploitation vs Exploration: vi + C*sqrt[ln(N) / ni]
        scores = [
            child.q() + \
            C * np.sqrt((np.log(self.n_visits) / child.n_visits)) # why 2*?
            for child in self.children
        ]
        # Approach minimax tree
        if self.turn == 0:
            return self.children[np.argmax(scores)]     # max turn
        else:
            return self.children[np.argmin(scores)]     # min turn

    def treePolicy(self):
        """
        Selection: return the node to run simulations from
        """
        stateNode = self

        while not stateNode.isTerminalNode():
            # keep exploring new moves until we have tried them all
            if stateNode.hasTriedAllMoves():    # go deep
                stateNode = stateNode.selectBestChild()
            else:                               # go wide
                return stateNode.expandOne()
    
        return stateNode
    
    def bestMove(self, iterations):
        for p in range(iterations):
            node = self.treePolicy()    # Selection and Expansion
            result = node.playout()     # Simulation
            node.backpropagate(result)  # Backpropagation
            # print("\tsimulation", p, "=", result)

        return self.selectBestChild(C=0).parentAction # pure exploitation
