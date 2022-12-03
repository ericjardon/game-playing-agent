# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import time
import numpy as np
from copy import deepcopy

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
        self.firstMove = True

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        First step
        """
        # Compute for 30 seconds...
        time.sleep(30)
        self.step = self.nextSteps
        self.firstMove = False
        return my_pos, self.dir_map["u"]

    def nextSteps(self, chess_board, my_pos, adv_pos, max_step):
        print("next Step, first move?", self.firstMove)
        return my_pos, self.dir_map["u"]

class StateSimulator():
    '''
    Can we reuse the World class?
    So we don't have to re-implement
    '''
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(self, board, my_pos, adv_pos, max_step, heavy_playout=False):
        # board is a 3d numpy array
        self.chess_board = board #deepcopy(board)
        self.board_size = board.shape[0]
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.max_step = max_step
        self._result = None
        self.turn = 0               # our turn
        if self.heavy_playout:
            self.getAction = self.pseudoRandomAction
    
    def getAction(self, turn):
        return self.randomValidAction(turn)

    def randomValidAction(self, turn):
        # Return a valid action for current player
        # TODO: check boundary conditions?
        if turn:
            temp_pos = deepcopy(self.adv_pos)
        else:
            temp_pos = deepcopy(self.my_pos)
        
        n_steps = np.random.randint(0, self.max_step+1)
        # Do a random walk of n_steps
        for _ in range(n_steps):
            r, c = temp_pos
            dir = np.random.randint(0,4)
            m_r, m_c = self.moves[dir]
            temp_pos = (r + m_r, c + m_c)

            # Special Case if enclosed by Adversary
            k = 0
            while self.chess_board[r, c, dir] or temp_pos == self.adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = self.moves[dir]
                temp_pos = (r + m_r, c + m_c)

            if k > 300:
                temp_pos = self.my_pos
                break
        
        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = temp_pos
        while self.chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)
        
        return temp_pos, dir

    def move(self):
        # Update state
        self.board
        pass
        
    def pseudoRandomAction(self):
        pass

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
            return False, None # no result

        # Determine winner
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)

        if p0_score > p1_score:
            res = 1          # we won
        elif p0_score < p1_score:
            res = -1         # adv won
        else:
            res = 0          # tie
        
        self._result = res
        return True, res
        
    def result(self):
        return self._result