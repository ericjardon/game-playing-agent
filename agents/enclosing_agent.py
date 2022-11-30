# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
from math import sqrt
from collections import deque
import heapq

def heuristic(ax, ay, bx, by):
    """Returns manhattan dist between two 2d points"""
    return abs(ax-bx) + abs(ay-by)

def getPath(endNode):
    path = deque()
    curr = endNode
    # Traverse path back to root (start)
    while curr is not None:
        path.appendleft(curr.pos)
        curr = curr.parent
    
    return list(path)



class Node():
    def __init__(self, parent=None, pos=None) -> None:
        self.parent = parent
        self.pos = pos 
        self.g = 0
        self.h = 0
        self.f = 0
        # f = g+h
    def __eq__(self, other):
        return self.pos == other.pos

    # define ordering for purposes of heap queue (heapq)
    def __lt__(self, other):
      return self.f < other.f
    def __gt__(self, other):
      return self.f > other.f

@register_agent("enclosing_agent")
class EnclosingAgent(Agent):
    """
    This agent searches for the other player in the grid and tries to enclose it.
    It does A* search to find shortest path to the other player and places a wall
    as close to the other player as it can.

    - Search the other player with shortest path
    - Place walls as close to the player as you can
    - Prefer to place walls facing the center of the game, taking lenX/2 and lenY/2 to know center position
    """

    def __init__(self):
        super(EnclosingAgent, self).__init__()
        self.name = "EnclosingAgent"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    def search(self, board, startPos, endPos):
        """
        Return a list of pairs (coords) from start to end in the
        given 3d board, where dimX is rows, dimY is cols, dimZ is direction
        > endPos is guaranteed to be reachable
        """
        if startPos == endPos: # should not be possible, but sanity check
            return [endPos]
    
        # Init start node
        start = Node(None, startPos)

        # Init end node
        end = Node(None, endPos)

        # Init priority queue and visited set
        queue = []          # open list
        seen = set()        # closed list

        heapq.heapify(queue)
        heapq.heappush(queue, start)

        MOVES = [(-1,0), (0,1), (1,0), (0,-1)] # u r d l

        while len(queue):
            curr = heapq.heappop(queue)
            seen.add(curr.pos)
            row, col = curr.pos

            if curr == end:
                return getPath(curr)
            
            # Traverse Neighbors
            for dir, move in enumerate(MOVES):

                # Check the way is clear
                if board[row, col, dir] == 1:
                    continue

                i = row + move[0]
                j = col + move[1]

                # Check position is within bounds
                if not (0 <= i < board.shape[0] and 0 <= j < board.shape[1]):
                    continue
                
                # Create neighbor Node
                neighbor = Node(curr, (i,j))

                # Expand only unseen nodes
                if neighbor.pos in seen:
                    continue 
                
                # Compute f score
                neighbor.g = curr.g + 1                 # cost so far
                neighbor.h = heuristic(row, col, i, j)  # manhattan
                neighbor.f = neighbor.g + neighbor.h    # f-score

                # (If present in the open list it does not matter, cost cannot be better bc consistent heuristic)
                heapq.heappush(queue, neighbor)
        
        print("WARN: A* did not find a path. Game ended")
        return None

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        - chess_board: a numpy array of shape (x_max, y_max, 4) up down left right
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        Return a tuple of ((x, y), dir),
            (x, y) is the next position,
            dir is the direction of the wall to place in position.
        """

        # Find path to adv_pos
        # Place wall as close to adv as possible
        # traversing the path while not end and  steps < max_steps
        # Guaranteed there exists a path to adv, otherwise endgame

        path_to_adv = self.search(chess_board, my_pos, adv_pos)
        if path_to_adv is None:
            # print("WARN: return dummy move")
            # dummy return
            return my_pos, self.dir_map["u"]
        # print("Path to adv")
        # print(path_to_adv)

        # Place wall at closest possible position
        if len(path_to_adv) - 2 <= max_step:
            my_pos = path_to_adv[-2]

            if adv_pos[0] > my_pos[0]:
                wall = 'd'
            elif adv_pos[0] < my_pos[0]:
                wall = 'u'
            elif adv_pos[1] > my_pos[1]:
                wall = 'r'
            else:
                wall = 'l'
            
            return my_pos, self.dir_map[wall]
        
        else:
            my_pos = path_to_adv[max_step]
            # Place wall in next step of path
            next_pos = path_to_adv[max_step+1]
            if next_pos[0] > my_pos[0]:
                wall = 'd'
            elif next_pos[0] < my_pos[0]:
                wall = 'u'
            elif next_pos[1] > my_pos[1]:
                wall = 'r'
            else:
                wall = 'l'
            
            return my_pos, self.dir_map[wall]

            
        