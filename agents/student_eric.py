# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import time


@register_agent("student_eric")
class StudentEric(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentEric, self).__init__()
        self.name = "StudentEric"
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