import curses
import time
from environment import Environment
from agent import Agent

def curses_routine(win):
    env = Environment()
    agent = Agent(env)
    t = 0

    while True:


        env.step(t, agent)
        t += 1

        if t % 1000 == 0:
            env.new_goal()

        win.clear()
        env.display(win)
        win.refresh()

        # if env.last_reinforcement > 0:
            # time.sleep(0.01)
            # pass
        # win.getch()

if __name__ == '__main__':
    curses.wrapper(curses_routine)