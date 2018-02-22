import curses
import time
from environment import Environment
from agent import Agent


def curses_routine(win):
    env = Environment()
    agent = Agent(env)
    t = 0

    win.nodelay(True)
    curses.curs_set(0)

    show_plot = False
    while True:
        env.step(t, agent, env.reinforcement < 0)
        t += 1

        if t % 1000 == 0:
            env.new_goal()

        win.clear()
        env.display(win)
        win.refresh()

        # curses.napms(10)
        # c = win.getch()

        # if c is 'p':
        #     show_plot = True
        #
        # if c is 'n':
        #     show_plot = False

        # if env.last_reinforcement > 0:
        # time.sleep(0.01)
            # pass

        # win.getch()

if __name__ == '__main__':
    # time.sleep(7)
    curses.wrapper(curses_routine)