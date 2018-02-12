import numpy as np
import curses
import neural

class Environment(neural.Rewarder):
    def __init__(self):
        self.goal = self.new_goal()
        self.agent = None
        self.last_reward = 0
        self.last_reinforcement = 0
        self.score = 0
        self.t = 0
        self.reinforcement = 0

    def new_goal(self):
        return np.random.random(2)

    def step(self, t, agent):
        self.agent = agent
        self.t = t

        sensor_positions = self.agent.sensors
        sensor_measurements = []
        for sensor in sensor_positions:
            sensor_measurements.append(self.reward_field(sensor))

        sensor_measurements = np.array(sensor_measurements)
        sensor_measurements = (sensor_measurements >= sensor_measurements.max()) * np.ones(4)

        action = agent.step(t, sensor_measurements, plot_net=False)

        directions = sensor_positions - agent.position
        for i in range(4):
            if action[i] > 0:
                agent.position += directions[i]

        agent.position = np.clip(agent.position, a_min=np.zeros(2), a_max=np.ones(2))

        self.evaluate()

        agent.reinforce()

    def reward_field(self, at_location):
        dist = np.sqrt(((self.goal - at_location) ** 2).sum())

        if dist == 0:
            dist = 0.00001

        return 1 / dist

    def evaluate(self):
        self.last_reinforcement = self.reinforcement
        reward = self.reward_field(self.agent.position)

        self.reinforcement = (reward - self.last_reward) * 10
        self.last_reward = reward

    def display(self, win):
        h, w = win.getmaxyx()
        win_size = np.array([h - 2, w - 2])
        goal_pos = self.goal * win_size
        agent_pos = self.agent.position * win_size

        win.addstr(0, 0, '%d %d' % (self.t, self.score))
        win.addstr(1, 0, str(self.last_reinforcement))
        win.addstr(2, 0, str(self.last_reward))

        win.addch(int(goal_pos[0]), int(goal_pos[1]), 'x')
        win.addch(int(agent_pos[0]), int(agent_pos[1]), '*')

        if np.sqrt(((self.agent.position - self.goal) ** 2).sum()) < 0.1:
            self.goal = self.new_goal()
            self.last_reward = 0
            self.score += 1

