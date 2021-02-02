from time import sleep
from caller.caller import Caller
from environment.environment import Environment
from agent.agent import Agent
from random import randint
from config import NUM_FLOORS
from collections import deque

class Controller:
    AVG_CALL_DELTA = int(300 / (20 * NUM_FLOORS * 0.14))

    def __init__(self, environment, caller, agent):
        self.environment = environment
        self.caller = caller
        self.agent = agent

    def run(self):
        window_size = 10000
        last_rewards = deque(maxlen=window_size)
        t = 1
        while True:
            if randint(0, Controller.AVG_CALL_DELTA) >= Controller.AVG_CALL_DELTA / 3:
                floor, direction = self.caller.generate_call()
                self.environment.call(floor, direction)
            state = self.environment.sample_state()
            action = self.agent.get_action(state)
            reward = self.environment.perform_action(action)
            new_state = self.environment.sample_state()
            self.agent.perform_update(state, action, reward, new_state)
            
            last_rewards.append(reward)
            t += 1
            if t % 10000 == 0:
                print("Average reward over last " + str(window_size) + " timesteps: " + str(sum(last_rewards) / window_size))
            
            if t > 2500000: #and sum(last_rewards) / window_size > 0.2:
                print("Old state: " + str(state))
                print("Action, Reward: " + str(action) + ", " + str(reward))
                print("New state: " + str(new_state))
                print("")

if __name__ == "__main__":
    building = Environment()
    caller = Caller()
    agent = Agent()

    ctrl = Controller(building, caller, agent)
    ctrl.run()