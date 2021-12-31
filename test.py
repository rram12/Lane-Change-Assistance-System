import os
import sys
import optparse
from environment import Environment
from agent import Agent
from utils import save_plot
from sumolib import checkBinary
import traci
from random import randint
import numpy as np



if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


def test():
    episodes = 100
    steps = 10000
    agent = Agent(n_actions=3, input_dim=(9, 7))
    env = Environment(agent)
    env.target_agent.load_model("final_model")
    env.target_agent.epsilon = 0
    env.target_agent.epsilon_min = 0

    traci.start([sumoBinary, "-c", "network/project.sumocfg", "--no-step-log", "true", "-W", "--duration-log.disable"],label="master")
    cur_state = env.reset()


    for step in range(steps):
        if env.target_agent.has_entred and env.target_agent.is_on_multilane_road:
            action = env.target_agent.choose_action(cur_state)
            new_state, reward, done, done_reason = env.step(action, step)
            env.target_agent.remember(
                cur_state, action, reward, new_state, done)
            env.target_agent.replay()
            env.target_agent.target_train()
            cur_state = new_state
            if done:
                break
        else:
            env.step(None, step)

    traci.close()
    sys.stdout.flush()


if __name__ == "__main__":
    runGUI = True
    sumoBinary = "sumo"
    if runGUI:
        sumoBinary = "/usr/share/sumo/bin/sumo-gui"
    test()


