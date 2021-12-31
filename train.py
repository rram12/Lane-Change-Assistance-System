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
import math


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option("--nogui", action="store_true",
                          default=False, help="run the commandline version of sumo")
    options, args = opt_parser.parse_args()
    return options


def train():
    episodes = 100
    steps = 10000
    agent = Agent(n_actions=3, input_dim=(9, 7))
    env = Environment(agent)
    stats = np.zeros((episodes, 4))
    episode_reward = 0
    edge_ids = ["m0", "m1", "m2", "in0", "in1", "out0", "out1"]
    for episode in range(episodes):
        traci.start([sumoBinary, "-c", "network/project.sumocfg", "--no-step-log", "true", "-W", "--duration-log.disable"],label="master")
        print("/////////////////////////////////////////")
        print("training episode "+str(episode))
        print("/////////////////////////////////////////")
        
        cur_state = env.reset()

        ## Randomization
        for edge in edge_ids:
            traci.edge.setMaxSpeed(edge, randint(40,80))
        
        traci.vehicletype.setMaxSpeed("target",randint(40,80))
        traci.vehicletype.setAccel("target",randint(1,6))
        traci.vehicletype.setDecel("target",randint(1,6))

        traci.vehicletype.setMaxSpeed("car",randint(40,80))
        traci.vehicletype.setAccel("car",randint(1,6))
        traci.vehicletype.setDecel("car",randint(1,6))

        traci.vehicle.add("v0", "r3", typeID='target', depart=str(randint(0,80)))
        for i in np.arange(randint(60,300)):
            traci.vehicle.add(str(i), "r"+str(randint(0,8)), typeID='target', depart=str(randint(0,80)))

        ## End randomization
        
        for step in range(steps):
            if env.target_agent.has_entred and env.target_agent.is_on_multilane_road:
                action = env.target_agent.choose_action(cur_state)
                new_state, reward, done, done_reason = env.step(action, step)
                env.target_agent.remember(
                    cur_state, action, reward, new_state, done)
                episode_reward += reward
                env.target_agent.replay()
                env.target_agent.target_train()
                cur_state = new_state
                if done:
                    break
            else:
                env.step(None, step)
        if done_reason == "out of road":
            stats[episode, 0] += 1
        elif done_reason == "collision":
            stats[episode, 1] += 1
        elif done_reason == "exited":
            stats[episode, 2] += 1
        stats[episode, 3] = episode_reward
        #print("++++ cumulative reward: {:.2f}".format(episode_reward))
        traci.close()
        sys.stdout.flush()
    env.target_agent.save_model("final_model")

    x_axis = list(range(0, episodes))
    save_plot(x_axis, stats[:, 0], "out of road", 'output/performance.png', 'episodes', 'performance', 'Performance of the model by episode', y2label="collision", y3label="exited", y2=stats[:, 1], y3=stats[:, 2])
    save_plot(x_axis, stats[:, 3], "cumulative reward", 'output/reward.png', 'episodes', 'reward', 'Evolution of rewards by episode')


if __name__ == "__main__":
    runGUI = False
    sumoBinary = "sumo"
    if runGUI:
        sumoBinary = "/usr/share/sumo/bin/sumo-gui"
    train()
