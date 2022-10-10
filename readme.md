# Lane Change Assistance System

Lane change is a crucial vehicle maneuver which needs coordination with surrounding vehicles. Automated lane
changing functions built on rule-based models may perform well under pre-defined operating conditions, but
they may be prone to failure when unexpected situations are encountered.
In our study, we propose a Reinforcement Learning based approach to train the vehicle agent to learn an
automated lane change behavior such that it can intelligently make a lane change under diverse and even
unforeseen scenarios. In this project, we use a Double Deep Q-Network, along with rule-based constraints to make
lane-changing decision. A safe and efficient lane change behavior may be obtained by combining high-level
lateral decision-making with low-level rule based trajectory monitoring.


A Non-stationary environment is an environment where sudden concept drift can occur due to dynamic and
unknown probability data distribution function. In our case, the highway is perfectly a non-stationary envi-
ronment since many of the state variables can be changed randomly without the interference of the considered
agent.


## Technologies: 
- python 
- tensorflow 
- sumolib
- traci