import traci
import numpy as np
from scipy.spatial import distance
import time


class Environment(object):

    def __init__(self, agent):
        self.distance_threshold = 30.0
        self.no_change = 0
        self.right_change = 1
        self.left_change = 2
        self.target_vehicle_id = "v0"
        self.normal_collision_penalty = -10
        self.emergency_collision_penalty = -30
        self.off_road_penalty = -10
        self.exit_reward = 100
        self.step_reward = 0.01
        self.total_distance = 1052.1943805015458
        self.target_agent = agent

    def reset(self):
        self.target_agent.is_collision = False
        self.target_agent.is_out_of_road = False
        self.target_agent.has_entred = False
        self.target_agent.has_arrived = False
        self.target_agent.is_on_multilane_road = False
        new_state = np.zeros((9, 7))
        return new_state

    def get_vehs_on_road(self):
        veh_ids = traci.vehicle.getIDList()
        vehs = {}
        for id in veh_ids:
            vehs[id] = {
                "vehicle_id": id,
                "road": traci.vehicle.getRoadID(id),
                "lane": traci.vehicle.getLaneIndex(id),
                "speed": traci.vehicle.getSpeed(id),
                "position": traci.vehicle.getPosition(id),
                "acceleration": traci.vehicle.getAcceleration(id),
                "angle": traci.vehicle.getAngle(id)
            }
        return vehs

    def get_near_vehicles(self):
        vehs = self.get_vehs_on_road()
        near_vehs = {}
        if self.target_vehicle_id in vehs.keys():
            for veh_id in vehs.keys():
                if veh_id != self.target_vehicle_id:
                    if distance.euclidean(traci.vehicle.getPosition(veh_id), traci.vehicle.getPosition(self.target_vehicle_id)) <= self.distance_threshold:
                        near_vehs[veh_id] = vehs[veh_id]
        return near_vehs

    def DTSE(self):
        adjacent_vehs = [[None for _ in np.arange(3)]for _ in np.arange(3)]
        near_vehs = self.get_near_vehicles()
        same_lane = []
        right_lane = []
        left_lane = []
        for veh_id in near_vehs.keys():
            if near_vehs[veh_id]["lane"] == traci.vehicle.getLaneIndex(self.target_vehicle_id):
                same_lane.append(near_vehs[veh_id])
            elif near_vehs[veh_id]["lane"] == 1 + traci.vehicle.getLaneIndex(self.target_vehicle_id):
                left_lane.append(near_vehs[veh_id])
            elif near_vehs[veh_id]["lane"] == 1 - traci.vehicle.getLaneIndex(self.target_vehicle_id):
                right_lane.append(near_vehs[veh_id])
        adjacent_vehs[0][2-traci.vehicle.getLaneIndex(
            self.target_vehicle_id)] = self.order_same_lane(same_lane)[0]
        adjacent_vehs[1][2-traci.vehicle.getLaneIndex(
            self.target_vehicle_id)] = self.order_same_lane(same_lane)[1]
        adjacent_vehs[2][2-traci.vehicle.getLaneIndex(
            self.target_vehicle_id)] = self.order_same_lane(same_lane)[2]
        if len(left_lane) > 0:
            ordred_left_lane = self.order_non_same_lane(
                left_lane, self.order_same_lane(same_lane))
            adjacent_vehs[0][2-traci.vehicle.getLaneIndex(
                self.target_vehicle_id)-1] = ordred_left_lane[0]
            adjacent_vehs[1][2-traci.vehicle.getLaneIndex(
                self.target_vehicle_id)-1] = ordred_left_lane[1]
            adjacent_vehs[2][2-traci.vehicle.getLaneIndex(
                self.target_vehicle_id)-1] = ordred_left_lane[2]
        if len(right_lane) > 0:
            ordred_right_lane = self.order_non_same_lane(
                right_lane, self.order_same_lane(same_lane))
            adjacent_vehs[0][2-traci.vehicle.getLaneIndex(
                self.target_vehicle_id)+1] = ordred_right_lane[0]
            adjacent_vehs[1][2-traci.vehicle.getLaneIndex(
                self.target_vehicle_id)+1] = ordred_right_lane[1]
            adjacent_vehs[2][2-traci.vehicle.getLaneIndex(
                self.target_vehicle_id)+1] = ordred_right_lane[2]
        return adjacent_vehs

    def order_same_lane(self, same_lane):
        target_veh = self.get_vehs_on_road()[self.target_vehicle_id]
        same_lane_ordred = [None, target_veh, None]
        ind_follower = -1
        ind_leader = -1
        for veh in same_lane:
            if target_veh["position"][0] > veh["position"][0]:
                if ((ind_follower != np.NAN and same_lane[ind_follower]["position"][0] < veh["position"][0]) or ind_follower == np.NAN):
                    ind_follower = same_lane.index(veh)
            if target_veh["position"][0] < veh["position"][0]:
                if ((ind_leader != np.NAN and same_lane[ind_leader]["position"][0] > veh["position"][0]) or ind_leader == np.NAN):
                    ind_leader = same_lane.index(veh)
        if ind_leader != -1:
            same_lane_ordred[0] = same_lane[ind_leader]
        if ind_follower != -1:
            same_lane_ordred[2] = same_lane[ind_follower]
        return same_lane_ordred

    def order_non_same_lane(self, lane, same_lane_ordred):
        ordered = [None, None, None]
        possible_indexes = [0,1,2]
        for veh in lane:
            if veh != None:
                if veh["position"][0] < same_lane_ordred[1]["position"][0] + 0.667*self.distance_threshold:
                    if veh["position"][0] > same_lane_ordred[1]["position"][0] - 0.667*self.distance_threshold:
                        if 1 in possible_indexes:
                            ordered[1] = veh
                            possible_indexes.remove(1)
                    else:
                        if 2 in possible_indexes:
                            ordered[2] = veh
                            possible_indexes.remove(2)
                else:
                    if 0 in possible_indexes:
                        ordered[0] = veh
                        possible_indexes.remove(0)
        return ordered

    def order_non_same_lane1(self, lane, same_lane_ordred):
        ordered = [None, None, None]
        indexes_same = [1, 2, 3]
        for veh in lane:
            if len(indexes_same) > 0:
                biggest_distance = 0
                biggest_distance_ind = 0
                for i in indexes_same:
                    if same_lane_ordred[i] != None:
                        if biggest_distance < distance.euclidean(same_lane_ordred[i]["position"], veh["position"]):
                            biggest_distance = distance.euclidean(
                                same_lane_ordred[i]["position"], veh["position"])
                            biggest_distance_ind = i
                ordered[biggest_distance_ind] = veh
                #print(*indexes_same, sep = ", ")
                #print(" deleting "+str(biggest_distance_ind))
                indexes_same.remove(biggest_distance_ind)
        return ordered

    def convert_DTSE(self, system_state):
        input_matrix = np.zeros((9, 7))
        row = 0
        for i in range(3):
            for j in range(3):
                veh = system_state[i][j]
                if veh:
                    input_matrix[row, 0] = veh["speed"]
                    input_matrix[row, 1] = veh["acceleration"]
                    input_matrix[row, 5] = veh["angle"]
                    if self.target_vehicle_id == veh["vehicle_id"]:
                        input_matrix[row, 2] = 0
                        input_matrix[row, 3] = 1
                        input_matrix[row, 4] = 1
                        if self.target_vehicle_id in traci.vehicle.getIDList():
                            input_matrix[row, 6] = self.total_distance - traci.vehicle.getDistance(self.target_vehicle_id)
                    else:
                        input_matrix[row, 2] = distance.euclidean(
                            veh["position"], traci.vehicle.getPosition(self.target_vehicle_id))
                        input_matrix[row, 3] = 1
                        input_matrix[row, 4] = 0
                row += 1
        return input_matrix
    
    def is_collision(self):
        collison_list = traci.simulation.getCollidingVehiclesIDList()
        return self.target_vehicle_id in collison_list

    def step(self, action, step, input=(9, 7)):
        if action != None and self.target_agent.is_on_multilane_road and self.target_agent.has_entred and (not self.target_agent.has_arrived):
            lane_index = traci.vehicle.getLaneIndex(self.target_vehicle_id)
            if lane_index == 0 and action == self.right_change:
                self.target_agent.is_out_of_road = True
                print("out of road")
                return (np.ones(input) * -1, self.off_road_penalty, True, "out of road")
            elif lane_index == 2 and action == self.left_change:
                self.target_agent.is_out_of_road = True
                print("out of road")
                return (np.ones(input) * -1, self.off_road_penalty, True, "out of road")
            else:
                target_lane = lane_index
                if action == self.right_change:
                    target_lane = lane_index - 1
                elif action == self.left_change:
                    target_lane = lane_index + 1
                traci.vehicle.changeLane(
                    self.target_vehicle_id, target_lane, 0)

        if not self.target_agent.is_out_of_road:
            traci.simulationStep()

        if self.is_collision():
            print("collision")
            return (np.ones(input) * -1, self.normal_collision_penalty, True, "collision")

        else:
            vehs_id_list = traci.vehicle.getIDList()
            if self.target_vehicle_id not in vehs_id_list:
                if self.target_agent.has_entred and not self.target_agent.has_arrived:
                    self.target_agent.has_arrived = True
                    self.target_agent.stop_time = traci.simulation.getTime()
                    print("arrived! took the vehicle {:.1f} s".format(self.target_agent.stop_time - self.target_agent.start_time))
                    # return (np.ones(input), self.exit_reward, True, "exited")
                    return (np.ones(input), np.exp(self.exit_reward/(self.target_agent.stop_time - self.target_agent.start_time)), True, "exited")
            else:
                vehs = self.get_vehs_on_road()
                if not self.target_agent.has_entred:
                    print("entred")
                    self.target_agent.has_entred = True
                    self.target_agent.start_time = traci.simulation.getTime()
                    traci.vehicle.setLaneChangeMode(
                        self.target_vehicle_id, 0b000000010000)
                road_id = traci.vehicle.getRoadID(self.target_vehicle_id)

                if traci.edge.getLaneNumber(road_id) > 1:
                    self.target_agent.is_on_multilane_road = True
                    system_state = self.DTSE()
                    input_matrix = self.convert_DTSE(system_state)
                    return (input_matrix, self.step_reward*step, False, "")
                else:
                    return (np.zeros(input), 0, False, "")
