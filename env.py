import os
import random
import numpy as np
from copy import deepcopy
from skimage import io
from skimage.measure import block_reduce

from sensor import sensor_work
from utils import *
from parameter import *

class Env:
    def __init__(self, episode_index, plot=False, test=False):
        self.episode_index = episode_index
        self.plot = plot
        self.test = test

        self.ground_truth, initial_cell, self.map_path = self.import_ground_truth(episode_index)
        self.cell_size = CELL_SIZE
        self.ground_truth_size = np.shape(self.ground_truth)

        self.global_belief  = np.ones(self.ground_truth_size) * UNKNOWN
        self.agent_beliefs  = [np.ones_like(self.global_belief) * UNKNOWN for _ in range(N_AGENTS)]

        self.belief_origin_x = -np.round(initial_cell[0] * self.cell_size, 1)
        self.belief_origin_y = -np.round(initial_cell[1] * self.cell_size, 1)

        self.sensor_range = SENSOR_RANGE
        self.explored_rate = 0.0
        self.done = False
        self.total_travel_dist = 0.0
        self.agent_travel_dists = np.zeros(N_AGENTS, dtype=float)
        self.max_travel_dist = 0.0

        self.global_belief = sensor_work(initial_cell,
                                         round(self.sensor_range / self.cell_size),
                                         self.global_belief,
                                         self.ground_truth)
        for i in range(N_AGENTS):
            self.agent_beliefs[i] = deepcopy(self.global_belief)

        tmp_info = MapInfo(self.agent_beliefs[0], self.belief_origin_x, self.belief_origin_y, self.cell_size)
        free_nodes, _ = get_updating_node_coords(np.array([0.0, 0.0]), tmp_info)
        choice = np.random.choice(free_nodes.shape[0], N_AGENTS, replace=True)
        self.robot_locations = np.array(free_nodes[choice])

        for cell in get_cell_position_from_coords(self.robot_locations, tmp_info).reshape(-1, 2):
            self.global_belief = sensor_work(cell, round(self.sensor_range / self.cell_size),
                                             self.global_belief, self.ground_truth)
        for i in range(N_AGENTS):
            self.agent_beliefs[i] = deepcopy(self.global_belief)

        self.belief_info = MapInfo(self.agent_beliefs[0], self.belief_origin_x, self.belief_origin_y, self.cell_size)
        self.ground_truth_info = MapInfo(self.ground_truth, self.belief_origin_x, self.belief_origin_y, self.cell_size)

        self.global_frontiers = get_frontier_in_map(self.belief_info)
        if self.plot:
            self.frame_files = []

        H, W = self.ground_truth_size

        self.ownership_map = -np.ones((H, W), dtype=np.int16)

        self._discover_free_masks = [np.zeros((H, W), dtype=bool) for _ in range(N_AGENTS)]
        self._discover_occ_masks  = [np.zeros((H, W), dtype=bool) for _ in range(N_AGENTS)]

        self.discovered_area_free_m2 = np.zeros(N_AGENTS, dtype=float)
        self.discovered_area_occ_m2  = np.zeros(N_AGENTS, dtype=float)
        self._cell_area = float(self.cell_size) ** 2

        self._known_cells_prev_per_agent = [
            int(np.count_nonzero(self.agent_beliefs[i] != UNKNOWN)) for i in range(N_AGENTS)
        ]
        H, W = self.ground_truth_size
        self._obs_total_cells = int(H * W)

        self._obs_rate_thr = 0.995
        self._milestone_hit_prev = [
            (self._known_cells_prev_per_agent[i] / max(1, self._obs_total_cells)) >= self._obs_rate_thr
            for i in range(N_AGENTS)
        ]
        self.last_personal_obs_gain = [0.0 for _ in range(N_AGENTS)]


    def import_ground_truth(self, episode_index):
        map_dir = f'maps_small' if not self.test else f'dataset/maps_eval'
        map_list = []
        for root, _, files in os.walk(map_dir):
            for f in files:
                map_list.append(os.path.join(root, f))
        if not self.test:
            rng = random.Random(1)
            rng.shuffle(map_list)

        idx = episode_index % len(map_list)
        gt = (io.imread(map_list[idx], 1)).astype(int)

        robot_cell = np.nonzero(gt == 208)
        robot_cell = np.array([np.array(robot_cell)[1, 10], np.array(robot_cell)[0, 10]])

        gt = (gt > 150) | ((gt <= 80) & (gt >= 50))
        gt = gt * 254 + 1
        return gt, robot_cell, map_list[idx]

    def _compute_comm_groups(self):
        adj = np.zeros((N_AGENTS, N_AGENTS), dtype=bool)
        for i in range(N_AGENTS):
            for j in range(i+1, N_AGENTS):
                if np.linalg.norm(self.robot_locations[i] - self.robot_locations[j]) <= COMMS_RANGE:
                    adj[i, j] = adj[j, i] = True
        groups, unseen = [], set(range(N_AGENTS))
        while unseen:
            r = unseen.pop()
            stack = [r]; comp = {r}
            while stack:
                u = stack.pop()
                for v in range(N_AGENTS):
                    if v in unseen and adj[u, v]:
                        unseen.remove(v)
                        comp.add(v)
                        stack.append(v)
            groups.append(comp)
        return groups

    def _merge_agent_beliefs(self, groups):
        for g in groups:
            merged = np.ones_like(self.global_belief) * UNKNOWN
            for i in g:
                known = (self.agent_beliefs[i] != UNKNOWN)
                merged[known] = self.agent_beliefs[i][known]
            for i in g:
                self.agent_beliefs[i] = merged.copy()

    def step(self, next_waypoint, agent_id):
        old = self.robot_locations[agent_id]
        dist = np.linalg.norm(next_waypoint - old)
        self.total_travel_dist += dist
        self.agent_travel_dists[agent_id] += dist

        self.robot_locations[agent_id] = next_waypoint

        cell = get_cell_position_from_coords(next_waypoint, self.belief_info)

        old_global = self.global_belief.copy()
        old_own    = self.agent_beliefs[agent_id].copy()

        self.global_belief = sensor_work(
            cell, round(self.sensor_range / self.cell_size),
            self.global_belief, self.ground_truth
        )
        self.agent_beliefs[agent_id] = sensor_work(
            cell, round(self.sensor_range / self.cell_size),
            self.agent_beliefs[agent_id], self.ground_truth
        )

        newly_free_global = (old_global == UNKNOWN) & (self.global_belief == FREE)
        newly_occ_global  = (old_global == UNKNOWN) & (self.global_belief == OCCUPIED)

        self._discover_free_masks[agent_id][:] = False
        self._discover_occ_masks[agent_id][:]  = False
        if newly_free_global.any():
            self._discover_free_masks[agent_id][newly_free_global] = True
            self.discovered_area_free_m2[agent_id] += float(newly_free_global.sum()) * self._cell_area
        if newly_occ_global.any():
            self._discover_occ_masks[agent_id][newly_occ_global] = True
            self.discovered_area_occ_m2[agent_id]  += float(newly_occ_global.sum()) * self._cell_area

        if newly_free_global.any():
            idx = newly_free_global & (self.ownership_map < 0)
            self.ownership_map[idx] = agent_id
        if newly_occ_global.any():
            idx = newly_occ_global & (self.ownership_map < 0)
            self.ownership_map[idx] = agent_id

        groups = self._compute_comm_groups()
        self._merge_agent_beliefs(groups)

        self.belief_info = MapInfo(
            self.agent_beliefs[agent_id],
            self.belief_origin_x, self.belief_origin_y, self.cell_size
        )


    def evaluate_exploration_rate(self):
        self.explored_rate = np.sum(self.global_belief == FREE) / np.sum(self.ground_truth == FREE)

    def calculate_reward(self):
        denom = float(max(1, (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)))
        self.evaluate_exploration_rate()
        binfo = MapInfo(self.global_belief, self.belief_origin_x, self.belief_origin_y, self.cell_size)
        global_frontiers = get_frontier_in_map(binfo)
        if len(global_frontiers) == 0:
            delta_frontier = len(self.global_frontiers)
        else:
            observed = self.global_frontiers - global_frontiers
            delta_frontier = len(observed)
        self.global_frontiers = global_frontiers
        R_frontier = float(delta_frontier) / denom
        known_now = [int(np.count_nonzero(self.agent_beliefs[i] != UNKNOWN)) for i in range(N_AGENTS)]
        deltas = [max(0, known_now[i] - self._known_cells_prev_per_agent[i]) for i in range(N_AGENTS)]
        self._known_cells_prev_per_agent = known_now
        per_agent_rewards = [float(d) / denom for d in deltas]
        R_obs_mean = (float(np.mean(deltas)) if len(deltas) > 0 else 0.0) / denom
        rates_now = [known_now[i] / max(1, self._obs_total_cells) for i in range(N_AGENTS)]
        newly_hit = 0
        for i in range(N_AGENTS):
            hit_now = (rates_now[i] >= self._obs_rate_thr)
            if (not self._milestone_hit_prev[i]) and hit_now:
                newly_hit += 1
            self._milestone_hit_prev[i] = hit_now
        R_milestone = 0.5 * float(newly_hit)
        team_reward = 0.4 * R_frontier + 0.2 * R_obs_mean + R_milestone
        self.last_personal_obs_gain = per_agent_rewards
        return team_reward, per_agent_rewards


    def get_agent_map(self, agent_id):
        return MapInfo(self.agent_beliefs[agent_id],
                       self.belief_origin_x, self.belief_origin_y, self.cell_size)

    def get_total_travel(self):
        return self.total_travel_dist

    def get_agent_travel(self):
        return self.agent_travel_dists.copy()

    def get_max_travel(self):
        return float(self.max_travel_dist)

    def pop_discovery_masks(self):
        free = [m.copy() for m in self._discover_free_masks]
        occ  = [m.copy() for m in self._discover_occ_masks]
        for m in self._discover_free_masks: m[:] = False
        for m in self._discover_occ_masks:  m[:] = False
        return free, occ

    def get_ownership_map(self):
        return self.ownership_map.copy()

    def get_discovered_area(self):
        return (self.discovered_area_free_m2.copy(),
                self.discovered_area_occ_m2.copy())

    def get_map_balance_stats(self, which="free"):
        if which == "free":
            arr = self.discovered_area_free_m2
        elif which == "occ":
            arr = self.discovered_area_occ_m2
        else:
            arr = self.discovered_area_free_m2 + self.discovered_area_occ_m2

        per_agent = arr.copy()
        mean = float(per_agent.mean())
        std  = float(per_agent.std(ddof=0))
        cv   = float(std / (mean + 1e-9))
        return dict(mean=mean, std=std, cv=cv, per_agent=per_agent)

    def get_last_personal_obs_gain(self):
        if hasattr(self, 'last_personal_obs_gain'):
            return list(self.last_personal_obs_gain)
        else:
            return [0.0 for _ in range(N_AGENTS)]
