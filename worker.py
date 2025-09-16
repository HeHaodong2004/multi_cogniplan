# worker.py
import os
import time
import math
from copy import deepcopy
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from env import Env
from agent import Agent
from utils import *  
from node_manager import NodeManager
from ground_truth_node_manager import GroundTruthNodeManager
from parameter import *

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

def make_gif_safe(frame_paths, out_path, duration_ms=120):
    frame_paths = [p for p in frame_paths if os.path.exists(p)]
    frame_paths.sort()
    if not frame_paths:
        return
    frames = []
    base_size = None
    for p in frame_paths:
        try:
            im = Image.open(p).convert("RGB")
            if base_size is None:
                base_size = im.size
            elif im.size != base_size:
                im = im.resize(base_size, Image.BILINEAR)
            frames.append(im)
        except Exception:
            continue
    if not frames:
        return
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, optimize=False)


class Contract:
    """
    Rendezvous 合同（“协商-待激活(armed)” -> “执行(active)” -> done/failed）
    - P_list: [P_primary, P_backup] 至多两个点，备选可空
    - r: 区域半径（米）
    - t_min_list / t_max_list: 每个点对应的时间窗（协商时估计；执行时会重算调度）
    - participants: 参与 agent id
    - status: 'armed' | 'active' | 'done' | 'failed'
    - target_idx: 当前执行的目标索引（0=主点，1=备选）
    - meta: 执行阶段的调度信息：{'T_tar','t_dep'[N_AGENTS],'q_i'[N_AGENTS]} 针对当前 target_idx
    """
    def __init__(self, P_list, r, t_min_list, t_max_list, participants, created_t):
        self.P_list = [np.array(P, dtype=float) for P in P_list]
        self.r = float(r)
        self.t_min_list = [int(x) for x in t_min_list]
        self.t_max_list = [int(x) for x in t_max_list]
        self.participants = set(participants)
        self.created_t = int(created_t)
        self.status = 'armed'
        self.target_idx = 0
        self.meta = {}

    @property
    def P(self):
        return self.P_list[self.target_idx]

    @property
    def t_min(self):
        return self.t_min_list[self.target_idx]

    @property
    def t_max(self):
        return self.t_max_list[self.target_idx]

    def within_region(self, pos_xy):
        return np.linalg.norm(np.asarray(pos_xy, dtype=float) - self.P) <= self.r

    def has_backup(self):
        return len(self.P_list) >= 2

    def switch_to_backup(self):
        if self.has_backup() and self.target_idx == 0:
            self.target_idx = 1
            self.meta = {} 
            return True
        return False


class Worker:
    def __init__(self, meta_agent_id, policy_net, predictor, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.node_manager = NodeManager(plot=self.save_image)
        self.robots = [Agent(i, policy_net, predictor, self.node_manager, device=device, plot=save_image)
                       for i in range(N_AGENTS)]
        self.gtnm = GroundTruthNodeManager(self.node_manager, self.env.ground_truth_info, device=device, plot=save_image)

        self.episode_buffer = [[] for _ in range(27)]
        self.perf_metrics = dict()

        self.last_known_locations = [self.env.robot_locations.copy() for _ in range(N_AGENTS)]
        self.last_known_intents = [{aid: [] for aid in range(N_AGENTS)} for _ in range(N_AGENTS)]

        self.run_dir = os.path.join(
            gifs_path, f"run_g{self.global_step}_w{self.meta_agent_id}_{os.getpid()}_{int(time.time()*1000)}"
        )
        if self.save_image:
            os.makedirs(self.run_dir, exist_ok=True)
        self.env.frame_files = []

        # Rendezvous
        self.contract: Contract = None
        self.candidate_buffer = []  
        self.cand_last_update_t = -1

        self.was_fully_connected = False

        gt_map = self.env.ground_truth_info.map
        self._gt_free_total = int(np.count_nonzero(gt_map == FREE))

        self._t = 0

    def _match_intent_channels(self, obs_pack):
        n, m, e, ci, ce, ep = obs_pack
        need, got = NODE_INPUT_DIM, n.shape[-1]
        if got < need:
            pad = torch.zeros((n.shape[0], n.shape[1], need - got), dtype=n.dtype, device=n.device)
            n = torch.cat([n, pad], dim=-1)
        return [n, m, e, ci, ce, ep]

    def run_episode(self):
        """
        主循环要点：
        - 全联通 && (无激活合同) 时：刷新候选池、生成 negotiated_plan（只记录最佳点与时间窗，不计算出发时刻）
        - “全联通→非全联通”的第一拍：若有 negotiated_plan，则用机会约束调度计算 t_dep/T_tar，激活 active 合同
        - 激活后：t < t_dep 自由探索；t >= t_dep 用 Dijkstra（Node 图） 强制导航至合同圈；圈内巡航不离圈
        - 若中途“全体重连”：取消旧约重新协商（不继续之前的redenzvous任务）
        """
        done = False

        # 初始化：图、预测、意图
        for i, r in enumerate(self.robots):
            r.update_graph(self.env.get_agent_map(i), self.env.robot_locations[i])
        for r in self.robots:
            r.update_predict_map()
        for i, r in enumerate(self.robots):
            r.update_planning_state(self.env.robot_locations)
            self.last_known_intents[r.id][r.id] = deepcopy(r.intent_seq)

        # 初始连通性
        groups0 = self._compute_groups_from_positions(self.env.robot_locations)
        self.was_fully_connected = (len(groups0) == 1 and len(groups0[0]) == N_AGENTS)

        # 本回合的协商方案
        self.negotiated_plan = None  # {'P','t_min','t_max','score','ig_total','risk'}

        if self.save_image:
            self.plot_env(step=0)

        # ================= 主循环 =================
        for t in range(MAX_EPISODE_STEP):
            self._t = t  # 供冲突消解使用

            # --- 全局地图与掩膜 ---
            global_map_info = MapInfo(self.env.global_belief,
                                      self.env.belief_origin_x,
                                      self.env.belief_origin_y,
                                      self.env.cell_size)
            belief_map = global_map_info.map
            p_free = (self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
                      if self.robots[0].pred_mean_map_info is not None
                      else (belief_map == FREE).astype(np.float32))
            trav_mask = (((p_free) >= RDV_TAU_FREE) | (belief_map == FREE)) & (belief_map != OCCUPIED)
            unknown_mask = (belief_map == UNKNOWN)

            # --- 当前连通性 ---
            groups = self._compute_groups_from_positions(self.env.robot_locations)
            is_fully_connected = (len(groups) == 1 and len(groups[0]) == N_AGENTS)

            # ========== 联通阶段：协商（只选点，不出发） ==========
            if is_fully_connected:
                # 若有激活合同：重连→取消旧约并重新协商
                if self.contract is not None and self.contract.status == 'active':
                    if RDV_VERBOSE:
                        print(f"[RDV] All-connected @t={t}: cancel active contract & renegotiate.")
                    self.contract = None

                # 全联通 & 无激活合同时刷新候选池
                if (t % RDV_CAND_UPDATE_EVERY == 0) and (self.contract is None):
                    try:
                        self._update_candidate_buffer(global_map_info, trav_mask, unknown_mask, p_free, t)
                        best = self._select_best_candidate_from_buffer()
                        self.negotiated_plan = best  
                    except Exception as e:
                        print(f"[RDV] candidate update failed @t={t}: {e}")

            # ========== 断开瞬间：激活合同 ==========
            if (not is_fully_connected) and self.was_fully_connected and (self.contract is None):
                cand = self.negotiated_plan or self._select_best_candidate_from_buffer()
                if cand is not None:
                    P_c = cand['P']
                    r_meet = RDV_REGION_FRAC * COMMS_RANGE
                    sched = self._chance_constrained_schedule(P_c, r_meet, t, global_map_info, belief_map, p_free)
                    if sched is not None:
                        T_tar, dep_times, q_i, eps_used = sched
                        self.contract = Contract(
                            P_list=[P_c],
                            r=r_meet,
                            t_min_list=[cand['t_min']],
                            t_max_list=[cand['t_max']],
                            participants=set(range(N_AGENTS)),
                            created_t=t
                        )
                        self.contract.meta = {
                            'score': cand['score'],
                            'ig': cand['ig_total'],
                            'risk': cand['risk'],
                            'T_tar': T_tar,
                            't_dep': dep_times,
                            'q_i': q_i,
                            'eps': eps_used
                        }
                        self.contract.status = 'active'
                        if RDV_VERBOSE:
                            print(f"[RDV] Activate negotiated @t={t} center={P_c}, "
                                  f"window=({cand['t_min']},{cand['t_max']}), T_tar={T_tar}, t_dep={dep_times}")

            # 记录本拍连通性
            self.was_fully_connected = is_fully_connected

            # ---------- 原策略：自由探索一步（用于未出发/无合同） ----------
            picks_raw, dists = [], []
            for i, r in enumerate(self.robots):
                obs = r.get_observation(self.last_known_locations[i], self.last_known_intents[i])
                c_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, self.env.robot_locations
                )
                r.save_observation(obs, self._match_intent_channels(c_obs))
                nxt, _, act = r.select_next_waypoint(obs)
                r.save_action(act)
                picks_raw.append(nxt)
                dists.append(np.linalg.norm(nxt - r.location))

            # ---------- 执行：出发前自由，出发后 Dijkstra 强制 ----------
            picks = []
            for i, r in enumerate(self.robots):
                # 没合同或合同非激活：自由探索
                if self.contract is None or self.contract.status != 'active':
                    picks.append(picks_raw[i])
                    continue

                # 已在圈内：圈内巡航不离圈
                if self.contract.within_region(r.location):
                    picks.append(self._in_zone_patrol_step(i, r, global_map_info))
                    continue

                # 出发前：自由探索
                t_dep_i = int(self.contract.meta['t_dep'][i])
                if t < t_dep_i:
                    picks.append(picks_raw[i])
                    continue

                # 出发后：基于 Node 图的 Dijkstra 强制导航
                nxt_xy = self._forced_dijkstra_next_xy(
                    aid=i, agent=r, picks_raw_i=picks_raw[i], global_map_info=global_map_info
                )
                picks.append(nxt_xy)

            # ---------- 冲突消解 & 推进环境 ----------
            picks = self.resolve_conflicts(picks, dists)
            prev_max = self.env.get_agent_travel().max()
            prev_total = self.env.get_total_travel()
            for r, loc in zip(self.robots, picks):
                self.env.step(loc, r.id)
            self.env.max_travel_dist = self.env.get_agent_travel().max()
            delta_max = self.env.max_travel_dist - prev_max
            delta_total = self.env.get_total_travel() - prev_total

            # ---------- 通信同步（同组可见） ----------
            groups_after_move = self._compute_groups_from_positions(self.env.robot_locations)
            for g in groups_after_move:
                for i in g:
                    for j in g:
                        self.last_known_locations[i][j] = self.env.robot_locations[j].copy()
                        self.last_known_intents[i][j] = deepcopy(self.robots[j].intent_seq)

            # ---------- 图/预测/意图更新 ----------
            for i, r in enumerate(self.robots):
                r.update_graph(self.env.get_agent_map(i), self.env.robot_locations[i])
                r.update_predict_map()
                r.update_planning_state(self.last_known_locations[i], self.last_known_intents[i])
                self.last_known_intents[i][r.id] = deepcopy(r.intent_seq)

            # ---------- 合同进度与回退 ----------
            if self.contract is not None and self.contract.status == 'active':
                try:
                    self._contract_progress_and_fallback(t, global_map_info, trav_mask, unknown_mask, p_free)
                except Exception as e:
                    print(f"[RDV] progress/fallback error @t={t}: {e}")

            # ---------- 奖励与终止 ----------
            team_reward_env, per_agent_obs_rewards = self.env.calculate_reward()
            team_reward = (
                team_reward_env
                - ((delta_max / UPDATING_MAP_SIZE) * MAX_TRAVEL_COEF)
                - ((delta_total / UPDATING_MAP_SIZE) * TOTAL_TRAVEL_COEF)
            )

            utilities_empty = all((r.utility <= 3).all() for r in self.robots)
            if self._gt_free_total > 0:
                per_agent_free_counts = [int(np.count_nonzero(r.map_info.map == FREE)) for r in self.robots]
                per_agent_cov = [c / self._gt_free_total for c in per_agent_free_counts]
                coverage_ok = all(c >= 0.995 for c in per_agent_cov)
            else:
                coverage_ok = False

            done = utilities_empty or coverage_ok
            if done:
                team_reward += 10.0

            for i, r in enumerate(self.robots):
                indiv_total = team_reward + per_agent_obs_rewards[i]
                r.save_reward(indiv_total)
                r.save_done(done)
                next_obs = r.get_observation(self.last_known_locations[i], self.last_known_intents[i])
                c_next_obs, _ = self.gtnm.get_ground_truth_observation(
                    r.location, r.pred_mean_map_info, self.env.robot_locations
                )
                r.save_next_observations(next_obs, self._match_intent_channels(c_next_obs))

            if self.save_image:
                self.plot_env(step=t + 1)

            if done:
                break

        # ---------- 总结 ----------
        self.perf_metrics.update({
            'travel_dist': self.env.get_total_travel(),
            'max_travel': self.env.get_max_travel(),
            'explored_rate': self.env.explored_rate,
            'success_rate': done
        })

        # === 新增：按“首次被某个 agent 发现”的定义计算面积方差（单位 m^2） ===   （用于计算test.py中的area std）
        free_m2, occ_m2 = self.env.get_discovered_area()   # 每个 agent 的首次 FREE / OCC 面积（m^2）
        area_per_agent_m2 = free_m2 + occ_m2               # “首次发现”的总面积 = FREE + OCC
        area_std_m2 = float(np.std(area_per_agent_m2, ddof=0))
        self.perf_metrics['area_std_m2'] = area_std_m2
        
        if self.save_image:
            make_gif_safe(self.env.frame_files, os.path.join(self.run_dir, f"ep_{self.global_step}.gif"))

        for r in self.robots:
            for k in range(len(self.episode_buffer)):
                self.episode_buffer[k] += r.episode_buffer[k]

    def resolve_conflicts(self, picks, dists):
        """
        改进版冲突消解：
        - 优先让“正在执行 RDV 的机器人”（合同 active、已到出发时刻、且还不在圈内) 先挑邻居
        - 其次才是自由探索者
        """
        picks = np.array(picks).reshape(-1, 2)

        forced_idx, free_idx = [], []
        for i in range(len(self.robots)):
            if self.contract is not None and self.contract.status == 'active':
                in_zone = self.contract.within_region(self.robots[i].location)
                dep_list = self.contract.meta.get('t_dep', [0] * N_AGENTS)
                t_dep_i = int(dep_list[i]) if i < len(dep_list) else 0
                in_forced = (not in_zone) and (self._t >= t_dep_i)
            else:
                in_forced = False
            (forced_idx if in_forced else free_idx).append(i)

        order = forced_idx + free_idx

        chosen_complex, resolved = set(), [None] * len(self.robots)
        for rid in order:
            robot = self.robots[rid]
            # 用 round(.,1) 的 key 查找，避免浮点误差 miss
            try:
                loc_key = np.around(robot.location, 1).tolist()
                neighbor_coords = sorted(
                    list(self.node_manager.nodes_dict.find(loc_key).data.neighbor_set),
                    key=lambda c: np.linalg.norm(np.array(c) - picks[rid])
                )
            except Exception:
                neighbor_coords = [robot.location.copy()]

            picked = None
            for cand in neighbor_coords:
                key = complex(cand[0], cand[1])
                if key not in chosen_complex:
                    picked = np.array(cand)
                    break
            resolved[rid] = picked if picked is not None else robot.location.copy()
            chosen_complex.add(complex(resolved[rid][0], resolved[rid][1]))

        return np.array(resolved).reshape(-1, 2)

    # -------------------------------------------------------------------------
    # 全联通时的“只协商不出发”（not used；当前主流程用 negotiated_plan 变量，不直接用此函数）
    # -------------------------------------------------------------------------
    def _negotiate_contract_now(self, t_now, map_info, trav_mask, unknown_mask, p_free):
        if not self.candidate_buffer:
            self._update_candidate_buffer(map_info, trav_mask, unknown_mask, p_free, t_now)
            if not self.candidate_buffer:
                return False

        best = self._select_best_candidate_from_buffer()
        if best is None:
            return False

        P_c = best['P']
        r_meet = RDV_REGION_FRAC * COMMS_RANGE

        # 采样法估计 q_i（只存 q_i，出发时刻等断联再算）
        sched = self._chance_constrained_schedule(
            P_c, r_meet, t_now, map_info, map_info.map,
            (self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
             if self.robots[0].pred_mean_map_info is not None
             else (map_info.map == FREE).astype(np.float32))
        )
        if sched is None:
            return False
        _T_tar_tmp, _dep_tmp, q_i, eps_used = sched

        self.contract = Contract(
            P_list=[P_c],
            r=r_meet,
            t_min_list=[best['t_min']],
            t_max_list=[best['t_max']],
            participants=set(range(N_AGENTS)),
            created_t=t_now
        )
        self.contract.status = 'armed'
        self.contract.meta = {
            'score': best['score'],
            'ig': best['ig_total'],
            'risk': best['risk'],
            'q_i': q_i,
            'eps': eps_used
        }
        if RDV_VERBOSE:
            print(f"[RDV] Negotiated@t={t_now} center={P_c}, q_i={np.round(q_i, 1)}")
        return True

    def _activate_contract_on_split(self, t_now):
        if self.contract is None or self.contract.status != 'negotiated':
            return False
        q_i = self.contract.meta.get('q_i', None)
        if not q_i:
            return False

        q_max = max(q_i)
        T_tar = int(round(t_now + q_max))
        t_min = int(round(T_tar - (RDV_WINDOW_ALPHA_EARLY * q_max + RDV_WINDOW_BETA_EARLY)))
        t_max = int(round(T_tar + (RDV_WINDOW_ALPHA_LATE * q_max + RDV_WINDOW_BETA_LATE)))
        dep_times = [int(max(t_now, math.floor(T_tar - qi))) for qi in q_i]

        self.contract.t_min_list[self.contract.target_idx] = t_min
        self.contract.t_max_list[self.contract.target_idx] = t_max
        self.contract.meta['T_tar'] = T_tar
        self.contract.meta['t_dep'] = dep_times
        self.contract.status = 'active'

        if RDV_VERBOSE:
            print(f"[RDV] Activated@t={t_now} T_tar={T_tar}, t_dep={dep_times}")
        return True

    # -------------------------------------------------------------------------
    # 候选池（选点） —— （trav_mask + BFS 估计）
    # -------------------------------------------------------------------------
    def _update_candidate_buffer(self, map_info: MapInfo, trav_mask, unknown_mask, p_free, t_now: int):
        H, W = trav_mask.shape
        cell_size = float(map_info.cell_size)

        r_idx, c_idx = np.where(unknown_mask & trav_mask)
        if r_idx.size == 0:
            self.candidate_buffer = []
            self.cand_last_update_t = t_now
            return

        # 下采样 & 稀疏
        n_cand = min(r_idx.size, RDV_CAND_K)
        sel = np.random.choice(r_idx.size, n_cand, replace=False)
        rc = np.stack([r_idx[sel], c_idx[sel]], axis=1)
        if RDV_CAND_STRIDE > 1:
            keep, seen = [], set()
            for r, c in rc:
                key = (int(r // RDV_CAND_STRIDE), int(c // RDV_CAND_STRIDE))
                if key in seen:
                    continue
                seen.add(key)
                keep.append((r, c))
            rc = np.array(keep, dtype=int)

        # BFS 距离图（基于可行栅格 trav_mask）
        dist_maps = []
        for agent in self.robots:
            start_rc = self._world_to_cell_rc(agent.location, map_info)
            if not trav_mask[start_rc[0], start_rc[1]]:
                start_rc = self._find_nearest_valid_cell(trav_mask, np.array(start_rc))
            dist_maps.append(self._bfs_dist_map(trav_mask, tuple(start_rc)))

        candidates = []
        R_info_pix = int(RDV_INFO_RADIUS_M / cell_size)
        for (r_c, c_c) in rc:
            r0 = max(0, r_c - R_info_pix); r1 = min(H, r_c + R_info_pix + 1)
            c0 = max(0, c_c - R_info_pix); c1 = min(W, c_c + R_info_pix + 1)
            local_ig = int(unknown_mask[r0:r1, c0:c1].sum())

            etas, risks, path_ig = [], [], []
            feasible = True
            for j, agent in enumerate(self.robots):
                d_steps = dist_maps[j][r_c, c_c]
                if not np.isfinite(d_steps):
                    feasible = False
                    break
                eta_j = d_steps / max(NODE_RESOLUTION, 1e-6)
                etas.append(float(eta_j))
                r_s, c_s = self._world_to_cell_rc(agent.location, map_info)
                line = self._bresenham_line_rc(r_s, c_s, r_c, c_c)
                line_risk, line_ig = 0.0, 0
                for (rr, cc) in line:
                    if 0 <= rr < H and 0 <= cc < W:
                        line_risk += float(1.0 - p_free[rr, cc])
                        if unknown_mask[rr, cc]:
                            line_ig += 1
                risks.append(line_risk / max(len(line), 1))
                path_ig.append(line_ig)
            if not feasible:
                continue

            ig_total = float(RDV_ALPHA * (sum(path_ig) + local_ig))
            disp = float(max(etas) - min(etas))
            risk_total = float(sum(risks))
            score = ig_total - RDV_BETA * disp - RDV_GAMMA * risk_total

            eta_max = max(etas)
            t_mid = t_now + int(round(eta_max))
            t_min = t_mid - int(round(RDV_WINDOW_ALPHA_EARLY * eta_max + RDV_WINDOW_BETA_EARLY))
            t_max = t_mid + int(round(RDV_WINDOW_ALPHA_LATE * eta_max + RDV_WINDOW_BETA_LATE))

            P_world = np.array([map_info.map_origin_x + c_c * cell_size,
                                map_info.map_origin_y + r_c * cell_size], dtype=float)
            candidates.append({
                'P': P_world, 'score': score, 'etas': etas, 'risk': risk_total, 'ig_total': ig_total,
                't_min': t_min, 't_max': t_max
            })

        candidates.sort(key=lambda d: d['score'], reverse=True)
        self.candidate_buffer = candidates[:RDV_TOP_M]
        self.cand_last_update_t = t_now

    def _select_best_candidate_from_buffer(self, idx=0, min_dist=0.0, ref=None):
        if not self.candidate_buffer:
            return None
        if idx == 0:
            return self.candidate_buffer[0]
        taken = 0
        for c in self.candidate_buffer:
            if ref is not None:
                if np.linalg.norm(np.asarray(c['P']) - np.asarray(ref['P'])) < max(1e-6, min_dist):
                    continue
            if taken == idx - 1:
                return c
            taken += 1
        return None

    # -------------------------------------------------------------------------
    # 激活合约时重算调度
    # -------------------------------------------------------------------------
    def _activate_contract_with_schedule(self, contract: Contract, target_idx, t_now, map_info, belief_map, p_free):
        P = contract.P_list[target_idx]
        r = contract.r
        sched = self._chance_constrained_schedule(P, r, t_now, map_info, belief_map, p_free)
        if sched is None:
            return False
        T_tar, dep_times, q_i, eps_used = sched
        contract.status = 'active'
        contract.target_idx = target_idx
        contract.meta = {
            'T_tar': int(T_tar),
            't_dep': [int(x) for x in dep_times],
            'q_i': [float(x) for x in q_i],
            'eps': float(eps_used)
        }
        return True

    # -------------------------------------------------------------------------
    # CC-ODS（机会约束调度）
    # -------------------------------------------------------------------------
    def _chance_constrained_schedule(self, P, r, t_now, map_info, belief_map, p_free):
        H, W = belief_map.shape
        cell_size = float(map_info.cell_size)

        def nearest_goal_rc(trav_mask_s):
            r_pix = int(max(1, round(r / cell_size)))
            c_rc = self._world_to_cell_rc(P, map_info)
            best, bestd = None, float('inf')
            r0 = max(0, c_rc[0] - r_pix); r1 = min(H, c_rc[0] + r_pix + 1)
            c0 = max(0, c_rc[1] - r_pix); c1 = min(W, c_rc[1] + r_pix + 1)
            for rr in range(r0, r1):
                for cc in range(c0, c1):
                    if trav_mask_s[rr, cc]:
                        d = (rr - c_rc[0]) ** 2 + (cc - c_rc[1]) ** 2
                        if d < bestd:
                            bestd, best = d, (rr, cc)
            return best

        # 采样可通行图（保留已知 FREE，排除 OCCUPIED，其他按 p_free 独立伯努利）
        samples = []
        for s in range(RDV_TT_N_SAMPLES):
            rand = np.random.rand(H, W).astype(np.float32)
            trav_s = (((rand < p_free) | (belief_map == FREE)) & (belief_map != OCCUPIED))
            samples.append(trav_s)

        # 每样本、每 agent 计算到“区域内最近可达点”的步数 -> 时间
        T_samples = {aid: [] for aid in range(N_AGENTS)}
        for s in range(RDV_TT_N_SAMPLES):
            trav_s = samples[s]
            goal_rc = nearest_goal_rc(trav_s)
            if goal_rc is None:
                for aid in range(N_AGENTS):
                    T_samples[aid].append(float('inf'))
                continue
            dist_maps = []
            for aid in range(N_AGENTS):
                start_rc = self._world_to_cell_rc(self.robots[aid].location, map_info)
                if not trav_s[start_rc[0], start_rc[1]]:
                    start_rc = self._find_nearest_valid_cell(trav_s, np.array(start_rc))
                dist_maps.append(self._bfs_dist_map(trav_s, tuple(start_rc)))
            for aid in range(N_AGENTS):
                d = dist_maps[aid][goal_rc[0], goal_rc[1]]
                T_samples[aid].append(float(d / max(NODE_RESOLUTION, 1e-6)) if np.isfinite(d) else float('inf'))

        # 取 (1-eps) 分位数
        q_i = []
        for aid in range(N_AGENTS):
            arr = np.array(T_samples[aid], dtype=np.float32)
            feasible = np.isfinite(arr)
            if feasible.sum() < max(1, int((1.0 - RDV_EPSILON) * RDV_TT_N_SAMPLES)):
                return None
            vals = np.sort(arr[feasible])
            k = max(0, int(math.ceil((1.0 - RDV_EPSILON) * len(vals))) - 1)
            q_i.append(float(vals[k]))

        # 共同到达时刻与出发时刻
        T_tar = max([t_now + qi for qi in q_i])
        dep_times = [int(max(t_now, math.floor(T_tar - qi))) for qi in q_i]
        return int(T_tar), dep_times, q_i, RDV_EPSILON

    # -------------------------------------------------------------------------
    # 基于 Node 图的 Dijkstra 强制导航
    # -------------------------------------------------------------------------
    def _graph_key(self, xy):
        """把浮点坐标对齐到节点 key（1 位小数），与 NodeManager 保持一致。"""
        return tuple(np.around(np.asarray(xy, dtype=float), 1).tolist())

    def _nearest_graph_key(self, xy):
        """把任意 xy snap 到图里最近的节点 key（若 find 失败就暴力最近）。"""
        key = self._graph_key(xy)
        try:
            _ = self.node_manager.nodes_dict.find(list(key))
            return key
        except Exception:
            best_key, best_d = None, float('inf')
            for nd in self.node_manager.nodes_dict.__iter__():
                c = np.asarray(nd.data.coords, dtype=float)
                d = np.linalg.norm(c - xy)
                if d < best_d:
                    best_d = d
                    best_key = self._graph_key(c)
            return best_key

    def _region_goal_keys(self, P, r):
        """收集合同圆内的所有图节点 key。"""
        keys = []
        for nd in self.node_manager.nodes_dict.__iter__():
            c = np.asarray(nd.data.coords, dtype=float)
            if np.linalg.norm(c - P) <= r:
                keys.append(self._graph_key(c))
        return keys

    def _forced_dijkstra_next_xy(self, aid, agent, picks_raw_i, global_map_info):
        """
        用 utils.Dijkstra 在 Node 图上做强制导航：
        - 若已在圈内：走圈内巡航；
        - 否则：start=当前位置最近节点，end=圈内所有节点里距离最短者；
        - 无可行路径就退回探索步 picks_raw_i（避免站桩）。
        """
        # 已在圈内：圈内巡航
        if self.contract is not None and self.contract.within_region(agent.location):
            return self._in_zone_patrol_step(aid, agent, global_map_info)

        # 起点 / 终点集合（图节点）
        start_key = self._nearest_graph_key(agent.location)
        if start_key is None:
            return picks_raw_i

        goal_keys = self._region_goal_keys(self.contract.P, self.contract.r)
        if not goal_keys:
            return picks_raw_i  # 按“之前版本”的逻辑：圈内没节点则先不强制

        # 在图上跑一次 Dijkstra
        dist_dict, prev_dict = Dijkstra(self.node_manager.nodes_dict, start_key)

        feas_goals = [g for g in goal_keys if dist_dict.get(g, 1e8) < 1e8 - 1]
        if not feas_goals:
            return picks_raw_i

        end_key = min(feas_goals, key=lambda g: dist_dict[g])

        # 复原路径，取下一步
        path, dist = get_Dijkstra_path_and_dist(dist_dict, prev_dict, end_key)
        if dist >= 1e7 or not path:
            return picks_raw_i

        nxt_xy = np.array(path[0], dtype=float)  # utils 返回的是去掉 start 后的节点序列
        return nxt_xy

    # -------------------------------------------------------------------------
    # 圈内巡航（不离开区域）
    # -------------------------------------------------------------------------
    def _in_zone_patrol_step(self, aid, agent, map_info):
        try:
            loc_key = np.around(agent.location, 1).tolist()
            node = self.node_manager.nodes_dict.find(loc_key).data
            neighbor_coords = list(node.neighbor_set)
        except Exception:
            neighbor_coords = [agent.location.copy()]

        def inside(xy):
            return self.contract.within_region(xy)

        best, best_score = agent.location.copy(), -1e18
        for nb in neighbor_coords:
            if not inside(nb):
                continue
            s_frontier = 0.0
            try:
                nd = self.node_manager.nodes_dict.find(np.around(nb, 1).tolist()).data
                s_frontier = float(max(nd.utility, 0.0))
            except Exception:
                pass
            rr, cc = self._world_to_cell_rc(nb, map_info)
            risk = 1.0
            pmap = (self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
                    if self.robots[0].pred_mean_map_info is not None else 1.0)
            if isinstance(pmap, np.ndarray):
                if 0 <= rr < pmap.shape[0] and 0 <= cc < pmap.shape[1]:
                    risk = 1.0 - float(pmap[rr, cc])
            score = s_frontier - RDV_RISK_LAMBDA * risk
            if score > best_score:
                best_score, best = score, np.array(nb, dtype=float)
        return best

    # -------------------------------------------------------------------------
    # 合同进度与回退
    # -------------------------------------------------------------------------
    def _contract_progress_and_fallback(self, t, map_info, trav_mask, unknown_mask, p_free):
        if self.contract is None or self.contract.status != 'active':
            return

        # 1) 所有人已进入圈内 → 完成
        all_in = all(self.contract.within_region(r.location) for r in self.robots)
        if all_in:
            if RDV_VERBOSE:
                print(f"[RDV] success rendezvous @t={t}")
            self.contract.status = 'done'
            # 任务完成后清空合同
            self.contract = None
            return

        # 2) 超过时间窗 → 尝试备选 / 否则失败
        if t > int(self.contract.t_max):
            if RDV_VERBOSE:
                print(f"[RDV] timeout @t={t}")
            # 如果已有备选但当前不是备选 → 切到备选并重算调度
            if self.contract.has_backup() and self.contract.target_idx == 0:
                switched = self.contract.switch_to_backup()
                if switched:
                    ok = self._activate_contract_with_schedule(
                        self.contract, target_idx=1, t_now=t,
                        map_info=map_info, belief_map=map_info.map, p_free=p_free
                    )
                    if ok and RDV_VERBOSE:
                        print(f"[RDV] switch to backup target @t={t}")
                    if ok:
                        return
            # 如无备选，则尝试从候选池补一个备选点
            backup_added = self._try_add_backup_and_activate(t, map_info, p_free)
            if backup_added:
                return
            # 还是不行 → 标记失败并清空
            self.contract.status = 'failed'
            self.contract = None
            return

        # 3) 主目标圈内没有任何图节点（无法落点） → 切备选/增备选（保持“之前逻辑”：强制导航也依赖圈内节点）
        if len(self._region_goal_keys(self.contract.P, self.contract.r)) == 0:
            if RDV_VERBOSE:
                print(f"[RDV] region has no graph node; try backup @t={t}")
            # 先尝试已有备选
            if self.contract.has_backup() and self.contract.target_idx == 0:
                switched = self.contract.switch_to_backup()
                if switched:
                    ok = self._activate_contract_with_schedule(
                        self.contract, target_idx=1, t_now=t,
                        map_info=map_info, belief_map=map_info.map, p_free=p_free
                    )
                    if ok:
                        return
            # 再尝试从候选池添加备选
            backup_added = self._try_add_backup_and_activate(t, map_info, p_free)
            if backup_added:
                return
            # 实在不行，等待探索开路
            return

        # 4) 其它情况：继续执行（强制导航由主循环完成）
        return

    def _try_add_backup_and_activate(self, t_now, map_info, p_free):
        """从候选池里找一个和当前主点距离足够远的备选，加入并激活。"""
        if not self.candidate_buffer:
            return False
        primary = {'P': self.contract.P_list[0]}
        cand_backup = self._select_best_candidate_from_buffer(idx=1, min_dist=max(1.0, 0.8 * self.contract.r), ref=primary)
        if cand_backup is None:
            return False
        # 添加备选点
        self.contract.P_list.append(np.array(cand_backup['P'], dtype=float))
        self.contract.t_min_list.append(int(cand_backup['t_min']))
        self.contract.t_max_list.append(int(cand_backup['t_max']))
        # 切换并激活
        self.contract.target_idx = 1
        ok = self._activate_contract_with_schedule(
            self.contract, target_idx=1, t_now=t_now, map_info=map_info, belief_map=map_info.map, p_free=p_free
        )
        if ok and RDV_VERBOSE:
            print(f"[RDV] add & activate backup center={cand_backup['P']} @t={t_now}")
        return ok

    def _compute_groups_from_positions(self, positions):
        n = len(positions)
        if n == 0:
            return []
        used = [False] * n
        groups = []
        for i in range(n):
            if used[i]:
                continue
            comp, q = [], [i]
            used[i] = True
            while q:
                u = q.pop()
                comp.append(u)
                for v in range(n):
                    if used[v]:
                        continue
                    if np.linalg.norm(np.asarray(positions[u]) - np.asarray(positions[v])) <= COMMS_RANGE + 1e-6:
                        used[v] = True
                        q.append(v)
            groups.append(tuple(sorted(comp)))
        return groups

    def _world_to_cell_rc(self, world_xy, map_info):
        cell = get_cell_position_from_coords(np.array(world_xy, dtype=float), map_info)
        return int(cell[1]), int(cell[0])

    def _find_nearest_valid_cell(self, mask, start_rc):
        q = deque([tuple(start_rc)])
        visited = {tuple(start_rc)}
        H, W = mask.shape
        while q:
            r, c = q.popleft()
            if 0 <= r < H and 0 <= c < W and mask[r, c]:
                return np.array([r, c])
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                    q.append((nr, nc))
                    visited.add((nr, nc))
        return np.asarray(start_rc)

    def _bfs_dist_map(self, trav_mask, start_rc):
        H, W = trav_mask.shape
        dist_map = np.full((H, W), np.inf, dtype=np.float32)
        q = deque([start_rc])
        if 0 <= start_rc[0] < H and 0 <= start_rc[1] < W and trav_mask[start_rc]:
            dist_map[start_rc[0], start_rc[1]] = 0.0
        while q:
            r, c = q.popleft()
            base = dist_map[r, c]
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and trav_mask[nr, nc] and np.isinf(dist_map[nr, nc]):
                    dist_map[nr, nc] = base + 1.0
                    q.append((nr, nc))
        return dist_map

    def _bfs_path_rc(self, trav_mask, start_rc, goal_rc):
        H, W = trav_mask.shape
        parent = {tuple(start_rc): None}
        q = deque([tuple(start_rc)])
        if not (0 <= start_rc[0] < H and 0 <= start_rc[1] < W and trav_mask[start_rc[0], start_rc[1]]):
            return []
        while q:
            r, c = q.popleft()
            if (r, c) == tuple(goal_rc):
                path = []
                cur = (r, c)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and trav_mask[nr, nc] and (nr, nc) not in parent:
                    parent[(nr, nc)] = (r, c)
                    q.append((nr, nc))
        return []

    def _bresenham_line_rc(self, r0, c0, r1, c1):
        points = []
        dr, dc = abs(r1 - r0), abs(c1 - c0)
        sr, sc = (1 if r1 >= r0 else -1), (1 if c1 >= c0 else -1)
        if dc > dr:
            err, r = dc // 2, r0
            for c in range(c0, c1 + sc, sc):
                points.append((r, c))
                err -= dr
                if err < 0:
                    r += sr
                    err += dc
        else:
            err, c = dr // 2, c0
            for r in range(r0, r1 + sr, sr):
                points.append((r, c))
                err -= dc
                if err < 0:
                    c += sc
                    err += dr
        return points

    def plot_env(self, step):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(18, 10), dpi=110)

        gs = GridSpec(N_AGENTS, 3, figure=fig, width_ratios=[2.5, 1.2, 1.2], wspace=0.15, hspace=0.1)
        ax_global = fig.add_subplot(gs[:, 0])
        ax_locals_obs = [fig.add_subplot(gs[i, 1]) for i in range(N_AGENTS)]
        ax_locals_pred = [fig.add_subplot(gs[i, 2]) for i in range(N_AGENTS)]
        agent_colors = plt.cm.get_cmap('cool', N_AGENTS)

        global_info = MapInfo(self.env.global_belief, self.env.belief_origin_x, self.env.belief_origin_y,
                              self.env.cell_size)
        ax_global.set_title(f"Global View | Step {step}/{MAX_EPISODE_STEP}", fontsize=14, pad=10)
        ax_global.imshow(global_info.map, cmap='gray', origin='lower')
        ax_global.set_aspect('equal', adjustable='box')
        ax_global.set_axis_off()

        if self.robots and self.robots[0].pred_mean_map_info is not None:
            pred_mean = self.robots[0].pred_mean_map_info.map.astype(np.float32) / 255.0
            belief = global_info.map
            unknown_mask = (belief == UNKNOWN)
            prob = np.zeros_like(pred_mean)
            prob[unknown_mask] = pred_mean[unknown_mask]
            ax_global.imshow(prob, cmap='magma', origin='lower', alpha=0.35)

        groups = self._compute_groups_from_positions(self.env.robot_locations)
        for group in groups:
            for i_idx, i in enumerate(list(group)):
                for j in list(group)[i_idx + 1:]:
                    p1 = self._world_to_cell_rc(self.robots[i].location, global_info)
                    p2 = self._world_to_cell_rc(self.robots[j].location, global_info)
                    ax_global.plot([p1[1], p2[1]], [p1[0], p2[0]], color="#33ff88", lw=2, alpha=0.8, zorder=5)

        for i, r in enumerate(self.robots):
            pos_cell = self._world_to_cell_rc(r.location, global_info)
            if hasattr(r, 'trajectory_x') and r.trajectory_x:
                traj_cells = [self._world_to_cell_rc(np.array([x, y]), global_info)
                              for x, y in zip(r.trajectory_x, r.trajectory_y)]
                ax_global.plot([c for rr, c in traj_cells], [rr for rr, c in traj_cells],
                               color=agent_colors(i), lw=1.5, zorder=3)
            comms_radius = patches.Circle((pos_cell[1], pos_cell[0]), COMMS_RANGE / CELL_SIZE,
                                          fc=(0, 1, 0, 0.05), ec=(0, 1, 0, 0.4), ls='--', lw=1.5, zorder=4)
            ax_global.add_patch(comms_radius)
            ax_global.plot(pos_cell[1], pos_cell[0], 'o', ms=10, mfc=agent_colors(i),
                           mec='white', mew=1.5, zorder=10)

        if self.contract is not None:
            for k, Pk in enumerate(self.contract.P_list):
                p_cell = self._world_to_cell_rc(Pk, global_info)
                if k == self.contract.target_idx and self.contract.status == 'active':
                    ax_global.plot(p_cell[1], p_cell[0], '*', ms=22, mfc='yellow', mec='white', mew=2, zorder=12)
                else:
                    ax_global.plot(p_cell[1], p_cell[0], '+', ms=20, c=('yellow' if k == 0 else 'orange'),
                                   mew=2, zorder=11)
                radius = patches.Circle(
                    (p_cell[1], p_cell[0]),
                    self.contract.r / CELL_SIZE,
                    fc=(1, 1, 0, 0.07 if k == self.contract.target_idx else 0.04),
                    ec=('yellow' if k == 0 else 'orange'),
                    ls='--', lw=1.8, zorder=11
                )
                ax_global.add_patch(radius)

            if self.contract.status == 'active':
                ax_global.text(5, 5,
                               f"RDV(active) target={self.contract.target_idx} "
                               f"T_tar: {self.contract.meta.get('T_tar', '-')}",
                               fontsize=10, color='yellow', ha='left', va='top')
            elif self.contract.status == 'armed':
                ax_global.text(5, 5, f"RDV(armed) waiting for disconnect",
                               fontsize=10, color='yellow', ha='left', va='top')

        for i, r in enumerate(self.robots):
            ax_obs = ax_locals_obs[i]
            local_map_info = r.map_info
            ax_obs.set_title(f"Agent {i} View", fontsize=10, pad=5)
            ax_obs.imshow(local_map_info.map, cmap='gray', origin='lower')
            ax_obs.set_aspect('equal', adjustable='box')
            pos_cell_local = self._world_to_cell_rc(r.location, local_map_info)
            ax_obs.plot(pos_cell_local[1], pos_cell_local[0], 'o', ms=8, mfc=agent_colors(i),
                        mec='white', mew=1.5, zorder=10)
            if r.intent_seq:
                intent_world = [r.location] + r.intent_seq
                intent_cells = [self._world_to_cell_rc(pos, local_map_info) for pos in intent_world]
                ax_obs.plot([c for rr, c in intent_cells], [rr for rr, c in intent_cells],
                            'x--', c=agent_colors(i), lw=2, ms=6, zorder=8)
            ax_obs.set_axis_off()

            ax_pred = ax_locals_pred[i]
            ax_pred.set_title(f"Agent {i} Predicted (local)", fontsize=10, pad=5)
            ax_pred.set_aspect('equal', adjustable='box')
            ax_pred.set_axis_off()

            try:
                if r.pred_mean_map_info is not None or r.pred_max_map_info is not None:
                    pred_info = r.pred_mean_map_info if r.pred_mean_map_info is not None else r.pred_max_map_info
                    pred_local = r.get_updating_map(r.location, base=pred_info)
                    belief_local = r.get_updating_map(r.location, base=r.map_info)

                    ax_pred.imshow(pred_local.map, cmap='gray', origin='lower', vmin=0, vmax=255)
                    alpha_mask = (belief_local.map == FREE) * 0.45
                    ax_pred.imshow(belief_local.map, cmap='Blues', origin='lower', alpha=alpha_mask)

                    rc = get_cell_position_from_coords(r.location, pred_local)
                    ax_pred.plot(rc[0], rc[1], 'mo', markersize=8, zorder=6)
                else:
                    ax_pred.text(0.5, 0.5, 'No prediction', ha='center', va='center', fontsize=9)
            except Exception as e:
                ax_pred.text(0.5, 0.5, f'Pred plot err:\n{e}', ha='center', va='center', fontsize=8)

        out_path = os.path.join(self.run_dir, f"t{step:04d}.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        self.env.frame_files.append(out_path)
