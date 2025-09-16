# node_manager.py
import time
import heapq
import math
import numpy as np

from utils import *            # get_updating_node_coords, get_frontier_in_map, get_cell_position_from_coords,
                               # check_collision, get_quad_tree_box, MapInfo, etc.
from parameter import *        # SENSOR_RANGE, NODE_RESOLUTION, UTILITY_RANGE, MIN_UTILITY, FRONTIER_CELL_SIZE,
                               # UPDATING_MAP_SIZE, ENABLE_RAREFACTION, CLUSTER_RANGE, THR_NEXT_WAYPOINT, MAX_ASTAR_DIST
import quads


class KeyNode:
    """关键图中的轻量节点，仅保最小必要字段"""
    def __init__(self, coords, utility, visited):
        self.coords = np.around(np.array(coords, dtype=float), 1)
        self.utility = float(utility)
        self.visited = int(visited)
        self.neighbor_set = set()
        self.neighbor_set.add((self.coords[0], self.coords[1]))

    def add_neighbor(self, nb_coords):
        self.neighbor_set.add((round(float(nb_coords[0]), 1), round(float(nb_coords[1]), 1)))


class Node:
    """原始图中的完整节点，负责 utility 与邻接的维护"""
    def __init__(self, coords, frontiers, updating_map_info):
        self.coords = np.around(np.array(coords, dtype=float), 1)
        self.utility_range = UTILITY_RANGE
        self.utility = 0
        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info)
        self.visited = 0

        # 5×5 局部邻接（中心为自身）
        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_set = set()
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_set.add((self.coords[0], self.coords[1]))
        self.need_update_neighbor = True

    def initialize_observable_frontiers(self, frontiers, updating_map_info):
        if len(frontiers) == 0:
            self.utility = 0
            return set()
        observable_frontiers = set()
        frontiers_arr = np.array(list(frontiers)).reshape(-1, 2)
        dist_list = np.linalg.norm(frontiers_arr - self.coords, axis=-1)
        new_frontiers_in_range = frontiers_arr[dist_list < self.utility_range]
        for point in new_frontiers_in_range:
            if not check_collision(self.coords, point, updating_map_info):
                observable_frontiers.add((round(float(point[0]), 1), round(float(point[1]), 1)))
        self.utility = len(observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            observable_frontiers = set()
        return observable_frontiers

    def update_neighbor_nodes(self, updating_map_info, nodes_dict):
        """只在机器人附近调用，按 5×5 模板尝试连边（双向维护）"""
        H = self.neighbor_matrix.shape[0]
        center = H // 2
        for i in range(H):
            for j in range(H):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                if i == center and j == center:
                    self.neighbor_matrix[i, j] = 1
                    continue

                nx = self.coords[0] + (i - center) * NODE_RESOLUTION
                ny = self.coords[1] + (j - center) * NODE_RESOLUTION
                neighbor_coords = np.around(np.array([nx, ny]), 1)
                hit = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                if hit is None:
                    continue

                neighbor_node = hit.data
                collision = check_collision(self.coords, neighbor_coords, updating_map_info)

                rx = center + (center - i)
                ry = center + (center - j)

                if not collision:
                    self.neighbor_matrix[i, j] = 1
                    self.neighbor_set.add((neighbor_coords[0], neighbor_coords[1]))
                    neighbor_node.neighbor_matrix[rx, ry] = 1
                    neighbor_node.neighbor_set.add((self.coords[0], self.coords[1]))

        if self.utility == 0:
            self.need_update_neighbor = False

    def update_node_observable_frontiers(self, new_frontiers, global_frontiers, updating_map_info):
        # 移除已消失的 frontiers
        to_remove = []
        for f in self.observable_frontiers:
            if f not in global_frontiers:
                to_remove.append(f)
        for f in to_remove:
            self.observable_frontiers.discard(f)

        # 增加新 frontiers
        if len(new_frontiers) > 0:
            arr = np.array(list(new_frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(arr - self.coords, axis=-1)
            in_range = arr[dist_list < self.utility_range]
            for p in in_range:
                if not check_collision(self.coords, p, updating_map_info):
                    self.observable_frontiers.add((round(float(p[0]), 1), round(float(p[1]), 1)))

        self.utility = len(self.observable_frontiers)
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()

    def set_visited(self):
        self.visited = 1
        self.observable_frontiers = set()
        self.utility = 0


class NodeManager:
    def __init__(self, plot=False):
        # 原始图（四叉树索引）
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.plot = plot
        self.frontier = None

        # --- 稀疏关键图 ---
        self.key_node_dict = {}                 # {(x,y): KeyNode}
        self.path_to_nearest_frontier = None
        self.dist_to_nearest_frontier = 1e8

    # -------- 基础原始图操作 --------
    def check_node_exist_in_dict(self, coords):
        key = (round(float(coords[0]), 1), round(float(coords[1]), 1))
        return self.nodes_dict.find(key)

    def add_node_to_dict(self, coords, frontiers, updating_map_info):
        key = (round(float(coords[0]), 1), round(float(coords[1]), 1))
        node = Node(np.array(key), frontiers, updating_map_info)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def remove_node_from_dict(self, node):
        # 双向删边
        for nb in list(node.neighbor_set):
            if nb != (node.coords[0], node.coords[1]):
                hit = self.nodes_dict.find(nb)
                if hit is not None:
                    hit.data.neighbor_set.discard((node.coords[0], node.coords[1]))
        self.nodes_dict.remove((node.coords[0], node.coords[1]))

    def update_graph(self, robot_location, frontiers, updating_map_info, map_info):
        """维护原始图（局部增量），与原版一致"""
        node_coords, _ = get_updating_node_coords(robot_location, updating_map_info)

        # 前沿增量
        if self.frontier is None:
            new_frontier = set(frontiers)
        else:
            new_frontier = set(frontiers) - set(self.frontier)
            # 过滤超出感知范围的新增前沿（减少无意义更新）
            new_out_range = []
            for f in new_frontier:
                if np.linalg.norm(robot_location - np.array(f).reshape(2)) > SENSOR_RANGE + FRONTIER_CELL_SIZE:
                    new_out_range.append(f)
            for f in new_out_range:
                if f in new_frontier:
                    new_frontier.remove(f)
        self.frontier = set(frontiers)

        all_node_list = []
        global_frontiers = get_frontier_in_map(map_info)

        for coords in node_coords:
            hit = self.check_node_exist_in_dict(coords)
            if hit is None:
                node = self.add_node_to_dict(coords, self.frontier, updating_map_info)
            else:
                node = hit.data
                # 远 + 无收益 → 跳过
                if node.utility == 0 or np.linalg.norm(node.coords - robot_location) > 2 * SENSOR_RANGE:
                    pass
                else:
                    node.update_node_observable_frontiers(new_frontier, global_frontiers, updating_map_info)
            all_node_list.append(node)

        # 附近邻接更新
        for node in all_node_list:
            if node.need_update_neighbor and np.linalg.norm(node.coords - robot_location) < (SENSOR_RANGE + NODE_RESOLUTION):
                node.update_neighbor_nodes(updating_map_info, self.nodes_dict)

    # -------- 稀疏关键图：构建与访问 --------
    def iter_active_nodes(self):
        """
        Agent 侧统一遍历接口：
        - ENABLE_RAREFACTION=True 且关键图非空 → 遍历关键图
        - 否则 → 遍历原始图
        """
        use_sparse = bool(globals().get('ENABLE_RAREFACTION', True))
        if use_sparse and len(self.key_node_dict) > 0:
            for (x, y), kn in self.key_node_dict.items():
                # 产出“视图对象”，与 Node 的最小公共接口一致
                yield type("View", (), dict(
                    coords=np.array([x, y], dtype=float),
                    utility=float(kn.utility),
                    neighbor_set=set(kn.neighbor_set)
                ))()
        else:
            for it in self.nodes_dict.__iter__():
                yield it.data

    def build_key_graph(self, robot_location, map_info):
        """
        从原始图派生一张稀疏关键图：self.key_node_dict
        策略：
          1) 近处(SENSOR_RANGE)所有有用点：逐个最短路，抽取首段/拐点入关键图
          2) 远处：按 CLUSTER_RANGE + 连通性聚类，每簇仅对“簇心”走一条骨干路并抽拐点
          3) 邻接回填：仅对关键点之间、且原图已有邻接的连边
        """
        if not globals().get('ENABLE_RAREFACTION', True):
            self.key_node_dict = {}
            self.path_to_nearest_frontier, self.dist_to_nearest_frontier = None, 1e8
            return

        self.key_node_dict = {}
        self.path_to_nearest_frontier, self.dist_to_nearest_frontier = None, 1e8

        # 收集 utility>0 的“有用节点”
        useful_nodes = []
        for it in self.nodes_dict.__iter__():
            nd = it.data
            if nd.utility > 0:
                useful_nodes.append(nd)
        if not useful_nodes:
            return

        # 当前节点（或最近邻）
        cur_hit = self.nodes_dict.find((round(float(robot_location[0]), 1), round(float(robot_location[1]), 1)))
        if cur_hit is None:
            cur_hit = self.nodes_dict.nearest_neighbors(
                (round(float(robot_location[0]), 1), round(float(robot_location[1]), 1)), 1
            )[0]
        current = cur_hit.data
        self._ensure_keynode(current)

        # 近处一簇
        near_box = get_quad_tree_box(current.coords, 1.0 * SENSOR_RANGE)
        near_nodes = [it.data for it in self.nodes_dict.within_bb(near_box)]
        for nd in near_nodes:
            if nd.utility <= 0:
                continue
            self._add_path_skeleton(current.coords, nd.coords, map_info)

        # 远处按簇挑中心走一条骨干路
        centers = self._cluster_useful_by_connectivity(useful_nodes, center_radius=globals().get('CLUSTER_RANGE', 2.5 * SENSOR_RANGE))
        for center in centers:
            # 距当前很近的在“近处一簇”已处理，这里不会重复
            self._add_path_skeleton(current.coords, center.coords, map_info)

        # 邻接回填：只有关键点之间且原图有边的才连
        keyset = set(self.key_node_dict.keys())
        for key in list(self.key_node_dict.keys()):
            hit = self.nodes_dict.find(key)
            if hit is None:
                continue
            node = hit.data
            for nb in node.neighbor_set:
                if nb in keyset:
                    self.key_node_dict[key].add_neighbor(nb)
                    self.key_node_dict[nb].add_neighbor(key)

    # -------- 稀疏关键图：内部工具 --------
    def _ensure_keynode(self, node):
        key = (round(float(node.coords[0]), 1), round(float(node.coords[1]), 1))
        if key not in self.key_node_dict:
            self.key_node_dict[key] = KeyNode(node.coords, node.utility, node.visited)
        return self.key_node_dict[key]

    def _add_path_skeleton(self, start, goal, map_info):
        """对 start→goal 求一条路，并把“首段 + 拐点”压入关键图"""
        s = (round(float(start[0]), 1), round(float(start[1]), 1))
        g = (round(float(goal[0]), 1), round(float(goal[1]), 1))
        if s == g:
            return

        path, dist = self.a_star(s, g, max_dist=globals().get('MAX_ASTAR_DIST', None))
        if dist == 1e8 or len(path) == 0:
            return

        # 记录“最近frontier路径”（供引导）
        if dist < self.dist_to_nearest_frontier:
            self.dist_to_nearest_frontier = dist
            self.path_to_nearest_frontier = path

        # 保证起点进入关键图
        ref_kn = self._ensure_key_by_coords(s)

        # 抽取首段/拐点
        first_hop_done = False
        for i, coords in enumerate(path):
            coords = (round(float(coords[0]), 1), round(float(coords[1]), 1))
            if not first_hop_done:
                # 第一跳：若不可见/非直接邻居/步长过大 → 把 path[i-1] 作为第一关键点
                start_hit = self.nodes_dict.find(s)
                assert start_hit is not None
                start_node = start_hit.data

                is_neighbor = (coords in start_node.neighbor_set)
                too_far = (np.linalg.norm(np.array(coords) - np.array(s)) > globals().get('THR_NEXT_WAYPOINT', NODE_RESOLUTION * 1.5))
                occluded = check_collision(np.array(s), np.array(coords), map_info)

                if (not is_neighbor) and (occluded or too_far):
                    if i >= 1:
                        k1 = (round(float(path[i - 1][0]), 1), round(float(path[i - 1][1]), 1))
                        self._ensure_key_by_coords(k1)
                        self.key_node_dict[k1].add_neighbor(s)
                        ref_kn.add_neighbor(k1)
                    self._ensure_key_by_coords(coords)  # 把当前步也放进来
                    first_hop_done = True
                else:
                    # 如果第一步就落在一个“有用点”，也直接入关键图并连边
                    hit = self.nodes_dict.find(coords)
                    if hit is not None and hit.data.utility > 0:
                        self._ensure_key_by_coords(coords)
                        self.key_node_dict[coords].add_neighbor(s)
                        ref_kn.add_neighbor(coords)
                        first_hop_done = True
            else:
                self._ensure_key_by_coords(coords)

    def _ensure_key_by_coords(self, coords):
        """用坐标确保关键点存在；如果原图无该点，则跳过（防止异常）"""
        key = (round(float(coords[0]), 1), round(float(coords[1]), 1))
        if key in self.key_node_dict:
            return self.key_node_dict[key]
        hit = self.nodes_dict.find(key)
        if hit is None:
            return None
        nd = hit.data
        return self._ensure_keynode(nd)

    def _cluster_useful_by_connectivity(self, useful_nodes, center_radius):
        """
        极简聚类：以每个 useful 节点为“候选中心”，收集其在 center_radius 内
        且通过原图邻接可达的 useful 节点；返回一组“中心节点列表”（to do：可换更聪明策略（不过我估计也没时间 ~））。
        """
        centers = []
        seen = set()
        # 复数键
        def kxy(nd): return complex(round(float(nd.coords[0]), 1), round(float(nd.coords[1]), 1))
        key_all = {kxy(nd): nd for nd in useful_nodes}

        for nd in useful_nodes:
            k = kxy(nd)
            if k in seen:
                continue
            cluster = set([k])
            frontier = [nd]
            seen.add(k)

            while frontier:
                cur = frontier.pop()
                for nb in cur.neighbor_set:
                    kk = complex(round(float(nb[0]), 1), round(float(nb[1]), 1))
                    if kk in key_all and kk not in cluster:
                        if np.linalg.norm(np.array([nb[0] - nd.coords[0], nb[1] - nd.coords[1]])) <= center_radius:
                            cluster.add(kk)
                            seen.add(kk)
                            hit = self.nodes_dict.find((round(float(nb[0]), 1), round(float(nb[1]), 1)))
                            if hit is not None:
                                frontier.append(hit.data)
            # 简单取“触发者 nd”为簇心（可替换为 utility 最大等）
            centers.append(nd)
        return centers

    # -------- 最短路（A*） --------
    def a_star(self, start, destination, max_dist=None):
        """
        start/destination: (x, y) 已按 0.1 保留
        返回: (path_without_start, distance)
        """
        s = (round(float(start[0]), 1), round(float(start[1]), 1))
        d = (round(float(destination[0]), 1), round(float(destination[1]), 1))

        if self.nodes_dict.find(s) is None:
            return [], 1e8
        if self.nodes_dict.find(d) is None:
            return [], 1e8
        if s == d:
            return [], 0.0

        open_set = {s}
        closed = set()
        g = {s: 0.0}
        parent = {s: s}

        def h(a, b):
            return float(np.linalg.norm(np.array(a) - np.array(b)))

        pq = []
        heapq.heappush(pq, (h(s, d), s))

        while open_set:
            _, u = heapq.heappop(pq)
            node_u_hit = self.nodes_dict.find(u)
            if node_u_hit is None:
                open_set.discard(u)
                continue
            node_u = node_u_hit.data

            if max_dist is not None and g[u] > float(max_dist):
                return [], 1e8

            if u == d:
                # 回溯
                path = []
                cur = u
                length = g[u]
                while parent[cur] != cur:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path, round(float(length), 2)

            # 扩展邻接
            for nb in node_u.neighbor_set:
                v = (round(float(nb[0]), 1), round(float(nb[1]), 1))
                if v in closed:
                    continue
                cost = float(np.linalg.norm(np.array(v) - np.array(u)))
                alt = g[u] + cost
                if (v not in open_set) or (alt < g.get(v, 1e18)):
                    open_set.add(v)
                    parent[v] = u
                    g[v] = alt
                    heapq.heappush(pq, (alt + h(v, d), v))

            open_set.discard(u)
            closed.add(u)

        return [], 1e8
