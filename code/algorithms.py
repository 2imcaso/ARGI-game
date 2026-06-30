from collections import deque
from itertools import combinations
import heapq
import math
import random


INF = float("inf")

UNINFORMED_ALGORITHMS = ("BFS", "DFS", "IDS", "UCS")
INFORMED_ALGORITHMS = ("Greedy", "IDSA", "A*", "A*_v2")
LOCAL_ALGORITHMS = ("Local Beam", "Hill Climbing", "Annealing", "Restart Hill")
ONLINE_ALGORITHMS = (
    "Online A*",
    "Online BFS",
    "Belief-State BFS",
    "AND-OR Search",
)
CSP_ALGORITHMS = ("Backtrack", "Fwd Check", "AC-3", "Min Conflict")
CSP_CROPS = ("corn", "tomato", "wheat", "carrot")
CSP_FORBIDDEN_PAIRS = {
    frozenset(("corn", "tomato")),
    frozenset(("wheat", "carrot")),
}
ADVERSARIAL_ALGORITHMS = (
    "Minimax",
    "Alpha-Beta",
    "Expectimax",
    "Expectiminimax",
)

def csp_crop_pair_valid(left, right):
    """Return whether two adjacent Mode 5 crops satisfy all binary constraints."""
    if left is None or right is None:
        return True
    if left == right:
        return False
    if frozenset((left, right)) in CSP_FORBIDDEN_PAIRS:
        return False
    return True

ALGORITHM_GROUPS = {
    1: UNINFORMED_ALGORITHMS,
    2: INFORMED_ALGORITHMS,
    3: LOCAL_ALGORITHMS,
    4: ONLINE_ALGORITHMS,
    5: CSP_ALGORITHMS,
    6: ADVERSARIAL_ALGORITHMS,
}

ALGORITHM_GROUP_NAMES = {
    1: "Uninformed Search",
    2: "Informed Search",
    3: "Local Search",
    4: "Online Search",
    5: "Constraint Satisfaction",
    6: "Adversarial Search",
}


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path[1:]


def unit_step_cost(current, next_tile):
    return 1


def bfs(start, goal, blocked, neighbors):
    queue = deque([start])
    came_from = {}
    visited = {start}
    explored = set()

    while queue:
        current = queue.popleft()
        explored.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, explored, {
                "Nodes explored": len(explored),
                "Path length": len(path),
                "Queue max": len(visited),
            }

        for next_tile in neighbors(current, blocked):
            if next_tile not in visited:
                visited.add(next_tile)
                came_from[next_tile] = current
                queue.append(next_tile)

    return [], explored, {
        "Nodes explored": len(explored),
        "Path length": 0,
        "Queue max": len(visited),
    }


def dfs(start, goal, blocked, neighbors):
    if start == goal:
        return [], {start}, {
            "Nodes explored": 1,
            "Path length": 0,
            "Stack max": 1,
        }

    stack = [start]
    frontier = {start}
    came_from = {}
    reached = {start}
    explored = set()
    max_frontier = 1

    while stack:
        current = stack.pop()
        frontier.remove(current)
        explored.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, explored, {
                "Nodes explored": len(explored),
                "Path length": len(path),
                "Stack max": max_frontier,
            }

        for next_tile in neighbors(current, blocked):
            if next_tile not in reached and next_tile not in frontier:
                reached.add(next_tile)
                came_from[next_tile] = current
                stack.append(next_tile)
                frontier.add(next_tile)
                max_frontier = max(max_frontier, len(stack))

    return [], explored, {
        "Nodes explored": len(explored),
        "Path length": 0,
        "Stack max": max_frontier,
    }


def _depth_limited_search(start, goal, blocked, neighbors, limit):
    if start == goal:
        return [], {start}, False

    stack = [(start, 0)]
    came_from = {}
    reached = {start}
    frontier = {start}
    explored = set()
    cutoff = False

    while stack:
        current, depth = stack.pop()
        frontier.remove(current)
        explored.add(current)

        if current == goal:
            return reconstruct_path(came_from, current), explored, False

        if depth >= limit:
            cutoff = True
            continue

        for next_tile in neighbors(current, blocked):
            if next_tile not in reached and next_tile not in frontier:
                reached.add(next_tile)
                came_from[next_tile] = current
                stack.append((next_tile, depth + 1))
                frontier.add(next_tile)

    return [], explored, cutoff


def ids(start, goal, blocked, neighbors, max_depth=200):
    unique_explored = set()
    total_expansions = 0
    last_iteration_count = 0
    for limit in range(max_depth + 1):
        path, explored, cutoff = _depth_limited_search(
            start, goal, blocked, neighbors, limit)
        unique_explored |= explored
        last_iteration_count = len(explored)
        total_expansions += last_iteration_count
        if path or start == goal:
            return path, unique_explored, {
                "Nodes explored": total_expansions,
                "Unique explored": len(unique_explored),
                "Last iteration": last_iteration_count,
                "Path length": len(path),
                "Depth limit": limit,
            }
        if not cutoff:
            break

    return [], unique_explored, {
        "Nodes explored": total_expansions,
        "Unique explored": len(unique_explored),
        "Last iteration": last_iteration_count,
        "Path length": 0,
        "Depth limit": max_depth,
    }


def ucs(start, goal, blocked, neighbors, counter, step_cost=unit_step_cost):
    open_set = []
    heapq.heappush(open_set, (0, next(counter), start))
    came_from = {}
    best_cost = {start: 0}
    explored = set()

    while open_set:
        cost, _, current = heapq.heappop(open_set)
        if cost > best_cost.get(current, INF):
            continue
        explored.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, explored, {
                "Nodes explored": len(explored),
                "Path length": len(path),
                "Cost": cost,
            }

        for next_tile in neighbors(current, blocked):
            next_cost = cost + step_cost(current, next_tile)
            if next_cost < best_cost.get(next_tile, INF):
                best_cost[next_tile] = next_cost
                came_from[next_tile] = current
                heapq.heappush(open_set, (next_cost, next(counter), next_tile))

    return [], explored, {
        "Nodes explored": len(explored),
        "Path length": 0,
    }


def astar(start, goal, blocked, neighbors, heuristic, counter,
          step_cost=unit_step_cost):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), next(counter), start))
    came_from = {}
    g_score = {start: 0}
    explored = set()
    closed = set()

    while open_set:
        f_score, _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        explored.add(current)
        closed.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            g = g_score[current]
            h = heuristic(current, goal)
            return path, explored, (g + h, g, h), {
                "Nodes explored": len(explored),
                "Path length": len(path),
                "f(n)": f"{g + h}",
                "g(n)": f"{g}",
                "h(n)": f"{h}",
            }

        for next_tile in neighbors(current, blocked):
            tentative_g = g_score[current] + step_cost(current, next_tile)
            if next_tile in closed:
                continue
            if tentative_g < g_score.get(next_tile, INF):
                came_from[next_tile] = current
                g_score[next_tile] = tentative_g
                h = heuristic(next_tile, goal)
                heapq.heappush(
                    open_set,
                    (tentative_g + h, next(counter), next_tile))

    return [], explored, (0, 0, 0), {
        "Nodes explored": len(explored),
        "Path length": 0,
    }


def astar_4dir_f_update(start, goal, blocked, neighbors, heuristic, counter,
                        step_cost=unit_step_cost):
    """A* core for 4-direction grid movement.

    A node is updated only when the new f(n) is strictly smaller than the
    best f(n) recorded for that node.
    """
    open_set = []
    start_h = heuristic(start, goal)
    heapq.heappush(open_set, (start_h, start_h, 0, next(counter), start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: start_h}
    explored = set()
    max_frontier = 1

    while open_set:
        current_f, _, _, _, current = heapq.heappop(open_set)
        if current_f > f_score.get(current, INF):
            continue

        explored.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            g = g_score[current]
            h = heuristic(current, goal)
            return path, explored, (g + h, g, h), {
                "Nodes explored": len(explored),
                "Path length": len(path),
                "Frontier max": max_frontier,
                "f(n)": f"{g + h}",
                "g(n)": f"{g}",
                "h(n)": f"{h}",
                "Update rule": "only update when f_new < f_old",
                "Neighbor rule": "push only 4 adjacent neighbors",
            }

        for next_tile in neighbors(current, blocked):
            if (abs(next_tile[0] - current[0])
                    + abs(next_tile[1] - current[1]) != 1):
                continue

            tentative_g = g_score[current] + step_cost(current, next_tile)
            h = heuristic(next_tile, goal)
            tentative_f = tentative_g + h
            if tentative_f >= f_score.get(next_tile, INF):
                continue

            came_from[next_tile] = current
            g_score[next_tile] = tentative_g
            f_score[next_tile] = tentative_f
            heapq.heappush(
                open_set,
                (tentative_f, h, tentative_g, next(counter), next_tile))
            max_frontier = max(max_frontier, len(open_set))

    return [], explored, (0, 0, 0), {
        "Nodes explored": len(explored),
        "Path length": 0,
        "Frontier max": max_frontier,
        "Update rule": "only update when f_new < f_old",
        "Neighbor rule": "push only 4 adjacent neighbors",
    }


def greedy(start, goal, blocked, neighbors, heuristic, counter):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), next(counter), start))
    came_from = {}
    visited = {start}
    explored = set()

    while open_set:
        priority, _, current = heapq.heappop(open_set)
        explored.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, explored, {
                "Nodes explored": len(explored),
                "Path length": len(path),
                "h(n)": priority,
            }

        for next_tile in neighbors(current, blocked):
            if next_tile not in visited:
                visited.add(next_tile)
                came_from[next_tile] = current
                heapq.heappush(
                    open_set,
                    (heuristic(next_tile, goal), next(counter), next_tile))

    return [], explored, {
        "Nodes explored": len(explored),
        "Path length": 0,
    }


def idastar_iteration(start, goal, blocked, neighbors, heuristic, bound,
                      max_depth=900):
    """Run one IDA* f-bound iteration for time-based visualization."""
    explored = set()
    depth_limited = False

    def search(path, g_score, bound, visited):
        nonlocal depth_limited
        current = path[-1]
        f_score = g_score + heuristic(current, goal)
        if f_score > bound:
            return f_score, None
        if len(path) >= max_depth:
            depth_limited = True
            return INF, None
        explored.add(current)
        if current == goal:
            return f_score, list(path[1:])

        next_bound = INF
        ordered_neighbors = sorted(
            neighbors(current, blocked),
            key=lambda tile: heuristic(tile, goal))
        for next_tile in ordered_neighbors:
            if next_tile in visited:
                continue
            visited.add(next_tile)
            path.append(next_tile)
            result_bound, result_path = search(
                path, g_score + 1, bound, visited)
            if result_path is not None:
                return result_bound, result_path
            next_bound = min(next_bound, result_bound)
            path.pop()
            visited.remove(next_tile)
        return next_bound, None

    next_bound, path = search([start], 0, bound, {start})
    return next_bound, path, explored, depth_limited


def idastar(start, goal, blocked, neighbors, heuristic, max_depth=900):
    limit = heuristic(start, goal)
    total_explored = set()
    depth_limited = False

    while limit < INF:
        next_limit, path, explored, hit_depth_guard = idastar_iteration(
            start, goal, blocked, neighbors, heuristic, limit, max_depth)
        total_explored.update(explored)
        depth_limited = depth_limited or hit_depth_guard
        if path is not None:
            return path, total_explored, {
                "Nodes explored": len(total_explored),
                "Path length": len(path),
                "f limit": limit,
                "Depth guard": max_depth,
            }
        if next_limit == INF:
            break
        limit = next_limit

    stats = {
        "Nodes explored": len(total_explored),
        "Path length": 0,
        "f limit": limit,
        "Depth guard": max_depth,
    }
    if depth_limited:
        stats["Stopped by guard"] = "yes"
    return [], total_explored, stats


TURN_PENALTY = 1.5


def _expand_turn_penalty_node(state, blocked, neighbors):
    tile, prev_dir = state
    results = []
    for next_tile in neighbors(tile, blocked):
        dx = next_tile[0] - tile[0]
        dy = next_tile[1] - tile[1]
        if abs(dx) + abs(dy) != 1:
            continue
        if dx == 1:
            direction = "R"
        elif dx == -1:
            direction = "L"
        elif dy == 1:
            direction = "D"
        else:
            direction = "U"
        turn_cost = (
            0 if prev_dir is None or direction == prev_dir
            else TURN_PENALTY)
        results.append(((next_tile, direction), 1 + turn_cost))
    return results


def _reconstruct_turn_penalty_path(came_from, state):
    path = [state[0]]
    directions = [state[1]]
    while state in came_from:
        state = came_from[state]
        path.append(state[0])
        directions.append(state[1])
    path.reverse()
    directions.reverse()
    turns = sum(
        1 for index in range(1, len(directions))
        if directions[index] is not None
        and directions[index - 1] is not None
        and directions[index] != directions[index - 1])
    return path[1:], turns


def astar_turn_penalty(start, goal, blocked, neighbors, heuristic,
                       prev_dir=None, counter=None, step_cost=unit_step_cost):
    counter = counter or iter(range(10**9))
    start_state = (start, prev_dir)
    start_h = heuristic(start, goal)
    open_set = [(start_h, 0.0, next(counter), start_state)]
    came_from = {}
    g_score = {start_state: 0.0}
    closed = set()
    explored = set()
    max_frontier = 1

    while open_set:
        _, weighted_g, _, state = heapq.heappop(open_set)
        if state in closed:
            continue
        tile, _ = state
        closed.add(state)
        explored.add(tile)

        if tile == goal:
            path, turns = _reconstruct_turn_penalty_path(came_from, state)
            real_g = len(path)
            goal_h = heuristic(tile, goal)
            weighted_f = weighted_g + goal_h
            return path, explored, (weighted_f, real_g, goal_h), {
                "Nodes explored": len(explored),
                "Path length": len(path),
                "Turns": turns,
                "Turn penalty": TURN_PENALTY,
                "f(n)": f"{weighted_f:.1f}",
                "g(n)": f"{real_g} + {turns}*{TURN_PENALTY:.1f}",
                "g steps": real_g,
                "h(n)": f"{goal_h:.1f}",
                "Frontier max": max_frontier,
            }

        for next_state, move_cost in _expand_turn_penalty_node(
                state, blocked, neighbors):
            if next_state in closed:
                continue
            tentative_g = g_score[state] + move_cost
            if tentative_g >= g_score.get(next_state, INF):
                continue
            came_from[next_state] = state
            g_score[next_state] = tentative_g
            next_h = heuristic(next_state[0], goal)
            heapq.heappush(
                open_set,
                (tentative_g + next_h, tentative_g,
                 next(counter), next_state))
            max_frontier = max(max_frontier, len(open_set))

    return [], explored, (0, 0, 0), {
        "Nodes explored": len(explored),
        "Path length": 0,
        "Turns": 0,
        "Turn penalty": TURN_PENALTY,
        "Frontier max": max_frontier,
    }


def find_path_by_algorithm(algorithm, start, goal, blocked, neighbors,
                           heuristic, counter, step_cost=unit_step_cost,
                           prev_dir_override=None):
    if algorithm == "BFS":
        path, explored, stats = bfs(start, goal, blocked, neighbors)
        return path, explored, (0, 0, 0), stats
    if algorithm == "DFS":
        path, explored, stats = dfs(start, goal, blocked, neighbors)
        return path, explored, (0, 0, 0), stats
    if algorithm == "IDS":
        path, explored, stats = ids(start, goal, blocked, neighbors)
        return path, explored, (0, 0, 0), stats
    if algorithm == "UCS":
        path, explored, stats = ucs(
            start, goal, blocked, neighbors, counter, step_cost)
        return path, explored, (0, 0, 0), stats
    if algorithm == "Greedy":
        path, explored, stats = greedy(
            start, goal, blocked, neighbors, heuristic, counter)
        h = heuristic(start, goal)
        return path, explored, (h, 0, h), stats
    if algorithm == "IDSA":
        path, explored, stats = idastar(
            start, goal, blocked, neighbors, heuristic)
        g = len(path)
        h = heuristic(start, goal)
        return path, explored, (g + h, g, h), stats

    if algorithm == "A*_v2":
        return astar_turn_penalty(
            start, goal, blocked, neighbors, heuristic,
            prev_dir=prev_dir_override, counter=counter,
            step_cost=step_cost)

    path, explored, fgh, stats = astar(
        start, goal, blocked, neighbors, heuristic, counter, step_cost)
    return path, explored, fgh, stats


def bfs_to_any_goal(start, goals, blocked, neighbors):
    goals = set(goals)
    queue = deque([start])
    came_from = {}
    visited = {start}
    explored = set()

    while queue:
        current = queue.popleft()
        explored.add(current)

        if current in goals:
            path = reconstruct_path(came_from, current)
            return path, current, explored, {
                "Nodes explored": len(explored),
                "Path length": len(path),
                "Queue max": len(visited),
            }

        for next_tile in neighbors(current, blocked):
            if next_tile not in visited:
                visited.add(next_tile)
                came_from[next_tile] = current
                queue.append(next_tile)

    return [], None, explored, {
        "Nodes explored": len(explored),
        "Path length": 0,
        "Queue max": len(visited),
    }


def dfs_to_any_goal(start, goals, blocked, neighbors):
    goals = set(goals)
    if start in goals:
        return [], start, {start}, {
            "Nodes explored": 1,
            "Path length": 0,
            "Stack max": 1,
        }

    stack = [start]
    frontier = {start}
    came_from = {}
    reached = {start}
    explored = set()
    max_frontier = 1

    while stack:
        current = stack.pop()
        frontier.remove(current)
        explored.add(current)

        if current in goals:
            path = reconstruct_path(came_from, current)
            return path, current, explored, {
                "Nodes explored": len(explored),
                "Path length": len(path),
                "Stack max": max_frontier,
            }

        for next_tile in neighbors(current, blocked):
            if next_tile not in reached and next_tile not in frontier:
                reached.add(next_tile)
                came_from[next_tile] = current
                stack.append(next_tile)
                frontier.add(next_tile)
                max_frontier = max(max_frontier, len(stack))

    return [], None, explored, {
        "Nodes explored": len(explored),
        "Path length": 0,
        "Stack max": max_frontier,
    }


def _depth_limited_search_to_any_goal(start, goals, blocked, neighbors, limit):
    goals = set(goals)
    if start in goals:
        return [], start, {start}, False

    stack = [(start, 0)]
    came_from = {}
    reached = {start}
    frontier = {start}
    explored = set()
    cutoff = False

    while stack:
        current, depth = stack.pop()
        frontier.remove(current)
        explored.add(current)

        if current in goals:
            return reconstruct_path(came_from, current), current, explored, False

        if depth >= limit:
            cutoff = True
            continue

        for next_tile in neighbors(current, blocked):
            if next_tile not in reached and next_tile not in frontier:
                reached.add(next_tile)
                came_from[next_tile] = current
                stack.append((next_tile, depth + 1))
                frontier.add(next_tile)

    return [], None, explored, cutoff


def depth_limited_search_to_any_goal(start, goals, blocked, neighbors, limit):
    """Run one IDS iteration so the controller can visualize each limit."""
    return _depth_limited_search_to_any_goal(
        start, goals, blocked, neighbors, limit)


def ids_to_any_goal(start, goals, blocked, neighbors, max_depth=200):
    unique_explored = set()
    total_expansions = 0
    last_iteration_count = 0
    for limit in range(max_depth + 1):
        path, target, explored, cutoff = _depth_limited_search_to_any_goal(
            start, goals, blocked, neighbors, limit)
        unique_explored |= explored
        last_iteration_count = len(explored)
        total_expansions += last_iteration_count
        if target is not None:
            return path, target, unique_explored, {
                "Nodes explored": total_expansions,
                "Unique explored": len(unique_explored),
                "Last iteration": last_iteration_count,
                "Path length": len(path),
                "Depth limit": limit,
            }
        if not cutoff:
            break

    return [], None, unique_explored, {
        "Nodes explored": total_expansions,
        "Unique explored": len(unique_explored),
        "Last iteration": last_iteration_count,
        "Path length": 0,
        "Depth limit": max_depth,
    }


def ucs_to_any_goal(start, goals, blocked, neighbors, counter,
                    step_cost=unit_step_cost, initial_cost=0):
    goals = set(goals)
    open_set = []
    heapq.heappush(open_set, (initial_cost, 0, initial_cost, next(counter), start))
    came_from = {}
    g_score = {start: initial_cost}
    explored = set()
    closed = set()
    max_frontier = 1
    last_neighbors = 0
    last_pushed = 1

    while open_set:
        current_g, _, _, _, current = heapq.heappop(open_set)
        if current_g > g_score.get(current, INF):
            continue
        if current in closed:
            continue
        closed.add(current)
        explored.add(current)

        neighbor_list = list(neighbors(current, blocked))
        last_neighbors = len(neighbor_list)
        last_pushed = 0
        for next_tile in neighbor_list:
            new_g = current_g + step_cost(current, next_tile)
            if new_g < g_score.get(next_tile, INF):
                g_score[next_tile] = new_g
                came_from[next_tile] = current
                heapq.heappush(
                    open_set,
                    (new_g, new_g, new_g, next(counter), next_tile))
                last_pushed += 1
        max_frontier = max(max_frontier, len(open_set))

        if current in goals:
            path = reconstruct_path(came_from, current)
            return path, current, explored, {
                "Nodes explored": len(explored),
                "Path length": len(path),
                "Frontier max": max_frontier,
                "Cost": current_g,
                "g(n)": current_g,
                "Priority": "g(n)",
                "PQ rule": "push 4-neighbor with priority = tentative g(n)",
                "Neighbor rule": "expand 4-neighbor like A*, no h(n)",
                "Last expanded": f"{current}",
                "Neighbors considered": f"{last_neighbors}",
                "Pushed to PQ": f"{last_pushed}",
            }

    return [], None, explored, {
        "Nodes explored": len(explored),
        "Path length": 0,
        "Frontier max": max_frontier,
    }


def find_path_to_any_goal_by_algorithm(algorithm, start, goals, blocked,
                                       neighbors, counter,
                                       step_cost=unit_step_cost):
    goals = set(goals)
    if not goals:
        return [], None, set(), (0, 0, 0), {
            "Nodes explored": 0,
            "Path length": 0,
        }
    if algorithm == "BFS":
        path, target, explored, stats = bfs_to_any_goal(
            start, goals, blocked, neighbors)
        return path, target, explored, (0, 0, 0), stats
    if algorithm == "DFS":
        path, target, explored, stats = dfs_to_any_goal(
            start, goals, blocked, neighbors)
        return path, target, explored, (0, 0, 0), stats
    if algorithm == "IDS":
        path, target, explored, stats = ids_to_any_goal(
            start, goals, blocked, neighbors)
        return path, target, explored, (0, 0, 0), stats
    if algorithm == "UCS":
        path, target, explored, stats = ucs_to_any_goal(
            start, goals, blocked, neighbors, counter, step_cost)
        return path, target, explored, (0, 0, 0), stats
    return [], None, set(), (0, 0, 0), {
        "Nodes explored": 0,
        "Path length": 0,
    }


def dfs_full_traversal_plan(start, goals, blocked, neighbors):
    """Collect goals in the same explicit LIFO order used by ``dfs``."""
    goals = set(goals)
    stack = [start]
    frontier = {start}
    reached = {start}
    came_from = {}
    explored = set()
    explored_order = []
    plan = []
    max_stack = 1

    while stack and len(plan) < len(goals):
        current = stack.pop()
        frontier.remove(current)
        explored.add(current)
        explored_order.append(current)

        if current in goals:
            plan.append(current)

        for next_tile in neighbors(current, blocked):
            if next_tile in reached or next_tile in frontier:
                continue
            reached.add(next_tile)
            came_from[next_tile] = current
            stack.append(next_tile)
            frontier.add(next_tile)
            max_stack = max(max_stack, len(stack))

    reachable = set(plan)
    stats = {
        "Algorithm": "DFS",
        "Planning mode": "Full garden DFS traversal",
        "Plan targets": len(plan),
        "Unreachable": len(goals - reachable),
        "Nodes explored": len(explored_order),
        "Unique explored": len(explored),
        "Stack max": max_stack,
    }
    return plan, explored, stats


def _nearest_goal_distance(tile, remaining_goals, heuristic):
    if not remaining_goals:
        return 0
    return min(heuristic(tile, goal) for goal in remaining_goals)


def _full_traversal_stats(algorithm, plan, goals, explored, extra=None):
    stats = {
        "Algorithm": algorithm,
        "Planning mode": f"Full garden {algorithm} traversal",
        "Plan targets": len(plan),
        "Unreachable": len(set(goals) - set(plan)),
        "Nodes explored": len(explored),
        "Unique explored": len(set(explored)),
    }
    if extra:
        stats.update(extra)
    return stats


def bfs_full_traversal_plan(start, goals, blocked, neighbors):
    goals = set(goals)
    queue = deque([start])
    visited = {start}
    explored = []
    plan = []

    while queue and len(plan) < len(goals):
        current = queue.popleft()
        explored.append(current)
        if current in goals and current not in plan:
            plan.append(current)

        for next_tile in neighbors(current, blocked):
            if next_tile not in visited:
                visited.add(next_tile)
                queue.append(next_tile)

    stats = _full_traversal_stats(
        "BFS", plan, goals, explored,
        {"Queue max": len(visited)})
    return plan, set(explored), stats


def ids_full_traversal_plan(start, goals, blocked, neighbors, max_depth=200):
    goals = set(goals)
    plan = []
    unique_explored = set()
    total_expansions = 0
    final_limit = 0

    for limit in range(max_depth + 1):
        stack = [(start, 0)]
        reached = {start}
        cutoff = False
        final_limit = limit

        while stack:
            current, depth = stack.pop()
            unique_explored.add(current)
            total_expansions += 1
            if current in goals and current not in plan:
                plan.append(current)
                if len(plan) == len(goals):
                    break

            if depth >= limit:
                cutoff = True
                continue

            for next_tile in neighbors(current, blocked):
                if next_tile not in reached:
                    reached.add(next_tile)
                    stack.append((next_tile, depth + 1))

        if len(plan) == len(goals) or not cutoff:
            break

    stats = _full_traversal_stats(
        "IDS", plan, goals, unique_explored,
        {"Depth limit": final_limit, "Total expansions": total_expansions})
    return plan, unique_explored, stats


def ucs_full_traversal_plan(start, goals, blocked, neighbors, counter,
                            step_cost=unit_step_cost):
    goals = set(goals)
    open_set = []
    heapq.heappush(open_set, (0, 0, next(counter), start))
    came_from = {}
    g_score = {start: 0}
    closed = set()
    explored = []
    plan = []
    target_g = []
    max_frontier = 1
    last_neighbors = 0
    last_pushed = 1

    while open_set and len(plan) < len(goals):
        current_g, _, _, current = heapq.heappop(open_set)
        if current_g > g_score.get(current, INF):
            continue
        if current in closed:
            continue
        closed.add(current)
        explored.append(current)

        neighbor_list = list(neighbors(current, blocked))
        last_neighbors = len(neighbor_list)
        last_pushed = 0
        for next_tile in neighbor_list:
            tentative_g = current_g + step_cost(current, next_tile)
            if tentative_g < g_score.get(next_tile, INF):
                came_from[next_tile] = current
                g_score[next_tile] = tentative_g
                heapq.heappush(
                    open_set,
                    (tentative_g, tentative_g, next(counter), next_tile))
                last_pushed += 1
        max_frontier = max(max_frontier, len(open_set))

        if current in goals and current not in plan:
            plan.append(current)
            target_g.append((current, current_g))

    max_target_g = max((cost for _, cost in target_g), default=0)
    stats = _full_traversal_stats(
        "UCS", plan, goals, explored,
        {
            "Max cost": max_target_g,
            "Target g(n)": dict(target_g),
            "Frontier max": max_frontier,
            "Priority": "g(n)",
            "PQ rule": "one PQ for full plan; pop -> expand 4-neighbor -> record goal",
            "Last expanded": f"{explored[-1]}" if explored else "None",
            "Neighbors considered": f"{last_neighbors}",
            "Pushed to PQ": f"{last_pushed}",
        })
    return plan, set(explored), stats


def best_first_full_traversal_plan(algorithm, start, goals, blocked,
                                   neighbors, heuristic, counter,
                                   step_cost=unit_step_cost):
    goals = set(goals)
    open_set = []
    start_h = _nearest_goal_distance(start, goals, heuristic)
    start_priority = start_h
    heapq.heappush(open_set, (start_priority, 0, next(counter), start))
    best_cost = {start: 0}
    explored = []
    plan = []
    last_fgh = (start_priority, 0, start_h)

    while open_set and len(plan) < len(goals):
        _, cost, _, current = heapq.heappop(open_set)
        if cost > best_cost.get(current, INF):
            continue
        explored.append(current)
        if current in goals and current not in plan:
            plan.append(current)

        remaining_goals = goals - set(plan)
        for next_tile in neighbors(current, blocked):
            next_cost = cost + step_cost(current, next_tile)
            if next_cost >= best_cost.get(next_tile, INF):
                continue
            best_cost[next_tile] = next_cost
            h = _nearest_goal_distance(next_tile, remaining_goals, heuristic)
            if algorithm == "Greedy":
                priority = h
                heap_cost = next_cost
            else:
                priority = next_cost + h
                heap_cost = next_cost
            last_fgh = (priority, next_cost, h)
            heapq.heappush(
                open_set, (priority, heap_cost, next(counter), next_tile))

    stats = _full_traversal_stats(
        algorithm, plan, goals, explored,
        {"Frontier rule": "nearest remaining goal heuristic"})
    return plan, set(explored), last_fgh, stats


def build_full_garden_plan_by_algorithm(algorithm, start, goals, blocked,
                                        neighbors, heuristic, counter,
                                        step_cost=unit_step_cost):
    """Build one whole-garden traversal order and collect goals while exploring."""
    if algorithm == "BFS":
        plan, explored, stats = bfs_full_traversal_plan(
            start, goals, blocked, neighbors)
        return plan, explored, (0, 0, 0), stats
    if algorithm == "DFS":
        plan, explored, stats = dfs_full_traversal_plan(
            start, goals, blocked, neighbors)
        return plan, explored, (0, 0, 0), stats
    if algorithm == "IDS":
        plan, explored, stats = ids_full_traversal_plan(
            start, goals, blocked, neighbors)
        return plan, explored, (0, 0, 0), stats
    if algorithm == "UCS":
        plan, explored, stats = ucs_full_traversal_plan(
            start, goals, blocked, neighbors, counter, step_cost)
        return plan, explored, (0, 0, 0), stats

    plan, explored, fgh, stats = best_first_full_traversal_plan(
        algorithm, start, goals, blocked, neighbors, heuristic, counter,
        step_cost)
    return plan, explored, fgh, stats


def hill_score(tile, start, dryness, heuristic, turn_penalty=0):
    """Mode 3 local-search score.

    New Mode 3 does NOT rank by random dryness anymore.
    It ranks by tree condition first, then Manhattan distance:

        dry      = 40
        critical = 30
        pest     = 20
        dead     = 10

        score = type_score + Manhattan(start, tile) - turn_penalty

    The ``dryness`` argument is kept for compatibility with old calls, but it
    may now also be a tile -> condition map such as self.tile_conditions.
    """
    type_scores = {
        "dry": 40,
        "critical": 30,
        "pest": 20,
        "dead": 10,
    }
    value = dryness.get(tile, "dry")
    if isinstance(value, str):
        base_score = type_scores.get(value, 0)
    else:
        # Backward-compatible fallback for old callers that still pass a
        # numeric sensor map. Mode 3 controller should pass tile_conditions.
        base_score = float(value)

    score = base_score + heuristic(start, tile)
    if tile != start:
        score -= turn_penalty
    return score


def hill_climbing_choose(remaining, start, dryness, heuristic,
                         initial_tile=None):
    scores = {
        tile: hill_score(tile, start, dryness, heuristic)
        for tile in remaining
    }
    current_best = (
        initial_tile
        if initial_tile in remaining
        else min(remaining, key=lambda tile: heuristic(start, tile)))
    current_score = scores.get(current_best, 0)
    trace = [current_best]

    improved = True
    steps = 0
    while improved:
        improved = False
        cx, cy = current_best
        local_neighbors = [
            tile for tile in remaining
            if abs(tile[0] - cx) + abs(tile[1] - cy) == 1
            and tile != current_best
        ]
        lower_neighbors = [
            tile for tile in local_neighbors
            if scores.get(tile, 0) < current_score
        ]
        if not lower_neighbors:
            break

        # Test variant: downhill move.
        # lower_neighbors chi gom cac o co score nho hon current_score.
        # min o day chon o co score thap nhat trong cac lang gieng thap hon.
        best_neighbor = min(
            lower_neighbors,
            key=lambda tile: (scores.get(tile, 0), heuristic(start, tile), tile))
        best_neighbor_score = scores.get(best_neighbor, 0)
        current_best = best_neighbor
        current_score = best_neighbor_score
        trace.append(current_best)
        improved = True
        steps += 1

    return current_best, current_score, scores, steps, trace


def _task_action_time(tile, dryness):
    return 1.0 + max(0, min(dryness.get(tile, 50), 100)) / 100.0


def _task_urgency(tile, dryness):
    return 1.0 + max(0, min(dryness.get(tile, 50), 100)) / 50.0


def _plan_cost(plan, start, dryness, heuristic):
    current = start
    elapsed = 0.0
    weighted_completion = 0.0
    travel = 0
    for tile in plan:
        step_distance = heuristic(current, tile)
        travel += step_distance
        elapsed += step_distance + _task_action_time(tile, dryness)
        weighted_completion += elapsed * _task_urgency(tile, dryness)
        current = tile
    return weighted_completion, travel


def _plan_value(plan, start, dryness, heuristic):
    return -_plan_cost(plan, start, dryness, heuristic)[0]


def _plan_objective(plan, start, dryness, heuristic, scores):
    # Downhill variant: lower score = drier = higher urgency.
    # Negate so that minimising plan cost also puts the lowest-score tile first.
    if not plan:
        return float("inf")
    first_target_score = scores.get(plan[0], 0)
    # Subtract first_target_score so lower score gives a lower (better) objective.
    return first_target_score * 1000 - _plan_value(
        plan, start, dryness, heuristic)


def _nearest_neighbor_plan(remaining, start, heuristic):
    unvisited = list(remaining)
    current = start
    plan = []
    while unvisited:
        next_tile = min(unvisited, key=lambda tile: (heuristic(current, tile), tile))
        plan.append(next_tile)
        unvisited.remove(next_tile)
        current = next_tile
    return plan


def _plan_successors(plan):
    if len(plan) <= 1:
        return []
    successors = []
    for index in range(len(plan) - 1):
        child = list(plan)
        child[index], child[index + 1] = child[index + 1], child[index]
        successors.append(child)
    for index in range(len(plan)):
        child = list(plan)
        tile = child.pop(index)
        child.insert(0, tile)
        successors.append(child)
    return successors


def _plan_scores(remaining, start, dryness, heuristic):
    return {
        tile: hill_score(tile, start, dryness, heuristic)
        for tile in remaining
    }


def _plan_result_stats(algorithm, plan, start, dryness, heuristic, scores,
                       extra=None):
    target = plan[0] if plan else None
    target_score = scores.get(target, 0) if target is not None else 0
    plan_cost, travel = _plan_cost(plan, start, dryness, heuristic)
    stats = {
        "Local algorithm": algorithm,
        "Current score": f"{target_score:.0f}",
        "Target": f"{target}",
        "Dryness": f"{dryness.get(target, 0)}%" if target is not None else "0%",
        "Plan objective": (
            f"{_plan_objective(plan, start, dryness, heuristic, scores):.1f}"),
        "Plan value": f"{_plan_value(plan, start, dryness, heuristic):.1f}",
        "Plan cost": f"{plan_cost:.1f}",
        "Plan travel": travel,
    }
    if extra:
        stats.update(extra)
    return target, target_score, scores, stats


def local_beam_choose(remaining, start, dryness, heuristic,
                      beam_width=3, max_steps=20):
    scores = _plan_scores(remaining, start, dryness, heuristic)
    if not remaining:
        return None, 0, scores, {
            "Local algorithm": "Local Beam",
            "Target": "None",
        }

    remaining_list = list(remaining)
    beam_size = max(1, min(beam_width, len(remaining_list)))
    starts = [
        _nearest_neighbor_plan(remaining_list, start, heuristic),
        sorted(
            remaining_list,
            key=lambda tile: (dryness.get(tile, 50),   # ascending: lowest first
                              heuristic(start, tile), tile)),
    ]
    while len(starts) < beam_size:
        plan = list(remaining_list)
        random.shuffle(plan)
        starts.append(plan)

    beam = starts[:beam_size]
    expansions = 0

    for _ in range(max_steps):
        children = []
        for plan in beam:
            children.extend(_plan_successors(plan))
        if not children:
            break

        expansions += len(children)
        unique_children = {
            tuple(child): child
            for child in children
        }
        next_beam = sorted(
            unique_children.values(),
            key=lambda plan: (_plan_objective(
                                  plan, start, dryness, heuristic, scores),
                              tuple(plan)))[:beam_size]
        if {tuple(plan) for plan in next_beam} == {tuple(plan) for plan in beam}:
            break
        beam = next_beam

    best_plan = min(
        beam,
        key=lambda plan: (_plan_objective(
                              plan, start, dryness, heuristic, scores),
                          tuple(plan)))
    return _plan_result_stats(
        "Local Beam", best_plan, start, dryness, heuristic, scores,
        {"Beam width": beam_width, "Generated": expansions})


def simulated_annealing_choose(remaining, start, dryness, heuristic,
                               temperature=80.0, cooling=0.95,
                               min_temperature=0.1, max_steps=140):
    scores = _plan_scores(remaining, start, dryness, heuristic)
    if not remaining:
        return None, 0, scores, {
            "Local algorithm": "Annealing",
            "Target": "None",
        }

    def cost(plan):
        return -_plan_objective(plan, start, dryness, heuristic, scores)

    current = list(remaining)
    random.shuffle(current)
    current_cost = cost(current)
    best = list(current)
    best_cost = current_cost
    accepted_worse = 0
    steps = 0

    while temperature > min_temperature and steps < max_steps:
        if len(current) <= 1:
            break

        candidate = list(current)
        left, right = random.sample(range(len(candidate)), 2)
        candidate[left], candidate[right] = candidate[right], candidate[left]
        candidate_cost = cost(candidate)
        delta = candidate_cost - current_cost

        accept = False
        if delta < 0:
            accept = True
        elif random.random() < math.exp(-delta / temperature):
            accept = True

        if accept:
            if delta > 0:
                accepted_worse += 1
            current = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best = list(current)
                best_cost = current_cost

        temperature *= cooling
        steps += 1

    return _plan_result_stats(
        "Annealing", best, start, dryness, heuristic, scores,
        {"Anneal steps": steps, "Accepted worse": accepted_worse})


def build_local_search_full_plan(algorithm, remaining, start, dryness,
                                 heuristic):
    """Build a full plan with the current local-neighbor Mode 3 semantics."""
    remaining_set = set(remaining)
    scores = _plan_scores(remaining_set, start, dryness, heuristic)
    if not remaining_set:
        return [], scores, {
            "Local algorithm": algorithm,
            "Planning mode": "Full garden local-neighbor search",
            "Plan targets": 0,
        }

    current = start
    plan = []
    local_moves = 0
    restarts = 0
    generated = 0
    accepted_worse = 0
    temperature = 80.0
    stop_rule = "All targets planned"

    def adjacent_targets(center):
        return [
            tile for tile in remaining_set
            if heuristic(center, tile) == 1
        ]

    def random_restart_target():
        return random.choice(sorted(remaining_set))

    while remaining_set:
        neighbors = adjacent_targets(current)
        if not neighbors:
            if algorithm in ("Hill Climbing", "Annealing", "Local Beam"):
                stop_rule = "No adjacent target"
                break
            target = random_restart_target()
            if plan:
                restarts += 1
        elif algorithm == "Hill Climbing":
            current_score = (
                scores.get(current, hill_score(current, current, dryness, heuristic))
                if current in dryness else scores.get(neighbors[0], 0)
            )
            better = [
                tile for tile in neighbors
                if scores.get(tile, 0) < current_score
            ]
            if not better:
                stop_rule = "No lower adjacent target"
                break
            target = min(
                better,
                key=lambda tile: (scores.get(tile, 0), heuristic(current, tile), tile))
            local_moves += 1
        elif algorithm == "Restart Hill":
            target = min(
                neighbors,
                key=lambda tile: (scores.get(tile, 0), heuristic(current, tile), tile))
            local_moves += 1
        elif algorithm == "Local Beam":
            beam_width = 3
            beam = sorted(
                neighbors,
                key=lambda tile: (scores.get(tile, 0), tile)
            )[:beam_width]
            candidates = set(beam)
            for tile in beam:
                candidates.update(adjacent_targets(tile))
            generated += len(candidates)
            target = min(
                candidates,
                key=lambda tile: (scores.get(tile, 0), heuristic(current, tile), tile))
            if target not in neighbors:
                target = min(
                    neighbors,
                    key=lambda tile: (
                        heuristic(tile, target),
                        scores.get(tile, 0),
                        tile))
            local_moves += 1
        elif algorithm == "Annealing":
            current_score = (
                scores.get(current, hill_score(current, current, dryness, heuristic))
                if current in dryness else scores.get(neighbors[0], 0)
            )
            ordered = list(neighbors)
            random.shuffle(ordered)
            target = ordered[0]
            for candidate in ordered:
                delta = scores.get(candidate, 0) - current_score
                if delta <= 0:
                    target = candidate
                    break
                probability = math.exp(-delta / max(temperature, 0.001))
                if random.random() < probability:
                    target = candidate
                    accepted_worse += 1
                    break
            temperature = max(0.1, temperature * 0.90)
            local_moves += 1
        else:
            target = min(
                neighbors,
                key=lambda tile: (scores.get(tile, 0), heuristic(current, tile), tile))
            local_moves += 1

        plan.append(target)
        remaining_set.remove(target)
        current = target

    extra = {
        "Local rule": "Adjacent targets first",
        "Local moves": local_moves,
        "Restarts": restarts,
        "Stop rule": stop_rule,
    }
    if algorithm == "Local Beam":
        extra.update({"Beam width": 3, "Generated": generated})
    if algorithm == "Annealing":
        extra.update({
            "Anneal temp": f"{temperature:.1f}",
            "Accepted worse": accepted_worse,
        })
    if algorithm == "Restart Hill":
        extra["Restart rule"] = "Random remaining target"

    _, target_score, _, stats = _plan_result_stats(
        algorithm, plan, start, dryness, heuristic, scores, extra)
    stats["Planning mode"] = "Full garden local-neighbor search"
    stats["Plan targets"] = len(plan)
    stats["Current score"] = f"{target_score:.0f}"
    return plan, scores, stats


def local_search_choose(algorithm, remaining, start, dryness, heuristic):
    scores = _plan_scores(remaining, start, dryness, heuristic)
    if not remaining:
        return None, 0, scores, {
            "Local algorithm": algorithm,
            "Target": "None",
        }

    if algorithm == "Local Beam":
        return local_beam_choose(remaining, start, dryness, heuristic)
    if algorithm == "Hill Climbing":
        current, current_score, scores, hill_steps, trace = hill_climbing_choose(
            remaining, start, dryness, heuristic)
        return current, current_score, scores, {
            "Local algorithm": "Hill Climbing",
            "Current score": f"{current_score:.0f}",
            "Target": f"{current}",
            "Target type": f"{dryness.get(current, 'unknown')}",
            "Score formula": "type + Manhattan",
            "Hill steps": hill_steps,
            "Trace": trace,
            "Stop rule": "No lower neighbor",
        }
    if algorithm == "Annealing":
        return simulated_annealing_choose(remaining, start, dryness, heuristic)
    if algorithm == "Restart Hill":
        best_tile = None
        best_score = float("inf")   # downhill: want the LOWEST score found
        best_trace = []
        starts = random.sample(list(remaining), min(len(remaining), 8))
        for restart_tile in starts:
            candidate, candidate_score, _, _, trace = hill_climbing_choose(
                remaining, start, dryness, heuristic,
                initial_tile=restart_tile)
            if candidate_score < best_score:  # lower = drier = higher priority
                best_tile = candidate
                best_score = candidate_score
                best_trace = trace
        return best_tile, best_score, scores, {
            "Local algorithm": "Restart Hill",
            "Current score": f"{best_score:.0f}",
            "Target": f"{best_tile}",
            "Dryness": f"{dryness.get(best_tile, 0)}%",
            "Restarts": len(starts),
            "Trace": best_trace,
            "Stop rule": "No lower neighbor",
        }

    current, current_score, scores, hill_steps, trace = hill_climbing_choose(
        remaining, start, dryness, heuristic)
    return current, current_score, scores, {
        "Local algorithm": algorithm,
        "Current score": f"{current_score:.0f}",
        "Target": f"{current}",
        "Target type": f"{dryness.get(current, 'unknown')}",
            "Score formula": "type + Manhattan",
        "Hill steps": hill_steps,
        "Trace": trace,
    }


# Mode 4: Online and belief-state search
# ---------------------------------------------------------------------------

def has_line_of_sight(center, tile, blockers):
    cx, cy = center
    tx, ty = tile
    dx = tx - cx
    dy = ty - cy
    steps = math.gcd(abs(dx), abs(dy))
    if steps == 0:
        return True

    step_x = dx // steps
    step_y = dy // steps
    x, y = cx, cy
    for _ in range(steps - 1):
        x += step_x
        y += step_y
        if (x, y) in blockers:
            return False
    return True


def expand_vision(center, radius, rows, cols, hidden_blocked,
                  explored_tiles, discovered_blocked):
    cx, cy = center
    scan_radius = int(radius) + 1
    blockers = set(hidden_blocked) | set(discovered_blocked)
    for dy in range(-scan_radius, scan_radius + 1):
        for dx in range(-scan_radius, scan_radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue

            tile = (nx, ny)
            if not has_line_of_sight(center, tile, blockers):
                continue

            explored_tiles.add(tile)
            if tile in hidden_blocked:
                discovered_blocked.add(tile)


def belief_state_bfs(start, goal, blocked, belief_worlds, belief_unknowns,
                     neighbors):
    """BFS over (robot_tile, possible_world_ids) belief states.

    The robot has one real position, but the map can still be one of many
    hidden-block worlds. Moving onto a tile is planned as a successful
    observation: worlds where that tile was blocked are filtered out.
    """
    worlds = list(belief_worlds or [])
    if not worlds:
        return [], set(), {
            "Algorithm": "Belief-State BFS",
            "Search style": "BFS over belief states",
            "Belief node": "(tile, possible_worlds)",
            "Belief states": 0,
            "Possible worlds": 0,
            "Belief unknowns": len(belief_unknowns),
            "Path length": 0,
            "Queue max": 0,
            "Result": "Inconsistent belief: no possible worlds remain",
        }
    initial_world_ids = frozenset(range(len(worlds)))
    start_state = (start, initial_world_ids)
    queue = deque([start_state])
    came_from = {}
    visited = {start_state}
    explored_states = set()
    explored_tiles = set()
    max_queue = 1

    while queue:
        current_state = queue.popleft()
        current_tile, world_ids = current_state
        explored_states.add(current_state)
        explored_tiles.add(current_tile)

        if current_tile == goal:
            state_path = reconstruct_path(came_from, current_state)
            path = [tile for tile, _world_ids in state_path]
            risky_steps = sum(tile in belief_unknowns for tile in path)
            return path, explored_tiles, {
                "Algorithm": "Belief-State BFS",
                "Search style": "BFS over belief states",
                "Belief node": "(tile, possible_worlds)",
                "Belief states": len(explored_states),
                "Possible worlds": len(worlds),
                "Belief unknowns": len(belief_unknowns),
                "Risky steps": risky_steps,
                "Path length": len(path),
                "Queue max": max_queue,
                "Policy": "FIFO belief frontier; replan after observation",
            }

        for next_tile in neighbors(current_tile, blocked):
            if next_tile in blocked:
                continue
            next_world_ids = frozenset(
                world_id for world_id in world_ids
                if next_tile not in worlds[world_id]
            )
            if not next_world_ids:
                continue
            next_state = (next_tile, next_world_ids)
            if next_state in visited:
                continue
            visited.add(next_state)
            came_from[next_state] = current_state
            queue.append(next_state)
            max_queue = max(max_queue, len(queue))

    return [], explored_tiles, {
        "Algorithm": "Belief-State BFS",
        "Search style": "BFS over belief states",
        "Belief node": "(tile, possible_worlds)",
        "Belief states": len(explored_states),
        "Possible worlds": len(worlds),
        "Belief unknowns": len(belief_unknowns),
        "Risky steps": 0,
        "Path length": 0,
        "Queue max": max_queue,
        "Policy": "No belief-safe BFS path found",
    }


def generate_belief_worlds(unknown_tiles, max_hidden=2):
    """Build all possible hidden-block configurations over tracked tiles."""
    tiles = list(unknown_tiles)
    worlds = []
    max_count = len(tiles) if max_hidden is None else min(max_hidden, len(tiles))
    for rock_count in range(max_count + 1):
        for blocked in combinations(tiles, rock_count):
            worlds.append(frozenset(blocked))
    return worlds


def update_belief_worlds(worlds, observed_free=None, observed_blocked=None):
    """Filter worlds against newly observed free/blocked tiles."""
    observed_free = set(observed_free or ())
    observed_blocked = set(observed_blocked or ())
    if not worlds:
        return []

    return [
        world for world in worlds
        if world.isdisjoint(observed_free)
        and observed_blocked.issubset(world)
    ]


def belief_risk_map(worlds, candidate_tiles=None):
    """Return P(tile is blocked) estimated from current possible worlds."""
    if candidate_tiles is None:
        candidate_tiles = set()
        for world in worlds:
            candidate_tiles.update(world)
    else:
        candidate_tiles = set(candidate_tiles)

    if not worlds:
        return {tile: 0.0 for tile in candidate_tiles}

    total_worlds = float(len(worlds))
    return {
        tile: sum(tile in world for world in worlds) / total_worlds
        for tile in candidate_tiles
    }


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Mode 5: Constraint satisfaction
# ---------------------------------------------------------------------------


def _csp_adjacent(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def solve_csp_crop_plan(farm_tiles, algorithm="Backtrack"):
    """Assign crops using the selected CSP solving strategy."""
    variables = sorted(farm_tiles, key=lambda tile: (tile[1], tile[0]))

    steps = []
    backtracks = 0
    arc_checks = 0
    show_domain_steps = algorithm in ("Fwd Check", "AC-3")
    adjacency = {
        var: [other for other in variables if _csp_adjacent(var, other)]
        for var in variables
    }
    domain = CSP_CROPS

    def log(event, var, value):
        if len(steps) < 2000:
            steps.append((event, var, value))

    def log_domain(var, domains):
        if not show_domain_steps:
            return
        log("domain", var, tuple(domains[var]))

    def log_domains(domains):
        for var in variables:
            log_domain(var, domains)

    def crop_pair_valid(left, right):
        # Ràng buộc: hai ô kề nhau không được trồng cùng loại cây
        return csp_crop_pair_valid(left, right)

    def valid(var, value, assignment):
        return all(
            crop_pair_valid(value, assignment[neighbor])
            for neighbor in adjacency[var]
            if neighbor in assignment
        )

    def select_unassigned_row_major(assignment):
        for var in variables:
            if var not in assignment:
                return var
        return None

    def select_unassigned_mrv_degree(assignment, domains):
        remaining = [var for var in variables if var not in assignment]
        return min(
            remaining,
            key=lambda var: (len(domains[var]), -len(adjacency[var])),
        )

    def _prune_domain(domains, var, keep_values):
        kept = [value for value in domains[var] if value in keep_values]
        if len(kept) == len(domains[var]):
            return True
        domains[var] = kept
        log_domain(var, domains)
        return bool(domains[var])

    def backtrack(
            assignment, domains, forward_check=False,
            arc_consistency=False, use_heuristic=False):
        nonlocal backtracks
        if len(assignment) == len(variables):
            return dict(assignment)

        if use_heuristic:
            var = select_unassigned_mrv_degree(assignment, domains)
        else:
            var = select_unassigned_row_major(assignment)
        if var is None:
            return dict(assignment)

        candidate_values = list(domains[var])
        for value in candidate_values:
            # The UI shows the solver considering several crops before it
            # commits to the candidate. FC/AC-3 only consider the current
            # pruned domain, so removed crops are never "thought about" again.
            previews = list(domains[var])
            for preview in previews[:3]:
                log("think", var, preview)
            log("try", var, value)
            if value not in domains[var] or not valid(var, value, assignment):
                backtracks += 1
                log("backtrack", var, value)
                continue

            next_assignment = dict(assignment)
            next_assignment[var] = value
            next_domains = {
                key: list(values) for key, values in domains.items()
            }
            log("assign", var, value)
            next_domains[var] = [value]
            log_domain(var, next_domains)

            consistent = True
            if forward_check:
                for neighbor in adjacency[var]:
                    if neighbor in next_assignment:
                        continue
                    allowed = [
                        crop for crop in next_domains[neighbor]
                        if crop_pair_valid(crop, value)
                    ]
                    if not _prune_domain(next_domains, neighbor, allowed):
                        consistent = False
                        break
            if consistent and arc_consistency:
                consistent = enforce_arc_consistency(next_domains)

            if consistent:
                result = backtrack(
                    next_assignment,
                    next_domains,
                    forward_check=forward_check,
                    arc_consistency=arc_consistency,
                    use_heuristic=use_heuristic,
                )
                if result is not None:
                    return result

            backtracks += 1
            log("backtrack", var, value)
            log_domains(domains)
        return None

    def enforce_arc_consistency(domains):
        nonlocal arc_checks
        queue = deque(
            (left, right)
            for left in variables
            for right in adjacency[left]
        )
        while queue:
            left, right = queue.popleft()
            arc_checks += 1
            supported = [
                left_value
                for left_value in domains[left]
                if any(
                    crop_pair_valid(left_value, right_value)
                    for right_value in domains[right]
                )
            ]
            if len(supported) == len(domains[left]):
                continue
            if not _prune_domain(domains, left, supported):
                return False
            for neighbor in adjacency[left]:
                if neighbor != right:
                    queue.append((neighbor, left))
        return True

    def min_conflicts(max_steps=1000, restarts=20):
        """Min-Conflicts thuần theo AIMA Figure 6.8.

        1. Khởi tạo MỘT assignment ĐẦY ĐỦ cho tất cả biến (có thể vi phạm
           ràng buộc) bằng giá trị ngẫu nhiên trong domain.
        2. Mỗi bước: chọn NGẪU NHIÊN một biến đang xung đột (vi phạm ít
           nhất 1 ràng buộc), rồi gán cho nó giá trị làm xung đột TOÀN CỤC
           thấp nhất (min-conflicts value, ties random).
        3. Lặp tới khi không còn xung đột (solution) hoặc hết max_steps.

        Không dùng AC-3/forward-check: đây là local search trên không gian
        assignment đầy đủ, đúng bản chất khác biệt với Backtrack/FC/AC-3.
        """
        nonlocal backtracks
        if not variables:
            return {}

        def conflicts_of_var(v, value, assignment):
            return sum(
                1
                for nb in adjacency[v]
                if nb in assignment and not crop_pair_valid(value, assignment[nb])
            )

        def conflicted_vars(assignment):
            return [
                v for v in variables
                if conflicts_of_var(v, assignment[v], assignment) > 0
            ]

        def initial_complete_assignment():
            assignment = {
                v: _checker_crop(v[0], v[1])
                for v in variables
            }
            noisy_tiles = list(variables)
            random.shuffle(noisy_tiles)
            noise_count = max(1, min(4, len(variables) // 4))
            for v in noisy_tiles[:noise_count]:
                conflicting_neighbors = [nb for nb in adjacency[v] if nb in assignment]
                if conflicting_neighbors:
                    assignment[v] = assignment[random.choice(conflicting_neighbors)]
                else:
                    assignment[v] = random.choice(domain)
            return assignment

        for _restart in range(max(1, restarts)):
            # Bước 1: gán đầy đủ ngẫu nhiên (random initial complete assignment).
            assignment = initial_complete_assignment()
            log("init", None, dict(assignment))  # snapshot toan bo de sync csp_assigned

            for step in range(max_steps):
                conflicted = conflicted_vars(assignment)
                if not conflicted:
                    return assignment  # hết xung đột -> solution

                # Bước 2: chọn ngẫu nhiên 1 biến đang xung đột.
                worst_conflict = max(
                    conflicts_of_var(v, assignment[v], assignment)
                    for v in conflicted
                )
                worst_vars = [
                    v for v in conflicted
                    if conflicts_of_var(v, assignment[v], assignment) == worst_conflict
                ]
                var = random.choice(worst_vars)

                # Bước 3: chọn giá trị làm số xung đột của var thấp nhất.
                old_value = assignment[var]
                best_c = INF
                best_vals = []
                for value in domains[var]:
                    c = conflicts_of_var(var, value, assignment)
                    if c < best_c:
                        best_c = c
                        best_vals = [value]
                    elif c == best_c:
                        best_vals.append(value)

                alternatives = [v for v in best_vals if v != old_value]
                if alternatives:
                    value = random.choice(alternatives)
                elif conflicts_of_var(var, old_value, assignment) == 0:
                    value = old_value
                else:
                    non_old = [v for v in domains[var] if v != old_value]
                    value = min(
                        non_old,
                        key=lambda v: conflicts_of_var(var, v, assignment),
                    )
                # Log "conflict" TRUOC khi doi gia tri: danh dau o nay (voi
                # crop CU dang vi pham rang buoc) de UI to do va cho robot
                # di toi truoc khi sua. Sau do "reassign" voi gia tri MOI
                # se bao UI chuyen o nay sang xanh.
                log("conflict", var, old_value)
                assignment[var] = value
                log("reassign", var, value)
                backtracks += 1  # đếm mỗi lần sửa xung đột như 1 "step cost"

            # Hết max_steps mà chưa xong -> restart với assignment ngẫu nhiên khác.

        return None


    # Randomize each tile's value order so valid plans do not collapse into
    # the same two-crop checkerboard every replay.
    crop_list = list(CSP_CROPS)

    def _checker_crop(x, y):
        # (0,0)->corn, (1,0)->tomato, (0,1)->wheat, (1,1)->carrot
        idx = (x % 2) + (y % 2) * 2
        return crop_list[idx % 4]

    if variables and algorithm == "Min Conflict":
        def _preferred_domain(var):
            preferred = _checker_crop(var[0], var[1])
            rest = [c for c in domain if c != preferred]
            random.shuffle(rest)
            return [preferred] + rest

        domains = {var: _preferred_domain(var) for var in variables}
    else:
        domains = {}
        for var in variables:
            values = list(CSP_CROPS)
            random.shuffle(values)
            domains[var] = values


    if algorithm == "Fwd Check":
        result = backtrack({}, domains, forward_check=True, use_heuristic=True)
    elif algorithm == "AC-3":
        result = (
            backtrack(
                {}, domains, forward_check=True,
                arc_consistency=True, use_heuristic=True)
            if enforce_arc_consistency(domains)
            else None
        )
    elif algorithm == "Min Conflict":
        result = min_conflicts()
        if result is None:
            # Chỉ fallback khi Min-Conflicts thực sự THẤT BẠI (hết max_steps/restarts
            # mà vẫn còn xung đột), không phải vì nghiệm "ít màu" — một nghiệm chỉ
            # dùng 2/4 loại crop vẫn hợp lệ với ràng buộc "khác màu kề nhau".
            result = backtrack(
                {}, domains, forward_check=True, use_heuristic=True)

    else:
        result = backtrack({}, domains, use_heuristic=False)


    # Không post-process thay đổi assignment sau khi đã tìm nghiệm.
    # (Giữ replay CSP khớp với assignment cuối và tránh việc robot “quay lại ô cũ”).


    stats = {
        "Algorithm": algorithm,
        "Variables": len(variables),
        "Domain": "{corn, tomato, wheat, carrot}",
        "Value order": "random per tile",
        "Constraints": (
            "no same crop; corn-tomato forbidden; "
            "wheat-carrot forbidden"),
        "Backtracks": backtracks,
    }
    if algorithm == "AC-3":
        stats["Arc checks"] = arc_checks
    return result or {}, steps, stats


MODE6_SEARCH_DEPTH = 6
MODE6_TREE_STATUS = {
    "dry": {"value": 20, "risk": 0.20},
    "dead": {"value": 10, "risk": 0.30},
    "pest": {"value": 30, "risk": 0.40},
    "critical": {"value": 50, "risk": 0.70},
    "golden": {"value": 100, "risk": 0.90},
}
MODE6_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT", "STAY")
MODE6_ACTION_DELTAS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "STAY": (0, 0),
}
MODE6_CHANCE_LABELS = {
    "no_damage": "An toan",
    "lightning_rod": "Cot thu loi hut set",
    "weather_damage": "Set danh",
}
MODE6_LIGHTNING_ROD_BONUS = 15.0


def _crop_value(tile, crop_profiles):
    return crop_profiles.get(tile, MODE6_TREE_STATUS["dry"])["value"]


def _crop_risk(tile, crop_profiles):
    return crop_profiles.get(tile, MODE6_TREE_STATUS["dry"])["risk"]


def mode6_chance_outcomes(tile, crop_profiles=None):
    return (
        ("no_damage", 0.50),
        ("weather_damage", 0.20),
        ("lightning_rod", 0.30),
    )


def mode6_chance_tile_weights(remaining, crop_profiles=None):
    crop_profiles = crop_profiles or {}
    tiles = tuple(remaining)
    weights = tuple(max(0.0, _crop_risk(tile, crop_profiles)) for tile in tiles)
    if tiles and sum(weights) <= 0:
        weights = tuple(1.0 for _ in tiles)
    return tiles, weights


def _mode6_select_chance_tile(remaining, crop_profiles, preferred=None):
    remaining = set(remaining)
    if preferred in remaining:
        return preferred
    if not remaining:
        return None
    return max(
        remaining,
        key=lambda tile: (
            _crop_value(tile, crop_profiles) * (1.0 + _crop_risk(
                tile, crop_profiles)),
            _crop_risk(tile, crop_profiles),
            _crop_value(tile, crop_profiles),
            tile,
        ),
    )


def mode6_state_scores(protected, destroyed, crop_profiles=None):
    crop_profiles = crop_profiles or {}
    protected_score = sum(_crop_value(tile, crop_profiles) for tile in protected)
    destroyed_score = sum(_crop_value(tile, crop_profiles) for tile in destroyed)
    return protected_score, destroyed_score


def evaluate_farm_state(all_crops, protected, destroyed, crop_profiles=None):
    protected_score, destroyed_score = mode6_state_scores(
        protected, destroyed, crop_profiles)
    return protected_score - destroyed_score


def _mode6_manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _mode6_pressure(pos, remaining, crop_profiles):
    if not remaining:
        return 0.0
    return max(
        _crop_value(tile, crop_profiles) / (_mode6_manhattan(pos, tile) + 1)
        for tile in remaining)


def _mode6_nearest_distance(pos, remaining):
    if not remaining:
        return 0
    return min(_mode6_manhattan(pos, tile) for tile in remaining)


def _mode6_positional_evaluation(robot_pos, enemy_pos, all_crops, protected,
                                 destroyed, crop_profiles,
                                 enemy_threat_tile=None,
                                 enemy_threat_turns=0,
                                 robot_repair_tile=None,
                                 robot_repair_turns=0,
                                 enemy_active=True):
    score = evaluate_farm_state(all_crops, protected, destroyed, crop_profiles)
    remaining = _mode6_remaining(all_crops, protected, destroyed)
    if not remaining:
        return score

    robot_potential = _mode6_pressure(robot_pos, remaining, crop_profiles)
    enemy_threat = (
        _mode6_pressure(enemy_pos, remaining, crop_profiles)
        if enemy_active else 0.0)
    score += 0.2 * robot_potential - 0.2 * enemy_threat

    if enemy_active and enemy_threat_tile in remaining:
        threat_value = _crop_value(enemy_threat_tile, crop_profiles)
        robot_distance = _mode6_manhattan(robot_pos, enemy_threat_tile)
        enemy_distance = _mode6_manhattan(enemy_pos, enemy_threat_tile)
        urgency = 1.0 + 0.8 * max(0, enemy_threat_turns)

        if robot_pos == enemy_threat_tile:
            score += 4.0 * threat_value * urgency
        else:
            score += (
                max(0, 5 - robot_distance)
                * 0.9 * threat_value * urgency)

        if enemy_pos == enemy_threat_tile:
            score -= 3.0 * threat_value * urgency
        else:
            score -= (
                max(0, 3 - enemy_distance)
                * 0.7 * threat_value * urgency)

    if robot_repair_tile in remaining:
        repair_value = _crop_value(robot_repair_tile, crop_profiles)
        if robot_pos == robot_repair_tile:
            score += repair_value * (1.0 + robot_repair_turns)

    return score


def _mode6_move(pos, action):
    dx, dy = MODE6_ACTION_DELTAS[action]
    return pos[0] + dx, pos[1] + dy


def mode6_legal_actions(actor_tile, remaining_set=None, blocked=None,
                        neighbors=None):
    blocked = set(blocked or ())
    walkable = None if callable(neighbors) or neighbors is None else neighbors
    actions = []
    for action in MODE6_ACTIONS:
        next_tile = _mode6_move(actor_tile, action)
        if next_tile in blocked:
            continue
        if action == "STAY":
            actions.append(action)
            continue
        if callable(neighbors):
            if next_tile in neighbors(actor_tile, blocked):
                actions.append(action)
        elif neighbors is None:
            continue
        elif next_tile in walkable:
            actions.append(action)
    return actions


def _mode6_walkable_actions(pos, walkable_tiles, action_cache=None):
    if action_cache is not None and pos in action_cache:
        return list(action_cache[pos])

    actions = mode6_legal_actions(pos, blocked=set(), neighbors=walkable_tiles)
    if action_cache is not None:
        action_cache[pos] = tuple(actions)
    return actions


def _mode6_legal_actions(pos, walkable_tiles):
    return _mode6_walkable_actions(pos, walkable_tiles)


def _mode6_enemy_legal_actions(enemy_pos, remaining, all_crops, protected,
                               destroyed, crop_profiles, walkable_tiles,
                               enemy_threat_tile=None,
                               enemy_threat_turns=0, action_cache=None):
    actions = _mode6_walkable_actions(
        enemy_pos, walkable_tiles, action_cache)
    if not actions:
        return [mode6_enemy_greedy_action(
            enemy_pos, all_crops, protected, destroyed, crop_profiles,
            walkable_tiles, enemy_threat_tile, enemy_threat_turns)]

    return sorted(
        actions,
        key=lambda action: (
            -_mode6_action_score(
                enemy_pos, action, remaining, crop_profiles,
                priority_tile=enemy_threat_tile),
            MODE6_ACTIONS.index(action),
        ))


def _mode6_enemy_legal_frontier(enemy_pos, remaining, all_crops, protected,
                                destroyed, crop_profiles, walkable_tiles,
                                enemy_threat_tile=None,
                                enemy_threat_turns=0):
    actions = _mode6_walkable_actions(enemy_pos, walkable_tiles)
    if actions:
        return actions
    return [mode6_enemy_greedy_action(
        enemy_pos, all_crops, protected, destroyed, crop_profiles,
        walkable_tiles, enemy_threat_tile, enemy_threat_turns)]


def _mode6_action_score(pos, action, remaining, crop_profiles, avoid_pos=None,
                        resolved=None, priority_tile=None):
    next_pos = _mode6_move(pos, action)
    if priority_tile in remaining:
        priority_value = _crop_value(priority_tile, crop_profiles)
        priority_score = (
            max(0, 6 - _mode6_manhattan(next_pos, priority_tile))
            * priority_value)
        if next_pos == priority_tile:
            priority_score += 5.0 * priority_value
    else:
        priority_score = 0.0
    if next_pos in remaining:
        return 2.0 * _crop_value(next_pos, crop_profiles) + priority_score
    if action == "STAY":
        return -5.0 + priority_score
    if not remaining:
        return 0.0
    score = (
        2.0 * _mode6_pressure(next_pos, remaining, crop_profiles)
        - _mode6_nearest_distance(next_pos, remaining)
        + priority_score
    )
    if avoid_pos is not None and next_pos == avoid_pos:
        score -= 5.0
    if resolved and next_pos in resolved:
        score -= 5.0
    return score


def _mode6_rank_actions(pos, walkable_tiles, remaining, crop_profiles,
                        avoid_pos=None, resolved=None, priority_tile=None,
                        action_cache=None):
    actions = _mode6_walkable_actions(pos, walkable_tiles, action_cache)
    can_interact_here = pos in remaining or pos == priority_tile
    moving_actions = [action for action in actions if action != "STAY"]
    if moving_actions and not can_interact_here:
        actions = moving_actions

    return sorted(
        actions,
        key=lambda action: (
            -_mode6_action_score(
                pos, action, remaining, crop_profiles, avoid_pos, resolved,
                priority_tile),
            MODE6_ACTIONS.index(action),
        ),
    )

def _mode6_enemy_greedy_choice(enemy_pos, farm_tiles, protected, destroyed,
                               crop_profiles=None, walkable_tiles=None,
                               enemy_threat_tile=None,
                               enemy_threat_turns=0):
    crop_profiles = crop_profiles or {}
    remaining = _mode6_remaining(farm_tiles, protected, destroyed)
    walkable_tiles = set(walkable_tiles or farm_tiles)
    walkable_tiles.add(enemy_pos)
    if not remaining:
        return "STAY", None, 0.0

    if enemy_pos in remaining:
        profile = crop_profiles.get(enemy_pos, {})
        value = profile.get("value", MODE6_TREE_STATUS["dry"]["value"])
        risk = profile.get("risk", 0.0)
        return "STAY", enemy_pos, value * 3 + risk * 100

    priority_queue = []
    for tile in remaining:
        profile = crop_profiles.get(tile, {})
        value = profile.get("value", MODE6_TREE_STATUS["dry"]["value"])
        risk = profile.get("risk", 0.0)
        distance = _mode6_manhattan(enemy_pos, tile)
        score = value * 3 + risk * 100 - distance * 5
        heapq.heappush(priority_queue, (-score, distance, tile))

    while priority_queue:
        negative_score, _, target = heapq.heappop(priority_queue)
        queue = deque([(enemy_pos, None)])
        visited = {enemy_pos}
        while queue:
            current, first_action = queue.popleft()
            for action in ("UP", "DOWN", "LEFT", "RIGHT"):
                next_tile = _mode6_move(current, action)
                if next_tile in visited or next_tile not in walkable_tiles:
                    continue
                next_first = first_action or action
                if next_tile == target:
                    return next_first, target, -negative_score
                visited.add(next_tile)
                queue.append((next_tile, next_first))

    return "STAY", None, 0.0


def mode6_enemy_greedy_action(enemy_pos, farm_tiles, protected, destroyed,
                              crop_profiles=None, walkable_tiles=None,
                              enemy_threat_tile=None,
                              enemy_threat_turns=0):
    action, _, _ = _mode6_enemy_greedy_choice(
        enemy_pos, farm_tiles, protected, destroyed, crop_profiles,
        walkable_tiles, enemy_threat_tile, enemy_threat_turns)
    return action


def _mode6_remaining(all_crops, protected, destroyed):
    if isinstance(all_crops, frozenset):
        return set(all_crops.difference(protected, destroyed))
    return set(all_crops) - set(protected) - set(destroyed)


def _mode6_no_progress(robot_pos, enemy_pos, next_robot, next_enemy,
                       protected, destroyed, next_protected,
                       next_destroyed, enemy_threat_tile, next_threat_tile,
                       robot_repair_tile, robot_repair_turns,
                       next_repair_tile, next_repair_turns):
    return (
        len(next_protected) == len(protected)
        and len(next_destroyed) == len(destroyed)
        and next_threat_tile == enemy_threat_tile
        and next_repair_tile == robot_repair_tile
        and next_repair_turns == robot_repair_turns
        and next_robot == robot_pos
        and next_enemy == enemy_pos
    )


def _mode6_resolve_positions(robot_pos, enemy_pos, protected, destroyed,
                             all_crops, chance_mode, crop_profiles,
                             enemy_threat_tile=None,
                             enemy_threat_turns=0,
                             robot_repair_tile=None,
                             robot_repair_turns=0,
                             enemy_active=True):
    protected = set(protected)
    destroyed = set(destroyed)
    remaining = _mode6_remaining(all_crops, protected, destroyed)
    chance_tile = None

    if robot_pos in remaining:
        if robot_pos == robot_repair_tile:
            robot_repair_turns += 1
        else:
            robot_repair_tile = robot_pos
            robot_repair_turns = 1
        if robot_repair_turns >= 2:
            protected.add(robot_pos)
            remaining.remove(robot_pos)
            robot_repair_tile = None
            robot_repair_turns = 0
            if enemy_threat_tile == robot_pos:
                enemy_threat_tile = None
                enemy_threat_turns = 0
    else:
        robot_repair_tile = None
        robot_repair_turns = 0

    if enemy_active and enemy_pos in remaining:
        if enemy_pos == enemy_threat_tile:
            enemy_threat_turns += 1
        else:
            enemy_threat_tile = enemy_pos
            enemy_threat_turns = 1
        if enemy_threat_turns >= 2:
            if chance_mode:
                chance_tile = enemy_pos
            else:
                destroyed.add(enemy_pos)
            if robot_repair_tile == enemy_pos:
                robot_repair_tile = None
                robot_repair_turns = 0
            enemy_threat_tile = None
            enemy_threat_turns = 0
    elif enemy_active:
        enemy_threat_tile = None
        enemy_threat_turns = 0
    else:
        enemy_threat_tile = None
        enemy_threat_turns = 0

    return (
        frozenset(protected), frozenset(destroyed), chance_tile,
        enemy_threat_tile, enemy_threat_turns,
        robot_repair_tile, robot_repair_turns)


def _mode6_apply_chance(tile, protected, destroyed, all_crops,
                        crop_profiles, expected):
    if tile is None:
        return evaluate_farm_state(all_crops, protected, destroyed,
                                   crop_profiles)

    outcomes = mode6_chance_outcomes(tile, crop_profiles)
    damage_destroyed = set(destroyed)
    if tile not in protected:
        damage_destroyed.add(tile)
    damage_destroyed = frozenset(damage_destroyed)

    if expected:
        safe_value = evaluate_farm_state(
            all_crops, protected, destroyed, crop_profiles)
        damage_value = evaluate_farm_state(
            all_crops, protected, damage_destroyed, crop_profiles)
        return sum(
            (
                safe_value
                if event == "no_damage"
                else safe_value + MODE6_LIGHTNING_ROD_BONUS
                if event == "lightning_rod"
                else damage_value
            ) * prob
            for event, prob in outcomes)

    return evaluate_farm_state(
        all_crops, protected, damage_destroyed, crop_profiles)


def _mode6_chance_search_value(
        algorithm, chance_tiles, robot_pos, enemy_pos, protected, destroyed,
        all_crops, walkable_tiles, crop_profiles, next_depth, counters,
        memo, robot_prev, enemy_prev, enemy_threat_tile,
        enemy_threat_turns, robot_repair_tile, robot_repair_turns):
    chance_tiles = tuple(chance_tiles or ())
    safe_value, safe_line = _mode6_search(
        algorithm, robot_pos, enemy_pos, protected, destroyed,
        all_crops, walkable_tiles, crop_profiles, next_depth, counters,
        memo=memo, robot_prev=robot_prev, enemy_prev=enemy_prev,
        enemy_threat_tile=enemy_threat_tile,
        enemy_threat_turns=enemy_threat_turns,
        robot_repair_tile=robot_repair_tile,
        robot_repair_turns=robot_repair_turns)
    if not chance_tiles:
        return safe_value, safe_line

    # Runtime weather chooses by crop-risk weight every five seconds. Fold its
    # immediate expected delta into one recursive branch so depth 6 does not
    # duplicate the complete subtree for every CHANCE outcome.
    counters["chance"] += 3
    delta_cache = counters.setdefault("chance_delta_cache", {})
    delta_key = (protected, destroyed)
    if delta_key in delta_cache:
        return safe_value + delta_cache[delta_key], safe_line

    base_state_value = evaluate_farm_state(
        all_crops, protected, destroyed, crop_profiles)
    weighted_tiles, tile_weights = mode6_chance_tile_weights(
        chance_tiles, crop_profiles)
    total_weight = sum(tile_weights)
    expected_delta = sum(
        weight * (
            _mode6_apply_chance(
                tile, protected, destroyed, all_crops, crop_profiles,
                expected=True) - base_state_value)
        for tile, weight in zip(weighted_tiles, tile_weights)
    ) / total_weight
    delta_cache[delta_key] = expected_delta
    return safe_value + expected_delta, safe_line


def _mode6_expectimax_search(
        robot_pos, enemy_pos, protected, destroyed, all_crops,
        walkable_tiles, crop_profiles, depth, counters, memo,
        robot_prev=None, enemy_prev=None, enemy_threat_tile=None,
        enemy_threat_turns=0, robot_repair_tile=None,
        robot_repair_turns=0):
    memo_key = (
        "Expectimax", robot_pos, enemy_pos, protected, destroyed, depth,
        robot_prev, enemy_prev, robot_repair_tile, robot_repair_turns,
    )
    if memo_key in memo:
        return memo[memo_key]

    counters["evaluated"] += 1
    remaining = _mode6_remaining(all_crops, protected, destroyed)
    if depth <= 0 or not remaining:
        result = (
            _mode6_positional_evaluation(
                robot_pos, enemy_pos, all_crops, protected, destroyed,
                crop_profiles, robot_repair_tile=robot_repair_tile,
                robot_repair_turns=robot_repair_turns,
                enemy_active=False),
            [],
        )
        memo[memo_key] = result
        return result

    action_cache = counters.setdefault("action_cache", {})
    resolved = set(all_crops) - remaining
    robot_actions = _mode6_rank_actions(
        robot_pos, walkable_tiles, remaining, crop_profiles, robot_prev,
        resolved, None, action_cache)
    counters["max"] += 1
    best_value = -INF
    best_line = []

    for robot_action in robot_actions:
        next_robot = _mode6_move(robot_pos, robot_action)
        (next_protected, next_destroyed, _ignored_chance_tile,
         _next_threat_tile, _next_threat_turns,
         next_repair_tile, next_repair_turns) = _mode6_resolve_positions(
            next_robot, enemy_pos, protected, destroyed, all_crops,
            False, crop_profiles, None, 0,
            robot_repair_tile, robot_repair_turns, enemy_active=False)
        next_remaining = _mode6_remaining(
            all_crops, next_protected, next_destroyed)
        child, child_line = _mode6_chance_search_value(
            "Expectimax", next_remaining, next_robot, enemy_pos,
            next_protected, next_destroyed, all_crops, walkable_tiles,
            crop_profiles, depth - 1, counters, memo, robot_pos, enemy_pos,
            None, 0, next_repair_tile, next_repair_turns)
        if _mode6_no_progress(
                robot_pos, enemy_pos, next_robot, enemy_pos,
                protected, destroyed, next_protected, next_destroyed,
                None, None, robot_repair_tile, robot_repair_turns,
                next_repair_tile, next_repair_turns):
            child -= 100
        if child > best_value:
            best_value = child
            best_line = [(robot_action, "CHANCE")] + child_line

    result = (best_value, best_line)
    memo[memo_key] = result
    return result


def _mode6_expectiminimax_search(
        robot_pos, enemy_pos, protected, destroyed, all_crops,
        walkable_tiles, crop_profiles, depth, counters, memo,
        robot_prev=None, enemy_prev=None, enemy_threat_tile=None,
        enemy_threat_turns=0, robot_repair_tile=None,
        robot_repair_turns=0):
    memo_key = (
        "Expectiminimax", robot_pos, enemy_pos, protected, destroyed, depth,
        robot_prev, enemy_prev, enemy_threat_tile, enemy_threat_turns,
        robot_repair_tile, robot_repair_turns,
    )
    if memo_key in memo:
        return memo[memo_key]

    counters["evaluated"] += 1
    remaining = _mode6_remaining(all_crops, protected, destroyed)
    if depth <= 0 or not remaining:
        result = (
            _mode6_positional_evaluation(
                robot_pos, enemy_pos, all_crops, protected, destroyed,
                crop_profiles, enemy_threat_tile, enemy_threat_turns,
                robot_repair_tile, robot_repair_turns, enemy_active=True),
            [],
        )
        memo[memo_key] = result
        return result

    action_cache = counters.setdefault("action_cache", {})
    resolved = set(all_crops) - remaining
    robot_actions = _mode6_rank_actions(
        robot_pos, walkable_tiles, remaining, crop_profiles, robot_prev,
        resolved, enemy_threat_tile, action_cache)
    counters["max"] += 1
    best_value = -INF
    best_line = []

    for robot_action in robot_actions:
        next_robot = _mode6_move(robot_pos, robot_action)
        enemy_actions = _mode6_enemy_legal_actions(
            enemy_pos, remaining, all_crops, protected, destroyed,
            crop_profiles, walkable_tiles, enemy_threat_tile,
            enemy_threat_turns, action_cache)
        counters["min"] += 1
        min_value = INF
        min_line = []

        for enemy_action in enemy_actions:
            next_enemy = _mode6_move(enemy_pos, enemy_action)
            (next_protected, next_destroyed, _ignored_chance_tile,
             next_threat_tile, next_threat_turns,
             next_repair_tile, next_repair_turns) = (
                _mode6_resolve_positions(
                    next_robot, next_enemy, protected, destroyed,
                    all_crops, False, crop_profiles,
                    enemy_threat_tile, enemy_threat_turns,
                    robot_repair_tile, robot_repair_turns,
                    enemy_active=True))
            next_remaining = _mode6_remaining(
                all_crops, next_protected, next_destroyed)
            child, child_line = _mode6_chance_search_value(
                "Expectiminimax", next_remaining, next_robot, next_enemy,
                next_protected, next_destroyed, all_crops, walkable_tiles,
                crop_profiles, depth - 2, counters, memo,
                robot_pos, enemy_pos, next_threat_tile,
                next_threat_turns, next_repair_tile, next_repair_turns)
            if _mode6_no_progress(
                    robot_pos, enemy_pos, next_robot, next_enemy,
                    protected, destroyed, next_protected, next_destroyed,
                    enemy_threat_tile, next_threat_tile,
                    robot_repair_tile, robot_repair_turns,
                    next_repair_tile, next_repair_turns):
                child -= 100
            if child < min_value:
                min_value = child
                min_line = [(robot_action, enemy_action)] + child_line

        if min_value > best_value:
            best_value = min_value
            best_line = min_line

    result = (best_value, best_line)
    memo[memo_key] = result
    return result


def _mode6_search(algorithm, robot_pos, enemy_pos, protected, destroyed,
                   all_crops, walkable_tiles, crop_profiles, depth, counters,
                   alpha=-INF, beta=INF, memo=None,
                   robot_prev=None, enemy_prev=None,
                   enemy_threat_tile=None, enemy_threat_turns=0,
                   robot_repair_tile=None, robot_repair_turns=0):
    memo = memo if memo is not None else {}
    if algorithm == "Expectimax":
        return _mode6_expectimax_search(
            robot_pos, enemy_pos, protected, destroyed, all_crops,
            walkable_tiles, crop_profiles, depth, counters, memo,
            robot_prev, enemy_prev, enemy_threat_tile,
            enemy_threat_turns, robot_repair_tile, robot_repair_turns)
    if algorithm == "Expectiminimax":
        return _mode6_expectiminimax_search(
            robot_pos, enemy_pos, protected, destroyed, all_crops,
            walkable_tiles, crop_profiles, depth, counters, memo,
            robot_prev, enemy_prev, enemy_threat_tile,
            enemy_threat_turns, robot_repair_tile, robot_repair_turns)

    memo_key = (
        algorithm, robot_pos, enemy_pos, protected, destroyed, depth,
        robot_prev, enemy_prev, enemy_threat_tile, enemy_threat_turns,
        robot_repair_tile, robot_repair_turns,
        round(alpha, 3) if algorithm == "Alpha-Beta" else None,
        round(beta, 3) if algorithm == "Alpha-Beta" else None,
    )
    if memo_key in memo:
        return memo[memo_key]

    action_cache = counters.setdefault("action_cache", {})
    counters["evaluated"] += 1
    remaining = _mode6_remaining(all_crops, protected, destroyed)
    if depth <= 0 or not remaining:
        result = (
            _mode6_positional_evaluation(
                robot_pos, enemy_pos, all_crops, protected, destroyed,
                crop_profiles, enemy_threat_tile, enemy_threat_turns,
                robot_repair_tile, robot_repair_turns,
                enemy_active=algorithm != "Expectimax"), [])
        memo[memo_key] = result
        return result

    chance_mode = algorithm in ("Expectimax", "Expectiminimax")
    use_min = algorithm in ("Minimax", "Alpha-Beta", "Expectiminimax")
    resolved = set(all_crops) - remaining
    robot_actions = _mode6_rank_actions(
        robot_pos, walkable_tiles, remaining, crop_profiles, robot_prev,
        resolved, enemy_threat_tile, action_cache)

    best_value = -INF
    best_line = []
    counters["max"] += 1

    for robot_index, robot_action in enumerate(robot_actions):
        next_robot = _mode6_move(robot_pos, robot_action)

        if use_min:
            counters["min"] += 1
            enemy_value = INF
            enemy_line = []
            local_beta = beta
            enemy_actions = _mode6_enemy_legal_actions(
                enemy_pos, remaining, all_crops, protected, destroyed,
                crop_profiles, walkable_tiles, enemy_threat_tile,
                enemy_threat_turns, action_cache)
            for enemy_index, enemy_action in enumerate(enemy_actions):
                next_enemy = _mode6_move(enemy_pos, enemy_action)
                (next_protected, next_destroyed, chance_tile,
                 next_threat_tile, next_threat_turns,
                 next_repair_tile, next_repair_turns) = (
                    _mode6_resolve_positions(
                        next_robot, next_enemy, protected, destroyed,
                        all_crops, chance_mode, crop_profiles,
                        enemy_threat_tile, enemy_threat_turns,
                        robot_repair_tile, robot_repair_turns))
                if chance_tile is not None:
                    counters["chance"] += 1
                    chance_value = _mode6_apply_chance(
                        chance_tile, next_protected, next_destroyed,
                        all_crops, crop_profiles, expected=False)
                    if algorithm == "Expectiminimax":
                        outcomes = mode6_chance_outcomes(
                            chance_tile, crop_profiles)
                        safe_value, safe_line = _mode6_search(
                            algorithm, next_robot, next_enemy,
                            next_protected, next_destroyed,
                            all_crops, walkable_tiles, crop_profiles,
                            depth - 2, counters, alpha, local_beta, memo,
                            robot_pos, enemy_pos,
                            next_threat_tile, next_threat_turns,
                            next_repair_tile, next_repair_turns)
                        damage_destroyed = frozenset(
                            set(next_destroyed) | {chance_tile})
                        damage_value, damage_line = _mode6_search(
                            algorithm, next_robot, next_enemy,
                            next_protected, damage_destroyed,
                            all_crops, walkable_tiles, crop_profiles,
                            depth - 2, counters, alpha, local_beta, memo,
                            robot_pos, enemy_pos,
                            next_threat_tile, next_threat_turns,
                            next_repair_tile, next_repair_turns)
                        chance_value = sum(
                            (safe_value if event == "no_damage"
                             else damage_value) * prob
                            for event, prob in outcomes)
                        child_line = safe_line if safe_value >= damage_value else damage_line
                    else:
                        child_line = []
                    child = chance_value
                else:
                    child, child_line = _mode6_search(
                        algorithm, next_robot, next_enemy,
                        next_protected, next_destroyed,
                        all_crops, walkable_tiles, crop_profiles,
                        depth - 2, counters, alpha, local_beta, memo,
                        robot_pos, enemy_pos,
                        next_threat_tile, next_threat_turns,
                        next_repair_tile, next_repair_turns)
                if _mode6_no_progress(
                        robot_pos, enemy_pos, next_robot, next_enemy,
                        protected, destroyed, next_protected,
                        next_destroyed, enemy_threat_tile,
                        next_threat_tile, robot_repair_tile,
                        robot_repair_turns, next_repair_tile,
                        next_repair_turns):
                    child -= 100
                if child < enemy_value:
                    enemy_value = child
                    enemy_line = [(robot_action, enemy_action)] + child_line
                if algorithm == "Alpha-Beta":
                    local_beta = min(local_beta, enemy_value)
                    if local_beta <= alpha:
                        counters["pruned"] += max(
                            0, len(enemy_actions) - enemy_index - 1)
                        break
            value = enemy_value
            line = enemy_line
        else:
            next_enemy = enemy_pos
            enemy_action = "CHANCE"
            (next_protected, next_destroyed, chance_tile,
             next_threat_tile, next_threat_turns,
             next_repair_tile, next_repair_turns) = (
                _mode6_resolve_positions(
                    next_robot, next_enemy, protected, destroyed,
                    all_crops, True, crop_profiles,
                    enemy_threat_tile, enemy_threat_turns,
                    robot_repair_tile, robot_repair_turns,
                    enemy_active=False))
            child, child_line = _mode6_search(
                algorithm, next_robot, next_enemy,
                next_protected, next_destroyed,
                all_crops, walkable_tiles, crop_profiles,
                depth - 1, counters, alpha, beta, memo,
                robot_pos, enemy_pos,
                next_threat_tile, next_threat_turns,
                next_repair_tile, next_repair_turns)
            random_remaining = _mode6_remaining(
                all_crops, next_protected, next_destroyed)
            if random_remaining and depth >= MODE6_SEARCH_DEPTH:
                counters["chance"] += 1
                safe_now = evaluate_farm_state(
                    all_crops, next_protected, next_destroyed,
                    crop_profiles)
                expected_now = sum(
                    _mode6_apply_chance(
                        tile, next_protected, next_destroyed,
                        all_crops, crop_profiles, expected=True)
                    for tile in random_remaining) / len(random_remaining)
                child += expected_now - safe_now
            if _mode6_no_progress(
                    robot_pos, enemy_pos, next_robot, next_enemy,
                    protected, destroyed, next_protected,
                    next_destroyed, enemy_threat_tile,
                    next_threat_tile, robot_repair_tile,
                    robot_repair_turns, next_repair_tile,
                    next_repair_turns):
                child -= 100
            value = child
            line = [(robot_action, enemy_action)] + child_line

        if value > best_value:
            best_value = value
            best_line = line
        if algorithm == "Alpha-Beta":
            alpha = max(alpha, best_value)
            if alpha >= beta:
                counters["pruned"] += max(
                    0, len(robot_actions) - robot_index - 1)
                break

    result = (best_value, best_line)
    memo[memo_key] = result
    return result


def adversarial_choose(algorithm, robot_pos, enemy_pos, all_crops,
                       protected, destroyed, depth=MODE6_SEARCH_DEPTH,
                       crop_profiles=None, walkable_tiles=None,
                       robot_prev=None, enemy_prev=None,
                       enemy_threat_tile=None, enemy_threat_turns=0,
                       robot_repair_tile=None, robot_repair_turns=0):
    crop_profiles = crop_profiles or {}
    all_crops = frozenset(all_crops)
    protected = frozenset(protected)
    destroyed = frozenset(destroyed)
    walkable_tiles = set(walkable_tiles or all_crops)
    walkable_tiles.update(all_crops)
    walkable_tiles.add(robot_pos)
    walkable_tiles.add(enemy_pos)
    counters = {
        "max": 0,
        "min": 0,
        "chance": 0,
        "evaluated": 0,
        "pruned": 0,
    }

    value, line = _mode6_search(
        algorithm, robot_pos, enemy_pos, protected, destroyed,
        all_crops, walkable_tiles, crop_profiles, depth, counters, memo={},
        robot_prev=robot_prev, enemy_prev=enemy_prev,
        enemy_threat_tile=enemy_threat_tile,
        enemy_threat_turns=enemy_threat_turns,
        robot_repair_tile=robot_repair_tile,
        robot_repair_turns=robot_repair_turns)
    robot_action, enemy_action = line[0] if line else ("STAY", "STAY")
    protected_score, destroyed_score = mode6_state_scores(
        protected, destroyed, crop_profiles)
    remaining = _mode6_remaining(all_crops, protected, destroyed)
    robot_frontier = _mode6_rank_actions(
        robot_pos, walkable_tiles, remaining, crop_profiles, robot_prev,
        set(all_crops) - remaining, enemy_threat_tile,
        counters.setdefault("action_cache", {}))
    if algorithm == "Expectimax":
        greedy_enemy_action = "CHANCE"
        enemy_priority_target = None
        enemy_priority_score = 0.0
        enemy_frontier = ["random tree"]
        enemy_policy = "Random Chance"
        enemy_response = "random tree damage"
    else:
        (greedy_enemy_action, enemy_priority_target,
         enemy_priority_score) = _mode6_enemy_greedy_choice(
            enemy_pos, all_crops, protected, destroyed, crop_profiles,
            walkable_tiles, enemy_threat_tile, enemy_threat_turns)
        enemy_frontier = _mode6_enemy_legal_frontier(
            enemy_pos, remaining, all_crops, protected, destroyed,
            crop_profiles, walkable_tiles, enemy_threat_tile,
            enemy_threat_turns)
        enemy_policy = "Full MIN legal actions"
        enemy_response = "adversarial min over all legal enemy actions"

    stats = {
        "Algorithm": algorithm,
        "Robot Algorithm": algorithm,
        "Enemy Policy": enemy_policy,
        "Enemy Target": enemy_priority_target,
        "Enemy Target Score": f"{enemy_priority_score:.1f}",
        "Phase": "PLAN_STEP",
        "Search depth": depth,
        "Ply model": "1 ply = 1 actor action",
        "Lookahead rounds": depth // 2,
        "Enemy response": enemy_response,
        "MAX action": robot_action,
        "MIN action": "N/A" if algorithm == "Expectimax" else enemy_action,
        "Robot Action": robot_action,
        "Enemy Action": enemy_action,
        "Evaluation score": f"{value:.1f}",
        "Protected score": protected_score,
        "Destroyed score": destroyed_score,
        "Remaining trees": len(_mode6_remaining(
            all_crops, protected, destroyed)),
        "MAX nodes": counters["max"],
        "MIN nodes": counters["min"],
        "CHANCE nodes": counters["chance"],
        "Nodes explored": counters["evaluated"],
        "Nodes evaluated": counters["evaluated"],
        "MAX frontier": ", ".join(robot_frontier),
        "MIN frontier": ", ".join(enemy_frontier),
        "Threatened tile": enemy_threat_tile or "None",
        "Repairing tile": (
            f"{robot_repair_tile} ({robot_repair_turns}/2)"
            if robot_repair_tile is not None else "None"),
    }
    info = {
        "alpha": "-\u221e",
        "beta": "+\u221e",
        "pruned": counters["pruned"],
    }
    if algorithm == "Alpha-Beta":
        stats["Pruned Nodes"] = counters["pruned"]

    tree_details = {
        "robot_action": robot_action,
        "enemy_action": enemy_action,
        "enemy_policy": enemy_policy,
        "enemy_priority_target": enemy_priority_target,
        "enemy_priority_score": enemy_priority_score,
        "depth": depth,
        "lookahead_rounds": depth // 2,
        "enemy_response": enemy_response,
        "evaluation": value,
        "line": line,
        "protected_score": protected_score,
        "destroyed_score": destroyed_score,
        "pruned": counters["pruned"],
        "robot_prev": robot_prev,
        "enemy_prev": enemy_prev,
        "robot_frontier": robot_frontier,
        "enemy_frontier": enemy_frontier,
        "enemy_threat_tile": enemy_threat_tile,
        "enemy_threat_turns": enemy_threat_turns,
        "robot_repair_tile": robot_repair_tile,
        "robot_repair_turns": robot_repair_turns,
    }
    return robot_action, enemy_action, value, info, stats, tree_details


def _mode6_target_distance(start, target, walkable_tiles, neighbors=None):
    if start == target:
        return 0
    if target not in walkable_tiles:
        return INF
    if neighbors is None:
        return _mode6_manhattan(start, target)

    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        current, distance = queue.popleft()
        for next_tile in neighbors(current, set()):
            if next_tile in visited or next_tile not in walkable_tiles:
                continue
            if next_tile == target:
                return distance + 1
            visited.add(next_tile)
            queue.append((next_tile, distance + 1))
    return INF


def _mode6_shortest_target_path(start, target, walkable_tiles, neighbors=None):
    if target is None:
        return []
    if start == target:
        return []
    if target not in walkable_tiles:
        return []
    if neighbors is None:
        return [target]

    queue = deque([(start, [])])
    visited = {start}
    while queue:
        current, path = queue.popleft()
        for next_tile in neighbors(current, set()):
            if next_tile in visited or next_tile not in walkable_tiles:
                continue
            next_path = path + [next_tile]
            if next_tile == target:
                return next_path
            visited.add(next_tile)
            queue.append((next_tile, next_path))
    return []


def _mode6_path_score(path, remaining, crop_profiles):
    return sum(
        _crop_value(tile, crop_profiles)
        for tile in path
        if tile in remaining)


def _mode6_length_factor(distance):
    return 1.0 / (1.0 + 0.25 * max(0, distance))


def _mode6_target_score(tile, actor_pos, opponent_pos, crop_profiles,
                        walkable_tiles, neighbors, actor, remaining=None):
    path = _mode6_shortest_target_path(
        actor_pos, tile, walkable_tiles, neighbors)
    distance = len(path) if actor_pos != tile else 0
    if actor_pos != tile and not path:
        distance = INF
    if distance == INF:
        return -INF
    value = _crop_value(tile, crop_profiles)
    condition = crop_profiles.get(tile, {}).get("condition")
    golden_bonus = 80 if condition == "golden" else 0
    remaining = set(remaining or ())
    route_value = _mode6_path_score(path or [tile], remaining, crop_profiles)
    length_factor = _mode6_length_factor(distance)
    weighted_target = (value + golden_bonus) * length_factor
    side_route_bonus = max(0, route_value - value)
    opponent_distance = _mode6_target_distance(
        opponent_pos, tile, walkable_tiles, neighbors)
    if opponent_distance == INF:
        opponent_distance = _mode6_manhattan(opponent_pos, tile)

    if actor == "robot":
        route_bonus = 0.55 * side_route_bonus
        race_bonus = 28 if opponent_distance <= distance else 10
        return (
            weighted_target + route_bonus + race_bonus - 2 * distance)

    route_bonus = 0.35 * side_route_bonus
    far_robot_bonus = max(0, opponent_distance - distance) * 4
    return weighted_target + route_bonus + far_robot_bonus - 3 * distance


def adversarial_choose_target(algorithm, robot_pos, enemy_pos, all_crops,
                              protected, destroyed, crop_profiles=None,
                              walkable_tiles=None, neighbors=None,
                              depth=MODE6_SEARCH_DEPTH,
                              enemy_threat_tile=None,
                              enemy_threat_turns=0,
                              robot_repair_tile=None,
                              robot_repair_turns=0,
                              **_):
    crop_profiles = crop_profiles or {}
    all_crops = tuple(all_crops)
    protected = frozenset(protected)
    destroyed = frozenset(destroyed)
    processed = protected | destroyed
    remaining = set(all_crops) - processed
    walkable_tiles = set(walkable_tiles or all_crops)
    walkable_tiles.update(all_crops)
    walkable_tiles.add(robot_pos)
    walkable_tiles.add(enemy_pos)

    counters = {
        "max": 0,
        "min": 0,
        "chance": 0,
        "evaluated": 0,
        "pruned": 0,
    }
    if not remaining:
        protected_score, destroyed_score = mode6_state_scores(
            protected, destroyed, crop_profiles)
        stats = {
            "Algorithm": algorithm,
            "Phase": "CHOOSE_TARGET",
            "Search depth": depth,
            "Nodes evaluated": 0,
            "Remaining trees": 0,
            "Protected score": protected_score,
            "Destroyed score": destroyed_score,
            "Evaluation score": f"{protected_score - destroyed_score:.1f}",
        }
        return None, None, protected_score - destroyed_score, stats, {
            "robot_target": None,
            "enemy_target": None,
            "target_mode": "none",
        }

    target_radius = 3
    robot_distances = {
        tile: _mode6_target_distance(
            robot_pos, tile, walkable_tiles, neighbors)
        for tile in remaining
    }
    enemy_distances = {
        tile: _mode6_target_distance(
            enemy_pos, tile, walkable_tiles, neighbors)
        for tile in remaining
    }
    robot_candidates = {
        tile for tile in remaining
        if robot_distances[tile] <= target_radius
    }
    enemy_candidates = {
        tile for tile in remaining
        if enemy_distances[tile] <= target_radius
    }
    if not robot_candidates:
        robot_candidates = set(remaining)
    if not enemy_candidates:
        enemy_candidates = set(remaining)

    robot_ranked = sorted(
        robot_candidates,
        key=lambda tile: (
            -_mode6_target_score(
                tile, robot_pos, enemy_pos, crop_profiles,
                walkable_tiles, neighbors, "robot", remaining),
            robot_distances[tile],
            tile,
        ),
    )
    enemy_ranked = sorted(
        enemy_candidates,
        key=lambda tile: (
            -_mode6_target_score(
                tile, enemy_pos, robot_pos, crop_profiles,
                walkable_tiles, neighbors, "enemy", remaining),
            enemy_distances[tile],
            tile,
        ),
    )
    robot_target = robot_ranked[0] if robot_ranked else None
    enemy_target = enemy_ranked[0] if enemy_ranked else None
    if robot_repair_tile in remaining and robot_pos == robot_repair_tile:
        robot_target = robot_repair_tile
    if enemy_threat_tile in remaining and enemy_pos == enemy_threat_tile:
        enemy_target = enemy_threat_tile

    counters["max"] = len(robot_ranked)
    counters["min"] = len(enemy_ranked)
    counters["evaluated"] = len(robot_ranked) * max(1, len(enemy_ranked))

    protected_score, destroyed_score = mode6_state_scores(
        protected, destroyed, crop_profiles)
    robot_path_scores = {
        tile: _mode6_path_score(
            _mode6_shortest_target_path(
                robot_pos, tile, walkable_tiles, neighbors) or [tile],
            remaining,
            crop_profiles)
        for tile in robot_ranked[:5]
    }
    enemy_path_scores = {
        tile: _mode6_path_score(
            _mode6_shortest_target_path(
                enemy_pos, tile, walkable_tiles, neighbors) or [tile],
            remaining,
            crop_profiles)
        for tile in enemy_ranked[:5]
    }
    projected_protected = protected_score
    projected_destroyed = destroyed_score
    if robot_target is not None:
        projected_protected += _crop_value(robot_target, crop_profiles)
    if enemy_target is not None and enemy_target != robot_target:
        enemy_value = _crop_value(enemy_target, crop_profiles)
        if algorithm == "Expectiminimax":
            projected_destroyed += enemy_value
            counters["chance"] = 3
        elif algorithm == "Expectimax":
            counters["chance"] = 3
        else:
            projected_destroyed += enemy_value

    value = projected_protected - projected_destroyed
    stats = {
        "Algorithm": algorithm,
        "Phase": "CHOOSE_TARGET",
        "Search depth": depth,
        "Target radius": target_radius,
        "Target model": "adversarial target selection",
        "Route scoring": "remaining route trees only; processed trees = 0",
        "Robot target": robot_target,
        "Enemy target": enemy_target,
        "Robot path score": robot_path_scores.get(robot_target, 0),
        "Enemy path score": enemy_path_scores.get(enemy_target, 0),
        "Repairing tile": (
            f"{robot_repair_tile} ({robot_repair_turns}/2)"
            if robot_repair_tile is not None else "None"),
        "Threatened tile": (
            f"{enemy_threat_tile} ({enemy_threat_turns}/2)"
            if enemy_threat_tile is not None else "None"),
        "Evaluation score": f"{value:.1f}",
        "Protected score": protected_score,
        "Destroyed score": destroyed_score,
        "Remaining trees": len(remaining),
        "MAX nodes": counters["max"],
        "MIN nodes": counters["min"],
        "CHANCE nodes": counters["chance"],
        "Nodes evaluated": counters["evaluated"],
    }
    if algorithm == "Alpha-Beta":
        stats["Pruned Nodes"] = max(0, len(robot_ranked) - 1)

    details = {
        "robot_target": robot_target,
        "enemy_target": enemy_target,
        "robot_ranked_targets": robot_ranked,
        "enemy_ranked_targets": enemy_ranked,
        "depth": depth,
        "target_radius": target_radius,
        "evaluation": value,
        "target_mode": "adversarial target + A*/UCS one-step",
        "route_scoring": "remaining route trees only; processed trees = 0",
        "robot_path_scores": robot_path_scores,
        "enemy_path_scores": enemy_path_scores,
        "enemy_threat_tile": enemy_threat_tile,
        "enemy_threat_turns": enemy_threat_turns,
        "robot_repair_tile": robot_repair_tile,
        "robot_repair_turns": robot_repair_turns,
        "protected_score": protected_score,
        "destroyed_score": destroyed_score,
        "pruned": stats.get("Pruned Nodes", 0),
    }
    return robot_target, enemy_target, value, stats, details
