from collections import deque
from itertools import combinations
import heapq
import math
import random


INF = float("inf")

UNINFORMED_ALGORITHMS = ("BFS", "DFS", "IDS", "UCS")
INFORMED_ALGORITHMS = ("Greedy", "IDSA", "A*")
LOCAL_ALGORITHMS = ("Local Beam", "Hill Climbing", "Annealing", "Restart Hill")
ONLINE_ALGORITHMS = (
    "Online A*",
    "Online BFS",
    "Belief-State BFS",
    "AND-OR Search",
)
CSP_ALGORITHMS = ("Backtrack", "Fwd Check", "AC-3", "Min Conflict")
ADVERSARIAL_ALGORITHMS = (
    "Minimax",
    "Alpha-Beta",
    "Expectimax",
    "Expectiminimax",
)

def csp_crop_pair_valid(left, right):
    """Return whether two adjacent Mode 5 crops satisfy all binary constraints."""
    return left != right

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


def find_path_by_algorithm(algorithm, start, goal, blocked, neighbors,
                           heuristic, counter, step_cost=unit_step_cost):
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
                    step_cost=unit_step_cost):
    goals = set(goals)
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

        if current in goals:
            path = reconstruct_path(came_from, current)
            return path, current, explored, {
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

    return [], None, explored, {
        "Nodes explored": len(explored),
        "Path length": 0,
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
    """Traverse once with DFS and collect every goal in first-visit order."""
    goals = set(goals)
    visited = {start}
    explored_order = [start]
    plan = []

    def visit(tile):
        if tile in goals and tile not in plan:
            plan.append(tile)

        for next_tile in neighbors(tile, blocked):
            if next_tile in visited:
                continue
            visited.add(next_tile)
            explored_order.append(next_tile)
            visit(next_tile)

    visit(start)
    reachable = set(plan)
    stats = {
        "Algorithm": "DFS",
        "Planning mode": "Full garden DFS traversal",
        "Plan targets": len(plan),
        "Unreachable": len(goals - reachable),
        "Nodes explored": len(explored_order),
        "Unique explored": len(visited),
        "Stack max": len(explored_order),
    }
    return plan, set(explored_order), stats


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
    heapq.heappush(open_set, (0, next(counter), start))
    best_cost = {start: 0}
    explored = []
    plan = []

    while open_set and len(plan) < len(goals):
        cost, _, current = heapq.heappop(open_set)
        if cost > best_cost.get(current, INF):
            continue
        explored.append(current)
        if current in goals and current not in plan:
            plan.append(current)

        for next_tile in neighbors(current, blocked):
            next_cost = cost + step_cost(current, next_tile)
            if next_cost < best_cost.get(next_tile, INF):
                best_cost[next_tile] = next_cost
                heapq.heappush(
                    open_set, (next_cost, next(counter), next_tile))

    stats = _full_traversal_stats(
        "UCS", plan, goals, explored,
        {"Max cost": max((best_cost.get(tile, 0) for tile in plan), default=0)})
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


def _plan_hill_climb(initial_plan, start, dryness, heuristic, scores,
                     max_steps=40):
    current = (
        list(initial_plan)
        if initial_plan is not None
        else [])
    current_value = _plan_objective(
        current, start, dryness, heuristic, scores)
    steps = 0

    while steps < max_steps:
        successors = _plan_successors(current)
        if not successors:
            break
        best_neighbor = min(
            successors,
            key=lambda plan: (_plan_objective(
                                  plan, start, dryness, heuristic, scores),
                              tuple(plan)))
        best_value = _plan_objective(
            best_neighbor, start, dryness, heuristic, scores)
        if best_value >= current_value:
            break
        current = best_neighbor
        current_value = best_value
        steps += 1

    return current, current_value, steps


def plan_hill_climbing_choose(remaining, start, dryness, heuristic,
                              initial_plan=None, max_steps=40):
    scores = _plan_scores(remaining, start, dryness, heuristic)
    if not remaining:
        return None, 0, scores, {
            "Local algorithm": "Hill Climbing",
            "Target": "None",
        }

    start_plan = (
        list(initial_plan)
        if initial_plan is not None
        else _nearest_neighbor_plan(remaining, start, heuristic))
    current, _, steps = _plan_hill_climb(
        start_plan, start, dryness, heuristic, scores, max_steps)

    return _plan_result_stats(
        "Hill Climbing", current, start, dryness, heuristic, scores,
        {"Hill steps": steps})


def plan_restart_hill_choose(remaining, start, dryness, heuristic,
                             restarts=8):
    scores = _plan_scores(remaining, start, dryness, heuristic)
    if not remaining:
        return None, 0, scores, {
            "Local algorithm": "Restart Hill",
            "Target": "None",
        }

    starts = [_nearest_neighbor_plan(remaining, start, heuristic)]
    ordered_by_urgency = sorted(
        remaining,
        key=lambda tile: (-dryness.get(tile, 50), heuristic(start, tile), tile))
    starts.append(ordered_by_urgency)
    while len(starts) < max(1, restarts):
        plan = list(remaining)
        random.shuffle(plan)
        starts.append(plan)

    best_plan = None
    best_value = float("inf")
    for plan in starts[:max(1, restarts)]:
        candidate_plan, candidate_value, _ = _plan_hill_climb(
            plan, start, dryness, heuristic, scores)
        if candidate_value < best_value:
            best_value = candidate_value
            best_plan = candidate_plan

    return _plan_result_stats(
        "Restart Hill", best_plan, start, dryness, heuristic, scores,
        {"Restarts": max(1, restarts)})


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
    variables = list(farm_tiles)

    steps = []
    backtracks = 0
    arc_checks = 0
    show_domain_steps = algorithm in ("Fwd Check", "AC-3")
    adjacency = {
        var: [other for other in variables if _csp_adjacent(var, other)]
        for var in variables
    }
    domain = ("corn", "tomato", "wheat", "carrot")

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

    def select_unassigned(assignment, domains):
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

    def backtrack(assignment, domains, forward_check=False, arc_consistency=False):
        nonlocal backtracks
        if len(assignment) == len(variables):
            return dict(assignment)

        var = select_unassigned(assignment, domains)
        candidate_values = list(domains[var])
        for value in candidate_values:
            # The UI shows the solver considering several crops before it
            # commits to the candidate. FC/AC-3 only consider the current
            # pruned domain, so removed crops are never "thought about" again.
            previews = list(domains[var])
            random.shuffle(previews)
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


    # Min-Conflicts keeps the balanced preference used by its full random
    # assignment. The three systematic algorithms deliberately receive a
    # fresh random value order per tile so their search visibly tries,
    # rejects and backtracks instead of replaying a checkerboard answer.
    crop_list = ["corn", "tomato", "wheat", "carrot"]

    def _checker_crop(x, y):
        # (0,0)->corn, (1,0)->tomato, (0,1)->wheat, (1,1)->carrot
        idx = (x % 2) + (y % 2) * 2
        return crop_list[idx % 4]

    if variables and algorithm == "Min Conflict":
        def _preferred_domain(var):
            preferred = _checker_crop(var[0], var[1])
            rest = [c for c in domain if c != preferred]
            return [preferred] + rest

        domains = {var: _preferred_domain(var) for var in variables}
    else:
        domains = {}
        for var in variables:
            values = list(domain)
            random.shuffle(values)
            domains[var] = values


    if algorithm == "Fwd Check":
        result = backtrack({}, domains, forward_check=True)
    elif algorithm == "AC-3":
        result = (
            backtrack({}, domains, forward_check=True, arc_consistency=True)
            if enforce_arc_consistency(domains)
            else None
        )
    elif algorithm == "Min Conflict":
        result = min_conflicts()
        if result is None:
            # Chỉ fallback khi Min-Conflicts thực sự THẤT BẠI (hết max_steps/restarts
            # mà vẫn còn xung đột), không phải vì nghiệm "ít màu" — một nghiệm chỉ
            # dùng 2/4 loại crop vẫn hợp lệ với ràng buộc "khác màu kề nhau".
            result = backtrack({}, domains, forward_check=True)

    else:
        result = backtrack({}, domains)


    # Không post-process thay đổi assignment sau khi đã tìm nghiệm.
    # (Giữ replay CSP khớp với assignment cuối và tránh việc robot “quay lại ô cũ”).


    stats = {
        "Algorithm": algorithm,
        "Variables": len(variables),
        "Domain": "{corn, tomato, wheat, carrot}",
        "Constraints": "no same crop adjacent",
        "Backtracks": backtracks,
    }
    if algorithm == "AC-3":
        stats["Arc checks"] = arc_checks
    return result or {}, steps, stats


MODE6_SEARCH_DEPTH = 4
MODE6_MAX_BRANCHING = 7
MODE6_TREE_STATUS = {
    "healthy": {"value": 5, "risk": 0.10},
    "dry": {"value": 10, "risk": 0.20},
    "disease": {"value": 20, "risk": 0.40},
    "critical": {"value": 30, "risk": 0.70},
    "rare": {"value": 50, "risk": 0.50},
}
EXPECTIMAX_EVENTS = (
    ("Pest", 0.40),
    ("Hail", 0.35),
    ("Drought", 0.25),
)
EXPECTIMINIMAX_OUTCOMES = (
    ("Sabotage succeeds", 0.70),
    ("Weather damage", 0.20),
    ("No damage", 0.10),
)


def _crop_value(tile, crop_profiles):
    return crop_profiles.get(tile, MODE6_TREE_STATUS["healthy"])["value"]


def _crop_risk(tile, crop_profiles):
    return crop_profiles.get(tile, MODE6_TREE_STATUS["healthy"])["risk"]


def evaluate_farm_state(all_crops, protected, destroyed, crop_profiles=None):
    """Shared evaluation for every Mode-6 game-tree algorithm."""
    crop_profiles = crop_profiles or {}
    living = set(all_crops) - set(destroyed)
    living_value = sum(_crop_value(tile, crop_profiles) for tile in living)
    destroyed_value = sum(
        _crop_value(tile, crop_profiles) for tile in destroyed)
    fixed_bonus = sum(
        0.5 * _crop_value(tile, crop_profiles) for tile in protected)
    return living_value - 1.5 * destroyed_value + fixed_bonus


def _mode6_actions(all_crops, protected, destroyed, crop_profiles=None):
    actions = sorted(set(all_crops) - set(protected) - set(destroyed))
    if len(actions) <= MODE6_MAX_BRANCHING:
        return actions
    crop_profiles = crop_profiles or {}
    return sorted(
        actions,
        key=lambda tile: (
            -_crop_value(tile, crop_profiles),
            -_crop_risk(tile, crop_profiles),
            tile,
        ),
    )[:MODE6_MAX_BRANCHING]


def _tree_result(algorithm, move, response, value, counters, root_values,
                 depth, expected=False, probability=None, event=None,
                 alpha=None, beta=None):
    info = {
        "alpha": "-inf" if alpha is None else f"{alpha:.1f}",
        "beta": "+inf" if beta is None else f"{beta:.1f}",
        "pruned": counters["pruned"],
    }
    stats = {
        "Algorithm": algorithm,
        "Depth": depth,
        "Candidate cap": MODE6_MAX_BRANCHING,
        "Best Move": f"{move}",
        "Expected Utility" if expected else "Utility": f"{value:.1f}",
        "MAX nodes": counters["max"],
        "MIN nodes": counters["min"],
        "CHANCE nodes": counters["chance"],
        "Nodes evaluated": counters["evaluated"],
        "Pruned Nodes": counters["pruned"],
    }
    if response is not None:
        stats["Enemy Move" if not expected else "Chance Target"] = f"{response}"
    if probability is not None:
        stats["Probability"] = f"{probability * 100:.0f}%"
    if event:
        stats["Chance Event"] = event
    if algorithm == "Alpha-Beta":
        stats["Alpha"] = info["alpha"]
        stats["Beta"] = info["beta"]
    details = {
        "root_values": root_values,
        "pruned": counters["pruned"],
        "best_move": move,
        "response": response,
        "event": event,
        "probability": probability,
        "depth": depth,
    }
    return move, response, value, info, stats, details


def _minimax_search(all_crops, protected, destroyed, depth, maximizing,
                    counters, memo, crop_profiles):
    key = (frozenset(protected), frozenset(destroyed), depth, maximizing)
    if key in memo:
        return memo[key]
    counters["evaluated"] += 1
    actions = _mode6_actions(all_crops, protected, destroyed, crop_profiles)
    if depth == 0 or not actions:
        result = evaluate_farm_state(
            all_crops, protected, destroyed, crop_profiles), []
        memo[key] = result
        return result

    if maximizing:
        counters["max"] += 1
        best_value = -INF
        best_line = []
        for tile in actions:
            value, line = _minimax_search(
                all_crops, protected | {tile}, destroyed,
                depth - 1, False, counters, memo, crop_profiles)
            if value > best_value:
                best_value = value
                best_line = [tile] + line
        memo[key] = (best_value, best_line)
        return memo[key]

    counters["min"] += 1
    best_value = INF
    best_line = []
    for tile in actions:
        value, line = _minimax_search(
            all_crops, protected, destroyed | {tile},
            depth - 1, True, counters, memo, crop_profiles)
        if value < best_value:
            best_value = value
            best_line = [tile] + line
    memo[key] = (best_value, best_line)
    return memo[key]


def minimax_choose(all_crops, protected, destroyed, depth=MODE6_SEARCH_DEPTH,
                   crop_profiles=None):
    crop_profiles = crop_profiles or {}
    counters = {"max": 0, "min": 0, "chance": 0,
                "evaluated": 0, "pruned": 0}
    root_values = []
    memo = {}
    best_move = None
    best_response = None
    best_value = -INF
    for move in _mode6_actions(all_crops, protected, destroyed, crop_profiles):
        value, line = _minimax_search(
            all_crops, set(protected) | {move}, set(destroyed),
            depth - 1, False, counters, memo, crop_profiles)
        root_values.append((move, value, False))
        if value > best_value:
            best_value = value
            best_move = move
            best_response = line[0] if line else None
    return _tree_result(
        "Minimax", best_move, best_response, best_value, counters,
        root_values, depth)


def _alpha_beta_search(all_crops, protected, destroyed, depth, maximizing,
                       alpha, beta, counters, crop_profiles):
    counters["evaluated"] += 1
    actions = _mode6_actions(all_crops, protected, destroyed, crop_profiles)
    if depth == 0 or not actions:
        return evaluate_farm_state(
            all_crops, protected, destroyed, crop_profiles), []

    if maximizing:
        counters["max"] += 1
        value = -INF
        best_line = []
        for index, tile in enumerate(actions):
            child, line = _alpha_beta_search(
                all_crops, protected | {tile}, destroyed, depth - 1,
                False, alpha, beta, counters, crop_profiles)
            if child > value:
                value = child
                best_line = [tile] + line
            alpha = max(alpha, value)
            if beta <= alpha:
                counters["pruned"] += len(actions) - index - 1
                counters["cutoff_alpha"] = alpha
                counters["cutoff_beta"] = beta
                break
        return value, best_line

    counters["min"] += 1
    value = INF
    best_line = []
    for index, tile in enumerate(actions):
        child, line = _alpha_beta_search(
            all_crops, protected, destroyed | {tile}, depth - 1,
            True, alpha, beta, counters, crop_profiles)
        if child < value:
            value = child
            best_line = [tile] + line
        beta = min(beta, value)
        if beta <= alpha:
            counters["pruned"] += len(actions) - index - 1
            counters["cutoff_alpha"] = alpha
            counters["cutoff_beta"] = beta
            break
    return value, best_line


def alpha_beta_choose(all_crops, protected, destroyed,
                      depth=MODE6_SEARCH_DEPTH, crop_profiles=None):
    crop_profiles = crop_profiles or {}
    counters = {"max": 0, "min": 0, "chance": 0,
                "evaluated": 0, "pruned": 0}
    root_values = []
    best_move = None
    best_response = None
    best_value = -INF
    alpha = -INF
    beta = INF
    for move in _mode6_actions(all_crops, protected, destroyed, crop_profiles):
        value, line = _alpha_beta_search(
            all_crops, set(protected) | {move}, set(destroyed),
            depth - 1, False, alpha, beta, counters, crop_profiles)
        root_values.append((move, value, False))
        if value > best_value:
            best_value = value
            best_move = move
            best_response = line[0] if line else None
        alpha = max(alpha, best_value)
    return _tree_result(
        "Alpha-Beta", best_move, best_response, best_value, counters,
        root_values, depth,
        alpha=counters.get("cutoff_alpha", alpha),
        beta=counters.get("cutoff_beta", beta))


def _expectimax_search(all_crops, protected, destroyed, depth, maximizing,
                       counters, memo, crop_profiles):
    key = (frozenset(protected), frozenset(destroyed), depth, maximizing)
    if key in memo:
        return memo[key]
    counters["evaluated"] += 1
    actions = _mode6_actions(all_crops, protected, destroyed, crop_profiles)
    if depth == 0 or not actions:
        result = evaluate_farm_state(
            all_crops, protected, destroyed, crop_profiles), []
        memo[key] = result
        return result

    if maximizing:
        counters["max"] += 1
        best_value = -INF
        best_line = []
        for tile in actions:
            value, line = _expectimax_search(
                all_crops, protected | {tile}, destroyed,
                depth - 1, False, counters, memo, crop_profiles)
            if value > best_value:
                best_value = value
                best_line = [tile] + line
        memo[key] = (best_value, best_line)
        return memo[key]

    counters["chance"] += 1
    risk_total = sum(_crop_risk(tile, crop_profiles) for tile in actions)
    expected = 0.0
    most_likely_line = []
    most_likely_probability = -1.0
    for tile in actions:
        probability = (
            _crop_risk(tile, crop_profiles) / risk_total
            if risk_total > 0 else 1.0 / len(actions))
        value, line = _expectimax_search(
            all_crops, protected, destroyed | {tile},
            depth - 1, True, counters, memo, crop_profiles)
        expected += probability * value
        if probability > most_likely_probability:
            most_likely_probability = probability
            most_likely_line = [tile] + line
    memo[key] = (expected, most_likely_line)
    return memo[key]


def expectimax_choose(all_crops, protected, destroyed,
                      depth=MODE6_SEARCH_DEPTH, crop_profiles=None):
    crop_profiles = crop_profiles or {}
    counters = {"max": 0, "min": 0, "chance": 0,
                "evaluated": 0, "pruned": 0}
    root_values = []
    memo = {}
    best_move = None
    best_value = -INF
    best_response = None
    for move in _mode6_actions(all_crops, protected, destroyed, crop_profiles):
        value, line = _expectimax_search(
            all_crops, set(protected) | {move}, set(destroyed),
            depth - 1, False, counters, memo, crop_profiles)
        root_values.append((move, value, False))
        if value > best_value:
            best_value = value
            best_move = move
            best_response = line[0] if line else None
    chance_actions = _mode6_actions(
        all_crops, set(protected) | ({best_move} if best_move else set()),
        destroyed, crop_profiles)
    risk_total = sum(
        _crop_risk(tile, crop_profiles) for tile in chance_actions)
    probability = (
        _crop_risk(best_response, crop_profiles) / risk_total
        if best_response is not None and risk_total > 0 else 0.0)
    event = max(EXPECTIMAX_EVENTS, key=lambda item: item[1])[0]
    return _tree_result(
        "Expectimax", best_move, best_response, best_value, counters,
        root_values, depth, expected=True, probability=probability,
        event=event)


def _expectiminimax_search(all_crops, protected, destroyed, depth, node_type,
                           counters, memo, crop_profiles):
    key = (frozenset(protected), frozenset(destroyed), depth, node_type)
    if key in memo:
        return memo[key]
    counters["evaluated"] += 1
    actions = _mode6_actions(all_crops, protected, destroyed, crop_profiles)
    if depth == 0 or not actions:
        result = evaluate_farm_state(
            all_crops, protected, destroyed, crop_profiles), []
        memo[key] = result
        return result

    if node_type == "MAX":
        counters["max"] += 1
        best_value = -INF
        best_line = []
        for tile in actions:
            value, line = _expectiminimax_search(
                all_crops, protected | {tile}, destroyed,
                depth - 1, "MIN", counters, memo, crop_profiles)
            if value > best_value:
                best_value = value
                best_line = [tile] + line
        memo[key] = (best_value, best_line)
        return memo[key]

    if node_type == "MIN":
        counters["min"] += 1
        best_value = INF
        best_line = []
        for tile in actions:
            value, line = _expectiminimax_search(
                all_crops, protected, destroyed,
                depth - 1, ("CHANCE", tile), counters, memo, crop_profiles)
            if value < best_value:
                best_value = value
                best_line = [tile] + line
        memo[key] = (best_value, best_line)
        return memo[key]

    counters["chance"] += 1
    enemy_target = node_type[1]
    target_risk = _crop_risk(enemy_target, crop_profiles)
    sabotage_probability = 0.60
    weather_probability = 0.40 * target_risk
    safe_probability = 1.0 - sabotage_probability - weather_probability
    success_value, success_line = _expectiminimax_search(
        all_crops, protected, destroyed | {enemy_target},
        depth - 1, "MAX", counters, memo, crop_profiles)
    no_damage_value, no_damage_line = _expectiminimax_search(
        all_crops, protected, destroyed,
        depth - 1, "MAX", counters, memo, crop_profiles)
    weather_targets = [
        tile for tile in actions if tile != enemy_target
    ]
    if weather_targets:
        weather_value = 0.0
        weather_risk_total = sum(
            _crop_risk(tile, crop_profiles) for tile in weather_targets)
        for tile in weather_targets:
            tile_probability = (
                _crop_risk(tile, crop_profiles) / weather_risk_total
                if weather_risk_total > 0
                else 1.0 / len(weather_targets))
            child, _ = _expectiminimax_search(
                all_crops, protected, destroyed | {tile},
                depth - 1, "MAX", counters, memo, crop_profiles)
            weather_value += tile_probability * child
    else:
        weather_value = no_damage_value
    expected = (
        sabotage_probability * success_value
        + weather_probability * weather_value
        + safe_probability * no_damage_value
    )
    memo[key] = (expected, success_line or no_damage_line)
    return memo[key]


def expectiminimax_choose(all_crops, protected, destroyed,
                          depth=MODE6_SEARCH_DEPTH, crop_profiles=None):
    crop_profiles = crop_profiles or {}
    counters = {"max": 0, "min": 0, "chance": 0,
                "evaluated": 0, "pruned": 0}
    root_values = []
    memo = {}
    best_move = None
    best_response = None
    best_value = -INF
    for move in _mode6_actions(all_crops, protected, destroyed, crop_profiles):
        value, line = _expectiminimax_search(
            all_crops, set(protected) | {move}, set(destroyed),
            depth - 1, "MIN", counters, memo, crop_profiles)
        root_values.append((move, value, False))
        if value > best_value:
            best_value = value
            best_move = move
            best_response = line[0] if line else None
    return _tree_result(
        "Expectiminimax", best_move, best_response, best_value, counters,
        root_values, depth, expected=True,
        probability=(
            _crop_risk(best_response, crop_profiles)
            if best_response is not None else 0.0),
        event="MIN -> CHANCE")


def adversarial_choose(algorithm, all_crops, protected, destroyed,
                       depth=MODE6_SEARCH_DEPTH, crop_profiles=None):
    chooser = {
        "Minimax": minimax_choose,
        "Alpha-Beta": alpha_beta_choose,
        "Expectimax": expectimax_choose,
        "Expectiminimax": expectiminimax_choose,
    }.get(algorithm, minimax_choose)
    return chooser(
        all_crops, protected, destroyed, depth,
        crop_profiles=crop_profiles)
