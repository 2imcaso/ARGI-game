from collections import deque
import heapq
import math
import random


INF = float("inf")

UNINFORMED_ALGORITHMS = ("BFS", "DFS", "IDS", "UCS")
INFORMED_ALGORITHMS = ("Greedy", "IDSA", "A*")
LOCAL_ALGORITHMS = ("Local Beam", "Hill Climbing", "Annealing", "Restart Hill")
ONLINE_ALGORITHMS = (
    "Online A*",
    "Belief A*",
    "Belief Init-Goal A*",
    "AND-OR Search",
)
CSP_ALGORITHMS = ("Backtrack", "Fwd Check", "AC-3", "Min Conflict")

ALGORITHM_GROUPS = {
    1: UNINFORMED_ALGORITHMS,
    2: INFORMED_ALGORITHMS,
    3: LOCAL_ALGORITHMS,
    4: ONLINE_ALGORITHMS,
    5: CSP_ALGORITHMS,
}

ALGORITHM_GROUP_NAMES = {
    1: "Uninformed Search",
    2: "Informed Search",
    3: "Local Search",
    4: "Online Search",
    5: "Constraint Satisfaction",
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
    heapq.heappush(open_set, (0, next(counter), start))
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
        h = heuristic(start, goal)
        return path, explored, (len(path) + h, len(path), h), stats

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


def hill_score(tile, start, dryness, heuristic):
    return float(dryness.get(tile, 50))


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
        better_neighbors = [
            tile for tile in local_neighbors
            if scores.get(tile, 0) > current_score
        ]
        if not better_neighbors:
            break

        best_neighbor = max(
            better_neighbors,
            key=lambda tile: (scores.get(tile, 0), -heuristic(start, tile), tile))
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
    if not plan:
        return float("-inf")
    first_target_score = scores.get(plan[0], 0)
    return first_target_score * 1000 + _plan_value(
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
        best_neighbor = max(
            successors,
            key=lambda plan: (_plan_objective(
                                  plan, start, dryness, heuristic, scores),
                              tuple(plan)))
        best_value = _plan_objective(
            best_neighbor, start, dryness, heuristic, scores)
        if best_value <= current_value:
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
    best_value = float("-inf")
    for plan in starts[:max(1, restarts)]:
        candidate_plan, candidate_value, _ = _plan_hill_climb(
            plan, start, dryness, heuristic, scores)
        if candidate_value > best_value:
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
            key=lambda tile: (-dryness.get(tile, 50),
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
            key=lambda plan: (-_plan_objective(
                                  plan, start, dryness, heuristic, scores),
                              tuple(plan)))[:beam_size]
        if {tuple(plan) for plan in next_beam} == {tuple(plan) for plan in beam}:
            break
        beam = next_beam

    best_plan = min(
        beam,
        key=lambda plan: (-_plan_objective(
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
            "Dryness": f"{dryness.get(current, 0)}%",
            "Hill steps": hill_steps,
            "Trace": trace,
            "Stop rule": "No better neighbor",
        }
    if algorithm == "Annealing":
        return simulated_annealing_choose(remaining, start, dryness, heuristic)
    if algorithm == "Restart Hill":
        best_tile = None
        best_score = float("-inf")
        best_trace = []
        starts = random.sample(list(remaining), min(len(remaining), 8))
        for restart_tile in starts:
            candidate, candidate_score, _, _, trace = hill_climbing_choose(
                remaining, start, dryness, heuristic,
                initial_tile=restart_tile)
            if candidate_score > best_score:
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
            "Stop rule": "No better neighbor",
        }

    current, current_score, scores, hill_steps, trace = hill_climbing_choose(
        remaining, start, dryness, heuristic)
    return current, current_score, scores, {
        "Local algorithm": algorithm,
        "Current score": f"{current_score:.0f}",
        "Target": f"{current}",
        "Dryness": f"{dryness.get(current, 0)}%",
        "Hill steps": hill_steps,
        "Trace": trace,
    }


def expand_vision(center, radius, rows, cols, hidden_blocked,
                  explored_tiles, discovered_blocked):
    cx, cy = center
    scan_radius = int(radius) + 1
    for dy in range(-scan_radius, scan_radius + 1):
        for dx in range(-scan_radius, scan_radius + 1):
            if dx * dx + dy * dy > radius * radius:
                continue
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                tile = (nx, ny)
                explored_tiles.add(tile)
                if tile in hidden_blocked:
                    discovered_blocked.add(tile)


def belief_astar(start, goal, blocked, uncertain_tiles, neighbors, heuristic,
                 counter, uncertainty_cost=2.0):
    """A* over a belief map where unobserved tiles carry extra risk cost."""
    open_set = [(heuristic(start, goal), next(counter), start)]
    came_from = {}
    best_cost = {start: 0.0}
    explored = set()

    while open_set:
        _, _, current = heapq.heappop(open_set)
        explored.add(current)
        if current == goal:
            path = reconstruct_path(came_from, current)
            uncertain_steps = sum(tile in uncertain_tiles for tile in path)
            return path, explored, {
                "Algorithm": "Belief A*",
                "Belief states": len(uncertain_tiles),
                "Uncertain steps": uncertain_steps,
                "Path length": len(path),
                "Risk cost": f"{best_cost[current] - len(path):.1f}",
            }

        for next_tile in neighbors(current, blocked):
            risk = uncertainty_cost if next_tile in uncertain_tiles else 0.0
            next_cost = best_cost[current] + 1.0 + risk
            if next_cost >= best_cost.get(next_tile, INF):
                continue
            best_cost[next_tile] = next_cost
            came_from[next_tile] = current
            priority = next_cost + heuristic(next_tile, goal)
            heapq.heappush(
                open_set, (priority, next(counter), next_tile)
            )

    return [], explored, {
        "Algorithm": "Belief A*",
        "Belief states": len(uncertain_tiles),
        "Uncertain steps": 0,
        "Path length": 0,
        "Risk cost": "inf",
    }


def belief_init_goal_astar(initial_belief, goal_belief, blocked,
                           uncertain_tiles, neighbors, heuristic, counter,
                           uncertainty_cost=2.0, preferred_start=None):
    """Risk-aware A* from a set of possible starts to a set of goals."""
    starts = {
        state for state in initial_belief
        if state not in blocked
    }
    goals = {
        state for state in goal_belief
        if state not in blocked
    }
    if preferred_start in starts:
        starts = {preferred_start}

    if not starts or not goals:
        return [], set(), None, None, {
            "Algorithm": "Belief Init-Goal A*",
            "Initial belief": len(initial_belief),
            "Goal belief": len(goal_belief),
            "Path length": 0,
            "Risk cost": "inf",
        }

    def goal_distance(tile):
        return min(heuristic(tile, goal) for goal in goals)

    open_set = []
    came_from = {}
    source = {}
    best_cost = {}
    for start in starts:
        best_cost[start] = 0.0
        source[start] = start
        heapq.heappush(
            open_set, (goal_distance(start), next(counter), start)
        )

    explored = set()
    while open_set:
        _, _, current = heapq.heappop(open_set)
        explored.add(current)
        if current in goals:
            path = reconstruct_path(came_from, current)
            uncertain_steps = sum(tile in uncertain_tiles for tile in path)
            return path, explored, source[current], current, {
                "Algorithm": "Belief Init-Goal A*",
                "Initial belief": len(initial_belief),
                "Goal belief": len(goal_belief),
                "Chosen start": source[current],
                "Chosen goal": current,
                "Uncertain steps": uncertain_steps,
                "Path length": len(path),
                "Risk cost": f"{best_cost[current] - len(path):.1f}",
            }

        for next_tile in neighbors(current, blocked):
            risk = uncertainty_cost if next_tile in uncertain_tiles else 0.0
            next_cost = best_cost[current] + 1.0 + risk
            if next_cost >= best_cost.get(next_tile, INF):
                continue
            best_cost[next_tile] = next_cost
            came_from[next_tile] = current
            source[next_tile] = source[current]
            priority = next_cost + goal_distance(next_tile)
            heapq.heappush(
                open_set, (priority, next(counter), next_tile)
            )

    return [], explored, None, None, {
        "Algorithm": "Belief Init-Goal A*",
        "Initial belief": len(initial_belief),
        "Goal belief": len(goal_belief),
        "Path length": 0,
        "Risk cost": "inf",
    }


def and_or_search(start, goal, blocked, uncertain_tiles, neighbors,
                  heuristic, counter, max_depth=4, failure_penalty=1.0):
    """Recursive AND-OR tree search that builds a full contingent plan.

    OR  nodes  (robot's choice): pick the action minimising worst-case cost.
    AND nodes  (nature's choice): ALL four stochastic-drift outcomes must be
                                  handled — if any branch has no plan the
                                  action is rejected.

    Stochastic drift model: 31 % intended direction, 23 % each other.

    Returns
    -------
    path    : intended (success-branch) tile list, for visualisation.
    policy  : dict {state -> next_tile} — the full contingent plan the
              controller can follow on any observed outcome without replanning.
    explored: set of states visited during the search.
    stats   : dict of algorithm statistics.
    """
    INF = float("inf")
    DIRECTIONS = ((0, -1), (0, 1), (-1, 0), (1, 0))
    explored = set()
    ancestors: set = set()          # backtracking stack for cycle detection
    _nbr_cache: dict = {}           # neighbors(state) cache

    def valid_set(state):
        if state not in _nbr_cache:
            _nbr_cache[state] = set(neighbors(state, blocked))
        return _nbr_cache[state]

    def walkable_outcome(state, direction):
        """Tile reached after moving in direction; stays if wall."""
        cand = (state[0] + direction[0], state[1] + direction[1])
        return cand if cand in valid_set(state) else state

    # ------------------------------------------------------------------ OR node
    def or_node(state, depth):
        """Robot chooses the best action.  Returns (policy_fragment, worst_cost)."""
        explored.add(state)

        if state == goal:
            return {}, 0.0

        if state in ancestors or depth == 0:
            # Cycle detected, or depth exhausted → no guarantee.
            return None, INF

        ancestors.add(state)
        best_policy, best_cost = None, INF

        for action in valid_set(state):
            sub_policy, cost = and_node(state, action, depth - 1)
            if sub_policy is not None and cost < best_cost:
                best_cost = cost
                best_policy = {state: action}
                best_policy.update(sub_policy)

        ancestors.discard(state)
        return best_policy, best_cost

    # ----------------------------------------------------------------- AND node
    def and_node(state, intended, depth):
        """Nature picks a drift direction.
        ALL four outcomes must be plannable; worst-case cost is returned."""
        merged: dict = {}
        worst = 0.0

        for direction in DIRECTIONS:
            outcome = walkable_outcome(state, direction)

            sub_policy, sub_cost = or_node(outcome, depth)
            if sub_policy is None:
                return None, INF        # one branch unhandled → reject action

            # Merge per-state action entries (later may overwrite earlier
            # for the same state, which is acceptable in a flat policy dict).
            merged.update(sub_policy)
            worst = max(worst, 1.0 + sub_cost)

        return merged, worst

    # ------------------------------------------------------------------ search
    policy, cost = or_node(start, max_depth)

    # Extract intended path (follow policy along the success branch).
    path: list = []
    current = start
    seen = {start}
    while current != goal and policy and current in policy:
        nxt = policy[current]
        if nxt is None or nxt in seen:
            break
        path.append(nxt)
        seen.add(nxt)
        current = nxt

    stats = {
        "Algorithm": "AND-OR Search",
        "Tree depth": max_depth,
        "OR states explored": len(explored),
        "Policy size": len(policy) if policy else 0,
        "Worst-case cost": "inf" if cost == INF else f"{cost:.1f}",
        "Path length": len(path),
        "Contingent plan": "yes" if policy else "no",
        "Failure policy": "Follow policy on drift; replan if off-policy",
    }

    if not policy:
        return [], {}, explored, stats

    return path, policy, explored, stats


def _csp_adjacent(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def solve_csp_crop_plan(farm_tiles, algorithm="Backtrack"):
    """Assign crops using the selected CSP solving strategy."""
    variables = list(farm_tiles)

    steps = []
    backtracks = 0
    arc_checks = 0
    adjacency = {
        var: [other for other in variables if _csp_adjacent(var, other)]
        for var in variables
    }

    def log(event, var, value):
        if len(steps) < 200:
            steps.append((event, var, value))

    def valid(var, value, assignment):
        return all(
            assignment.get(neighbor) != value
            for neighbor in adjacency[var]
        )

    def select_unassigned(assignment, domains):
        remaining = [var for var in variables if var not in assignment]
        return min(
            remaining,
            key=lambda var: (len(domains[var]), -len(adjacency[var])),
        )

    def backtrack(assignment, domains, forward_check=False):
        nonlocal backtracks
        if len(assignment) == len(variables):
            return dict(assignment)

        var = select_unassigned(assignment, domains)
        for value in list(domains[var]):
            log("try", var, value)
            if not valid(var, value, assignment):
                continue

            next_assignment = dict(assignment)
            next_assignment[var] = value
            next_domains = {
                key: list(values) for key, values in domains.items()
            }
            log("assign", var, value)

            consistent = True
            if forward_check:
                for neighbor in adjacency[var]:
                    if neighbor in next_assignment:
                        continue
                    if value in next_domains[neighbor]:
                        next_domains[neighbor].remove(value)
                    if not next_domains[neighbor]:
                        consistent = False
                        break

            if consistent:
                result = backtrack(
                    next_assignment, next_domains, forward_check=forward_check
                )
                if result is not None:
                    return result

            backtracks += 1
            log("backtrack", var, value)
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
                    left_value != right_value
                    for right_value in domains[right]
                )
            ]
            if len(supported) == len(domains[left]):
                continue
            domains[left] = supported
            if not domains[left]:
                return False
            for neighbor in adjacency[left]:
                if neighbor != right:
                    queue.append((neighbor, left))
        return True

    def min_conflicts(max_steps=200):
        nonlocal backtracks
        if not variables:
            return {}

        assignment = {
            var: random.choice(("corn", "tomato")) for var in variables
        }
        log("init", None, None)
        for _ in range(max_steps):
            conflicted = [
                var for var in variables
                if any(
                    assignment[neighbor] == assignment[var]
                    for neighbor in adjacency[var]
                )
            ]
            if not conflicted:
                return assignment

            var = random.choice(conflicted)
            counts = {
                value: sum(
                    assignment[neighbor] == value
                    for neighbor in adjacency[var]
                )
                for value in ("corn", "tomato")
            }
            best_count = min(counts.values())
            best_values = [
                value for value, count in counts.items()
                if count == best_count
            ]
            new_value = random.choice(best_values)
            if assignment[var] != new_value:
                backtracks += 1
            assignment[var] = new_value
            log("assign", var, new_value)
        return None

    domains = {var: ["corn", "tomato"] for var in variables}
    if algorithm == "Fwd Check":
        result = backtrack({}, domains, forward_check=True)
    elif algorithm == "AC-3":
        result = (
            backtrack({}, domains, forward_check=True)
            if enforce_arc_consistency(domains)
            else None
        )
    elif algorithm == "Min Conflict":
        result = min_conflicts()
        if result is None:
            result = backtrack({}, domains, forward_check=True)
    else:
        result = backtrack({}, domains)

    stats = {
        "Algorithm": algorithm,
        "Variables": len(variables),
        "Domain": "{corn, tomato}",
        "Constraints": "No adjacent same crop",
        "Backtracks": backtracks,
    }
    if algorithm == "AC-3":
        stats["Arc checks"] = arc_checks
    return result or {}, steps, stats


def minimax_choose(remaining, player_start, enemy_tile, enemy_done_tiles,
                   care_value, heuristic, is_living_crop):
    if not remaining or not enemy_tile:
        return (remaining[0] if remaining else None), None, float("-inf"), {
            "alpha": "-inf",
            "beta": "+inf",
            "pruned": 0,
        }, {}

    alpha = float("-inf")
    beta = float("inf")
    best_tile = None
    best_value = float("-inf")
    pruned_count = 0

    for player_choice in remaining:
        crop_score = care_value(player_choice)
        my_cost = heuristic(player_start, player_choice)
        player_value = crop_score - my_cost

        enemy_remaining = [
            tile for tile in remaining
            if tile != player_choice
            and tile not in enemy_done_tiles
            and is_living_crop(tile)
        ]
        min_value = float("inf")
        for enemy_choice in enemy_remaining:
            enemy_cost = heuristic(enemy_tile, enemy_choice)
            enemy_crop = care_value(enemy_choice)
            value = player_value - (enemy_crop - enemy_cost * 0.5)
            min_value = min(min_value, value)
            if min_value <= alpha:
                pruned_count += 1
                break

        if not enemy_remaining:
            min_value = player_value

        if min_value > best_value:
            best_value = min_value
            best_tile = player_choice
        alpha = max(alpha, best_value)

    enemy_choices = [
        tile for tile in remaining
        if tile != best_tile
        and tile not in enemy_done_tiles
        and is_living_crop(tile)
    ]
    enemy_target = (
        min(enemy_choices, key=lambda tile: heuristic(enemy_tile, tile))
        if enemy_choices else None
    )
    alpha_beta_info = {
        "alpha": f"{alpha:.1f}",
        "beta": f"{beta:.1f}" if beta < float("inf") else "+inf",
        "pruned": pruned_count,
    }
    stats = {
        "Minimax value": f"{best_value:.1f}",
        "Alpha": alpha_beta_info["alpha"],
        "Beta": alpha_beta_info["beta"],
        "Pruned branches": f"{pruned_count}",
        "Enemy target": f"{enemy_target}",
    }
    return best_tile, enemy_target, best_value, alpha_beta_info, stats
