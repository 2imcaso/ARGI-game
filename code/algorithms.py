from collections import deque
import heapq
import random


INF = float("inf")

UNINFORMED_ALGORITHMS = ("BFS", "DFS", "IDS", "UCS")
INFORMED_ALGORITHMS = ("Greedy", "IDSA", "A*")
LOCAL_ALGORITHMS = ("Hill", "Hill Best", "Stochastic", "Restart Hill")

ALGORITHM_GROUPS = {
    1: UNINFORMED_ALGORITHMS,
    2: INFORMED_ALGORITHMS,
    3: LOCAL_ALGORITHMS,
}

ALGORITHM_GROUP_NAMES = {
    1: "Uninformed Search",
    2: "Informed Search",
    3: "Local Search",
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
    stack = [start]
    came_from = {}
    visited = {start}
    explored = set()

    while stack:
        current = stack.pop()
        explored.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, explored, {
                "Nodes explored": len(explored),
                "Path length": len(path),
                "Stack max": len(visited),
            }

        for next_tile in neighbors(current, blocked):
            if next_tile not in visited:
                visited.add(next_tile)
                came_from[next_tile] = current
                stack.append(next_tile)

    return [], explored, {
        "Nodes explored": len(explored),
        "Path length": 0,
        "Stack max": len(visited),
    }


def _depth_limited_search(start, goal, blocked, neighbors, limit):
    stack = [(start, 0)]
    came_from = {}
    best_depth = {start: 0}
    explored = set()
    cutoff = False

    while stack:
        current, depth = stack.pop()
        explored.add(current)

        if current == goal:
            return reconstruct_path(came_from, current), explored, False

        if depth >= limit:
            cutoff = True
            continue

        for next_tile in neighbors(current, blocked):
            next_depth = depth + 1
            if next_depth < best_depth.get(next_tile, INF):
                best_depth[next_tile] = next_depth
                came_from[next_tile] = current
                stack.append((next_tile, next_depth))

    return [], explored, cutoff


def ids(start, goal, blocked, neighbors, max_depth=200):
    total_explored = set()
    for limit in range(max_depth + 1):
        path, explored, cutoff = _depth_limited_search(
            start, goal, blocked, neighbors, limit)
        total_explored |= explored
        if path or start == goal:
            return path, total_explored, {
                "Nodes explored": len(total_explored),
                "Path length": len(path),
                "Depth limit": limit,
            }
        if not cutoff:
            break

    return [], total_explored, {
        "Nodes explored": len(total_explored),
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


def astar(start, goal, blocked, neighbors, heuristic, counter):
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
            tentative_g = g_score[current] + 1
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


def idastar(start, goal, blocked, neighbors, heuristic):
    limit = heuristic(start, goal)
    total_explored = set()

    def search(path, g_score, bound, visited):
        current = path[-1]
        f_score = g_score + heuristic(current, goal)
        if f_score > bound:
            return f_score, None
        total_explored.add(current)
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

    while limit < INF:
        next_limit, path = search([start], 0, limit, {start})
        if path is not None:
            return path, total_explored, {
                "Nodes explored": len(total_explored),
                "Path length": len(path),
                "f limit": limit,
            }
        if next_limit == INF:
            break
        limit = next_limit

    return [], total_explored, {
        "Nodes explored": len(total_explored),
        "Path length": 0,
        "f limit": limit,
    }


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
        start, goal, blocked, neighbors, heuristic, counter)
    return path, explored, fgh, stats


def _nearest_goal_heuristic(tile, goals, heuristic):
    if not goals:
        return 0
    return min(heuristic(tile, goal) for goal in goals)


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
    stack = [start]
    came_from = {}
    visited = {start}
    explored = set()

    while stack:
        current = stack.pop()
        explored.add(current)

        if current in goals:
            path = reconstruct_path(came_from, current)
            return path, current, explored, {
                "Nodes explored": len(explored),
                "Path length": len(path),
                "Stack max": len(visited),
            }

        for next_tile in neighbors(current, blocked):
            if next_tile not in visited:
                visited.add(next_tile)
                came_from[next_tile] = current
                stack.append(next_tile)

    return [], None, explored, {
        "Nodes explored": len(explored),
        "Path length": 0,
        "Stack max": len(visited),
    }


def _depth_limited_search_to_any_goal(start, goals, blocked, neighbors, limit):
    goals = set(goals)
    stack = [(start, 0)]
    came_from = {}
    best_depth = {start: 0}
    explored = set()
    cutoff = False

    while stack:
        current, depth = stack.pop()
        explored.add(current)

        if current in goals:
            return reconstruct_path(came_from, current), current, explored, False

        if depth >= limit:
            cutoff = True
            continue

        for next_tile in neighbors(current, blocked):
            next_depth = depth + 1
            if next_depth < best_depth.get(next_tile, INF):
                best_depth[next_tile] = next_depth
                came_from[next_tile] = current
                stack.append((next_tile, next_depth))

    return [], None, explored, cutoff


def ids_to_any_goal(start, goals, blocked, neighbors, max_depth=200):
    total_explored = set()
    for limit in range(max_depth + 1):
        path, target, explored, cutoff = _depth_limited_search_to_any_goal(
            start, goals, blocked, neighbors, limit)
        total_explored |= explored
        if target is not None:
            return path, target, total_explored, {
                "Nodes explored": len(total_explored),
                "Path length": len(path),
                "Depth limit": limit,
            }
        if not cutoff:
            break

    return [], None, total_explored, {
        "Nodes explored": len(total_explored),
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


def greedy_to_any_goal(start, goals, blocked, neighbors, heuristic, counter):
    goals = set(goals)
    open_set = []
    heapq.heappush(
        open_set,
        (_nearest_goal_heuristic(start, goals, heuristic), next(counter), start))
    came_from = {}
    visited = {start}
    explored = set()

    while open_set:
        priority, _, current = heapq.heappop(open_set)
        explored.add(current)

        if current in goals:
            path = reconstruct_path(came_from, current)
            return path, current, explored, {
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
                    (_nearest_goal_heuristic(next_tile, goals, heuristic),
                     next(counter), next_tile))

    return [], None, explored, {
        "Nodes explored": len(explored),
        "Path length": 0,
    }


def astar_to_any_goal(start, goals, blocked, neighbors, heuristic, counter):
    goals = set(goals)
    open_set = []
    start_h = _nearest_goal_heuristic(start, goals, heuristic)
    heapq.heappush(open_set, (start_h, next(counter), start))
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

        if current in goals:
            path = reconstruct_path(came_from, current)
            g = g_score[current]
            h = 0
            return path, current, explored, (g + h, g, h), {
                "Nodes explored": len(explored),
                "Path length": len(path),
                "f(n)": f"{g + h}",
                "g(n)": f"{g}",
                "h(n)": f"{h}",
            }

        for next_tile in neighbors(current, blocked):
            tentative_g = g_score[current] + 1
            if next_tile in closed:
                continue
            if tentative_g < g_score.get(next_tile, INF):
                came_from[next_tile] = current
                g_score[next_tile] = tentative_g
                h = _nearest_goal_heuristic(next_tile, goals, heuristic)
                heapq.heappush(
                    open_set,
                    (tentative_g + h, next(counter), next_tile))

    return [], None, explored, (0, 0, 0), {
        "Nodes explored": len(explored),
        "Path length": 0,
    }


def idastar_to_any_goal(start, goals, blocked, neighbors, heuristic):
    goals = set(goals)
    limit = _nearest_goal_heuristic(start, goals, heuristic)
    total_explored = set()

    def search(path, g_score, bound, visited):
        current = path[-1]
        h = _nearest_goal_heuristic(current, goals, heuristic)
        f_score = g_score + h
        if f_score > bound:
            return f_score, None
        total_explored.add(current)
        if current in goals:
            return f_score, list(path[1:])

        next_bound = INF
        ordered_neighbors = sorted(
            neighbors(current, blocked),
            key=lambda tile: _nearest_goal_heuristic(tile, goals, heuristic))
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

    while limit < INF:
        next_limit, path = search([start], 0, limit, {start})
        if path is not None:
            target = path[-1] if path else start
            return path, target, total_explored, {
                "Nodes explored": len(total_explored),
                "Path length": len(path),
                "f limit": limit,
            }
        if next_limit == INF:
            break
        limit = next_limit

    return [], None, total_explored, {
        "Nodes explored": len(total_explored),
        "Path length": 0,
        "f limit": limit,
    }


def find_path_to_any_goal_by_algorithm(algorithm, start, goals, blocked,
                                       neighbors, heuristic, counter,
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
    if algorithm == "Greedy":
        path, target, explored, stats = greedy_to_any_goal(
            start, goals, blocked, neighbors, heuristic, counter)
        h = _nearest_goal_heuristic(start, goals, heuristic)
        return path, target, explored, (h, 0, h), stats
    if algorithm == "IDSA":
        path, target, explored, stats = idastar_to_any_goal(
            start, goals, blocked, neighbors, heuristic)
        h = _nearest_goal_heuristic(start, goals, heuristic)
        return path, target, explored, (len(path) + h, len(path), h), stats

    path, target, explored, fgh, stats = astar_to_any_goal(
        start, goals, blocked, neighbors, heuristic, counter)
    return path, target, explored, fgh, stats


def hill_score(tile, start, dryness, heuristic):
    dry = dryness.get(tile, 50)
    dist = heuristic(start, tile)
    return dry - 6 * dist


def hill_climbing_choose(remaining, start, dryness, heuristic):
    scores = {
        tile: hill_score(tile, start, dryness, heuristic)
        for tile in remaining
    }
    current_best = min(remaining, key=lambda tile: heuristic(start, tile))
    current_score = scores.get(current_best, 0)

    improved = True
    while improved:
        improved = False
        cx, cy = current_best
        local_neighbors = [
            tile for tile in remaining
            if abs(tile[0] - cx) + abs(tile[1] - cy) <= 2
            and tile != current_best
        ]
        if not local_neighbors:
            break

        best_neighbor = max(local_neighbors, key=lambda tile: scores.get(tile, 0))
        best_neighbor_score = scores.get(best_neighbor, 0)
        if best_neighbor_score > current_score:
            current_best = best_neighbor
            current_score = best_neighbor_score
            improved = True

    return current_best, current_score, scores


def local_search_choose(algorithm, remaining, start, dryness, heuristic):
    scores = {
        tile: hill_score(tile, start, dryness, heuristic)
        for tile in remaining
    }
    if not remaining:
        return None, 0, scores, {
            "Local algorithm": algorithm,
            "Target": "None",
        }

    current = min(remaining, key=lambda tile: heuristic(start, tile))
    current_score = scores.get(current, 0)
    visited = {current}
    restarts = 1

    if algorithm == "Hill":
        for tile in sorted(remaining, key=lambda tile: heuristic(current, tile)):
            if tile == current:
                continue
            score = scores.get(tile, 0)
            if score > current_score:
                current = tile
                current_score = score
                break
    elif algorithm == "Hill Best":
        current, current_score, scores = hill_climbing_choose(
            remaining, start, dryness, heuristic)
    elif algorithm == "Stochastic":
        improved = True
        while improved:
            improved = False
            better = [
                tile for tile in remaining
                if tile not in visited and scores.get(tile, 0) > current_score
            ]
            if better:
                current = random.choice(better)
                current_score = scores.get(current, 0)
                visited.add(current)
                improved = True
    elif algorithm == "Restart Hill":
        best_tile = current
        best_score = current_score
        starts = random.sample(
            list(remaining), min(len(remaining), 8))
        restarts = len(starts)
        for restart_tile in starts:
            candidate, candidate_score, _ = hill_climbing_choose(
                remaining, restart_tile, dryness, heuristic)
            if candidate_score > best_score:
                best_tile = candidate
                best_score = candidate_score
        current = best_tile
        current_score = best_score

    stats = {
        "Local algorithm": algorithm,
        "Current score": f"{current_score:.0f}",
        "Target": f"{current}",
        "Dryness": f"{dryness.get(current, 0)}%",
    }
    if algorithm == "Restart Hill":
        stats["Restarts"] = restarts
    return current, current_score, scores, stats


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


def solve_csp_crop_plan(farm_tiles):
    variables = list(farm_tiles)
    domains = ["corn", "tomato"]
    assignment = {}
    steps = []

    def adjacent(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def valid(var, value):
        for other, other_value in assignment.items():
            if adjacent(var, other) and value == other_value:
                return False
        return True

    def backtrack(index=0):
        if index == len(variables):
            return True
        var = variables[index]
        for value in domains:
            steps.append(("try", var, value))
            if valid(var, value):
                assignment[var] = value
                steps.append(("assign", var, value))
                if backtrack(index + 1):
                    return True
                assignment.pop(var, None)
                steps.append(("backtrack", var, value))
        return False

    backtrack()
    stats = {
        "Variables": f"{len(variables)}",
        "Domain": "{corn, tomato}",
        "Constraints": "corn/tomato khong ke nhau",
        "Backtrack steps": f"{len([step for step in steps if step[0] == 'backtrack'])}",
    }
    return assignment, steps, stats


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
