from collections import deque
import heapq


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path[1:]


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


def astar(start, goal, blocked, neighbors, heuristic, counter):
    open_set = []
    heapq.heappush(open_set, (0, next(counter), start))
    came_from = {}
    g_score = {start: 0}
    explored = set()

    while open_set:
        f_score, _, current = heapq.heappop(open_set)
        explored.add(current)

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
            if tentative_g < g_score.get(next_tile, 10**9):
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
