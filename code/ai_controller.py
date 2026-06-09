from itertools import count
import math
import os
import random

import pygame

import algorithms
from design_tokens import Colors, MODE_COLORS, MODE_EXPLORE_FILLS
from settings import TILE_SIZE, LAYERS


class FarmAIController:
    """AI demo cho 6 nhĂ³m thuáº­t toĂ¡n trĂªn map gá»‘c Pydew Valley.

    Má»—i mode váº«n dĂ¹ng cĂ¹ng logic gá»‘c: Ä‘i tá»›i Ă´ Ä‘áº¥t -> cuá»‘c -> gieo háº¡t -> tÆ°á»›i.
    KhĂ¡c nhau á»Ÿ cĂ¡ch AI chá»n má»¥c tiĂªu / tĂ¬m Ä‘Æ°á»ng / gĂ¡n loáº¡i cĂ¢y.

    Mode 1: Uninformed Search
    Mode 2: Informed Search
    Mode 3: Local Search
    Mode 4: Online Search â€“ Uncertain Environment (fog-of-war)
    Mode 5: Backtracking  â€“ CSP
    Mode 6: Minimax       â€“ Adversarial Search
    """

    MODE_INFO = {
        1: ("NGAY 1 - KHU 1: VUON TRUOC TRAI", "Uninformed Search",
            "Robot chua biet gi: dung nhom uninformed de tim dat gan nhat",
            "Khu khoi dong sach vat can, khong uu tien huong nao"),
        2: ("NGAY 2 - KHU 2: LOI VAO NHA KHO", "Informed Search",
            "Da biet dich va vat can: dung heuristic de di vong ngan hon",
            "Manh vo nha kho chan duong, can duong toi uu"),
        3: ("NGAY 3 - KHU 3: VUON CAY PHIA DONG", "Local Search",
            "Chon cay co gia tri cham soc cao nhat bang nhom local search",
            "Nhieu cay heo kho, tham lam cuc bo co the bi ket"),
        4: ("NGAY 4 - KHU 4: BAI DAT SUONG MU", "Online Replanning",
            "Chi thay 2 o xung quanh, vua di vua cap nhat va re-plan",
            "Fog-of-war voi o sut/ngap an duoi suong"),
        5: ("NGAY 5 - KHU 5: O QUY HOACH TRUNG TAM", "CSP Backtracking",
            "Gan corn/tomato cho tung o ma khong vi pham rang buoc ke nhau",
            "Ngo va ca chua khong duoc trong canh nhau"),
        6: ("NGAY 6 - KHU 6: HANG CAY CAN BAO VE", "Minimax / Alpha-Beta",
            "Bao ve cay khi doi thu cung toi pha: gia su doi thu choi toi uu",
            "AGRI-1 dau tri voi ke pha hoai tren cac o can bao ve"),
    }

    ENEMY_MOVE_SPEED = 2.2
    ENEMY_DESTROY_TIME = 1.5
    ENEMY_RETARGET_DELAY = 0.08

    # ------------------------------------------------------------------ init
    def __init__(self, player, soil_layer, collision_sprites, farm_tiles,
                 mode=2, hidden_blocks=None, enemy_spawn=None,
                 selected_algorithm=None):
        self.player = player
        self.soil_layer = soil_layer
        self.collision_sprites = collision_sprites
        self.player.auto_control = True

        self.mode = mode
        info = self.MODE_INFO.get(mode, self.MODE_INFO[2])
        self.mode_name = info[0]
        self.algorithm_group_name = info[1]
        self.algorithm_name = selected_algorithm or self._default_algorithm(mode)
        self.mode_desc = info[2]
        self.difficulty_desc = info[3] if len(info) > 3 else ""
        self.farm_tiles = list(farm_tiles)
        self.rows = len(self.soil_layer.grid)
        self.cols = len(self.soil_layer.grid[0]) if self.rows else 0
        self.spawn_tile = self._world_to_tile(self.player.rect.center)
        self.walkable_tiles = self._build_walkable_tiles()

        self.state = "CHOOSE_TARGET"
        self.current_target = None
        self.path = []
        self.chosen_target_path = None
        self.dfs_walk = None
        self.dfs_walk_index = 0
        self.dfs_explored_order = []
        self.done_tiles = set()
        self.wait_time = 0.0
        self.counter = count()
        self.message = f"Khu {mode}: {self.algorithm_name}"
        self.is_running = mode not in (1, 2, 3)
        if not self.is_running:
            self.message = "Chon thuat toan roi nhan START"
        self.algorithm_buttons = {}
        self.control_buttons = {}
        self.visual_assets = self._load_visual_assets()
        self.fog_time = 0.0

        # --- Thá»‘ng kĂª hiá»ƒn thá»‹ chung ---
        self.stats = {}

        # --- Mode 1 (BFS): lÆ°u explored nodes ---
        self.bfs_explored = set()

        # --- Mode 2 (A*): lÆ°u explored + f/g/h hiá»‡n táº¡i ---
        self.astar_explored = set()
        self.astar_current_fgh = (0, 0, 0)  # (f, g, h) má»›i nháº¥t

        # --- Mode 3 (Hill Climbing): dryness duoc gan co chu dich ---
        # Cum gan spawn: dryness thap. Cum xa spawn: dryness cao.
        self.dryness = {}
        if self.mode == 3 and self.farm_tiles:
            spawn_tile = self._world_to_tile(self.player.rect.center)
            dists = [(self._heuristic(spawn_tile, t), t) for t in self.farm_tiles]
            dists.sort()
            n = len(dists)
            near_count = max(1, n // 3)
            for i, (_, t) in enumerate(dists):
                if i < near_count:
                    self.dryness[t] = random.randint(30, 45)
                elif i >= n - near_count:
                    self.dryness[t] = random.randint(70, 95)
                else:
                    self.dryness[t] = random.randint(46, 69)
        else:
            self.dryness = {tile: random.randint(30, 95) for tile in self.farm_tiles}

        self.hc_scores = {}  # tile -> score, Ä‘á»ƒ váº½ lĂªn map
        self.hc_current_score = 0
        self.hc_best_neighbor_score = 0
        self.tile_conditions = self._build_tile_conditions()

        # --- Mode 4 (Online Search): fog-of-war ---
        self.hidden_blocked = set()
        self.discovered_blocked = set()
        self.explored_tiles = set()
        self.vision_radius = 1.8
        self.replan_count = 0
        if self.mode == 4:
            if hidden_blocks is not None:
                # DĂ¹ng hidden blocks cá»‘ Ä‘á»‹nh tá»« Level (Ä‘áº·t lá»‡ch hĂ ng + cá»™t)
                for t in hidden_blocks:
                    if t in self.farm_tiles:
                        self.hidden_blocked.add(t)
            elif len(self.farm_tiles) >= 3:
                # Fallback: chá»n ngáº«u nhiĂªn náº¿u khĂ´ng truyá»n vĂ o
                candidates = self.farm_tiles[1:]
                random.shuffle(candidates)
                chosen = []
                for t in candidates:
                    # Äáº£m báº£o 2 Ă´ khĂ´ng cĂ¹ng hĂ ng vĂ  khĂ´ng cĂ¹ng cá»™t
                    if not chosen or (t[1] != chosen[0][1] and t[0] != chosen[0][0]):
                        chosen.append(t)
                    if len(chosen) == 2:
                        break
                for t in chosen:
                    self.hidden_blocked.add(t)
            # Explore quanh vá»‹ trĂ­ spawn ban Ä‘áº§u
            start_tile = self._world_to_tile(self.player.rect.center)
            self._expand_vision(start_tile)

        # --- Mode 5 (CSP): backtracking gĂ¡n crop ---
        self.seed_plan = {}
        self.csp_steps = []  # lÆ°u cĂ¡c bÆ°á»›c backtracking
        if self.mode == 5:
            self.seed_plan = self._solve_csp_crop_plan()

        # --- Mode 6 (Minimax): robot Ä‘á»‘i thá»§ ---
        self.enemy_tile = None
        self.enemy_target = None
        self.enemy_retreat_target = None
        self.enemy_path = []
        self.enemy_done_tiles = set()
        self.enemy_destroy_until = 0
        self.enemy_retarget_timer = 0.0
        self.minimax_value = 0
        self.alpha_beta_info = {"alpha": "-âˆ", "beta": "+âˆ", "pruned": 0}
        if self.mode == 6:
            if enemy_spawn is not None:
                # DĂ¹ng spawn cá»‘ Ä‘á»‹nh tá»« Level (gĂ³c pháº£i hĂ ng dÆ°á»›i)
                self.enemy_tile = tuple(enemy_spawn)
            elif self.farm_tiles:
                # Fallback: tĂ­nh tá»« farm tiles
                xs = [t[0] for t in self.farm_tiles]
                ys = [t[1] for t in self.farm_tiles]
                self.enemy_tile = (max(xs) + 3, max(ys) + 2)

    def _default_algorithm(self, mode):
        options = algorithms.ALGORITHM_GROUPS.get(mode)
        if options:
            return options[0]
        return self.MODE_INFO.get(mode, self.MODE_INFO[2])[1]

    def selectable_algorithms(self):
        return algorithms.ALGORITHM_GROUPS.get(self.mode, ())

    def set_algorithm(self, algorithm_name):
        options = self.selectable_algorithms()
        if not options or algorithm_name not in options:
            return False
        self.algorithm_name = algorithm_name
        self.message = f"Khu {self.mode}: {self.algorithm_name}"
        self.stats = {}
        self.path = []
        self.chosen_target_path = None
        self.dfs_walk = None
        self.dfs_walk_index = 0
        self.dfs_explored_order = []
        self.current_target = None
        self.state = "CHOOSE_TARGET"
        self.bfs_explored.clear()
        self.astar_explored.clear()
        self.hc_scores.clear()
        self.is_running = False
        self.message = f"Da chon {algorithm_name}. Nhan START"
        return True

    def cycle_algorithm(self, step=1):
        options = self.selectable_algorithms()
        if not options:
            return self.algorithm_name
        index = (
            options.index(self.algorithm_name)
            if self.algorithm_name in options else 0
        )
        self.set_algorithm(options[(index + step) % len(options)])
        return self.algorithm_name

    def start(self):
        if self.state == "DONE":
            return
        self.is_running = True
        self.message = f"Dang chay {self.algorithm_name}"

    def pause(self):
        self.is_running = False
        self.player.direction.update(0, 0)
        self.message = "Tam dung"

    def handle_panel_click(self, pos):
        for algorithm_name, rect in self.algorithm_buttons.items():
            if rect.collidepoint(pos):
                self.set_algorithm(algorithm_name)
                return "algorithm"

        for action, rect in self.control_buttons.items():
            if not rect.collidepoint(pos):
                continue
            if action == "start":
                self.start()
                return "start"
            if action == "pause":
                self.pause()
                return "pause"
            if action == "reset":
                return "reset"
        return None
    # -------------------------------------------------------- helpers chung
    def _tile_center(self, tile):
        x, y = tile
        return pygame.math.Vector2(
            x * TILE_SIZE + TILE_SIZE // 2,
            y * TILE_SIZE + TILE_SIZE // 2)

    def _world_to_tile(self, pos):
        return int(pos[0] // TILE_SIZE), int(pos[1] // TILE_SIZE)

    def _blocked_tiles(self, include_hidden=False):
        blocked = set()
        farm_set = set(self.farm_tiles)
        for sprite in self.collision_sprites.sprites():
            rect = getattr(sprite, "hitbox", getattr(sprite, "rect", None))
            if not rect:
                continue
            left = max(0, rect.left // TILE_SIZE)
            right = min(self.cols - 1, (rect.right - 1) // TILE_SIZE)
            top = max(0, rect.top // TILE_SIZE)
            bottom = min(self.rows - 1, (rect.bottom - 1) // TILE_SIZE)
            for ty in range(top, bottom + 1):
                for tx in range(left, right + 1):
                    if (tx, ty) not in farm_set:
                        blocked.add((tx, ty))
        blocked |= self.discovered_blocked
        if include_hidden:
            blocked |= self.hidden_blocked
        return blocked

    def _build_walkable_tiles(self):
        if self.mode != 1 or not self.farm_tiles:
            return None

        farm_set = set(self.farm_tiles)
        nearest = min(
            farm_set,
            key=lambda tile: self._heuristic(self.spawn_tile, tile))
        walkable = set(farm_set)
        walkable.add(self.spawn_tile)

        x, y = self.spawn_tile
        target_x, target_y = nearest
        step_x = 1 if target_x >= x else -1
        while x != target_x:
            walkable.add((x, y))
            x += step_x
        step_y = 1 if target_y >= y else -1
        while y != target_y:
            walkable.add((x, y))
            y += step_y
        walkable.add(nearest)
        return walkable

    def _neighbors(self, tile, blocked):
        x, y = tile
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < self.cols and 0 <= ny < self.rows \
                    and (nx, ny) not in blocked \
                    and (self.walkable_tiles is None
                         or (nx, ny) in self.walkable_tiles):
                yield (nx, ny)

    def _dfs_traversal_neighbors(self, tile, blocked):
        x, y = tile
        for nx, ny in ((x, y + 1), (x - 1, y), (x + 1, y), (x, y - 1)):
            if 0 <= nx < self.cols and 0 <= ny < self.rows \
                    and (nx, ny) not in blocked \
                    and (self.walkable_tiles is None
                         or (nx, ny) in self.walkable_tiles):
                yield (nx, ny)

    @staticmethod
    def _heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _search_heuristic(self, a, b):
        return self._heuristic(a, b)

    def _step_cost(self, current, next_tile):
        if self.mode != 2 or next_tile not in self.farm_tiles:
            return 1
        dryness = self.dryness.get(next_tile, 50)
        dryness_bonus = max(0, min(dryness, 100)) / 100.0
        return 1 + (1 - dryness_bonus)

    def _reconstruct(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path[1:]

    def _load_visual_assets(self):
        asset_dir = os.path.join("..", "graphics", "ai_visuals")
        asset_names = {
            "dry_plant": "dry_plant.png",
            "urgent_plant": "urgent_plant.png",
            "healthy_plant": "healthy_plant.png",
            "dead_plant": "dead_plant.png",
            "storm_debris": "storm_debris.png",
            "worm": "worm.png",
            "black_worm": "black_worm.png",
            "crow": "crow.png",
            "crow_damage": "crow_damage.png",
        }
        assets = {}
        for key, filename in asset_names.items():
            path = os.path.join(asset_dir, filename)
            if os.path.exists(path):
                image = pygame.image.load(path).convert_alpha()
                assets[key] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
        return assets

    def _blit_tile_asset(self, surface, offset, tile, asset_key, y_offset=0):
        image = self.visual_assets.get(asset_key)
        if image is None:
            return
        world = self._tile_center(tile)
        rect = image.get_rect(center=(int(world.x - offset.x),
                                      int(world.y - offset.y + y_offset)))
        surface.blit(image, rect)

    def _build_tile_conditions(self):
        conditions = {}
        if self.mode == 4:
            return conditions
        if self.mode == 5:
            return {tile: "replant" for tile in self.farm_tiles}

        for index, tile in enumerate(self.farm_tiles):
            dryness = self.dryness.get(tile, 50)
            if dryness >= 88:
                conditions[tile] = "dead"
            elif self.mode == 2 or dryness >= 65:
                conditions[tile] = "critical"
            elif index % 5 == 0 and self.mode in (1, 3):
                conditions[tile] = "pest"
            elif index % 7 == 0 and self.mode in (3, 6):
                conditions[tile] = "pest"
            else:
                conditions[tile] = "dry"
        return conditions

    def _condition_priority(self, tile):
        priorities = {
            "critical": 50,
            "crow": 45,
            "pest": 40,
            "dead": 20,
            "dry": 10,
            "replant": 5,
        }
        return priorities.get(self._condition_for_tile(tile), 0)

    def _care_value(self, tile):
        return self._condition_priority(tile) + self.dryness.get(tile, 50) / 10.0

    def _condition_asset(self, condition):
        if condition == "dead":
            return "dead_plant"
        if condition == "crow":
            return "crow_damage"
        if condition in ("critical", "pest", "replant"):
            return "urgent_plant"
        return "dry_plant"

    def _draw_resolved_asset(self, surface, offset, tile):
        condition = self._condition_for_tile(tile)
        if condition in ("dry", "critical", "pest", "crow"):
            self._blit_tile_asset(surface, offset, tile, "healthy_plant", y_offset=-6)

    def _draw_task_tile_marker(self, surface, offset, tile):
        world = self._tile_center(tile)
        sx = int(world.x - offset.x)
        sy = int(world.y - offset.y)
        marker = pygame.Surface((TILE_SIZE - 10, TILE_SIZE - 10),
                                pygame.SRCALPHA)
        pygame.draw.rect(marker, Colors.TASK_MARKER_FILL, marker.get_rect(),
                         border_radius=6)
        surface.blit(marker, (sx - TILE_SIZE // 2 + 5,
                              sy - TILE_SIZE // 2 + 5))
        rect = pygame.Rect(0, 0, TILE_SIZE - 10, TILE_SIZE - 10)
        rect.center = (sx, sy)
        pygame.draw.rect(surface, Colors.TASK_MARKER_BORDER, rect, 1,
                         border_radius=5)
        corner = 10
        color = Colors.TASK_MARKER_CORNER
        pygame.draw.line(surface, color, rect.topleft, (rect.left + corner, rect.top), 2)
        pygame.draw.line(surface, color, rect.topleft, (rect.left, rect.top + corner), 2)
        pygame.draw.line(surface, color, rect.topright, (rect.right - corner, rect.top), 2)
        pygame.draw.line(surface, color, rect.topright, (rect.right, rect.top + corner), 2)
        pygame.draw.line(surface, color, rect.bottomleft, (rect.left + corner, rect.bottom), 2)
        pygame.draw.line(surface, color, rect.bottomleft, (rect.left, rect.bottom - corner), 2)
        pygame.draw.line(surface, color, rect.bottomright, (rect.right - corner, rect.bottom), 2)
        pygame.draw.line(surface, color, rect.bottomright, (rect.right, rect.bottom - corner), 2)

    def _draw_mist_effect(self, surface):
        mist = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        width, height = surface.get_size()
        mist.fill(Colors.MIST_BASE)
        bands = [
            (0.08, 0.20, 1.20, 68, 46),
            (0.16, 0.36, 0.95, 62, 42),
            (0.24, 0.50, 1.10, 66, 44),
            (0.34, 0.24, 0.95, 60, 40),
            (0.46, 0.44, 1.05, 48, 34),
            (0.60, 0.30, 0.82, 34, 26),
            (0.74, 0.52, 0.90, 28, 22),
            (0.88, 0.26, 0.75, 22, 18),
        ]
        for index, (y_ratio, speed, scale, alpha, thickness) in enumerate(bands):
            y = int(height * y_ratio + math.sin(self.fog_time * 0.8 + index) * 14)
            drift = int((self.fog_time * 44 * speed + index * 155) % (width + 360)) - 360
            color = (*Colors.MIST_BAND, alpha)
            pygame.draw.ellipse(mist, color, (drift, y, int(420 * scale), thickness))
            pygame.draw.ellipse(mist, color, (drift + int(210 * scale), y + 12, int(360 * scale), thickness))
            pygame.draw.ellipse(
                mist, (*Colors.MIST_BAND, max(18, alpha - 14)),
                (drift - int(260 * scale), y + 26, int(340 * scale), max(18, thickness - 10)))
            pygame.draw.ellipse(
                mist, (*Colors.MIST_HIGHLIGHT, max(14, alpha - 20)),
                (drift + int(430 * scale), y - 10, int(280 * scale), max(16, thickness - 12)))
        surface.blit(mist, (0, 0))

    def _seed_for_tile(self, tile):
        if self.mode == 5:
            return self.seed_plan.get(tile, "corn")
        if self.dryness.get(tile, 0) > 60:
            return "tomato"
        return "corn"

    def _condition_for_tile(self, tile):
        return self.tile_conditions.get(tile, "replant")

    def _has_condition(self, tile):
        return tile in self.tile_conditions

    def _soil_cell(self, tile):
        x, y = tile
        if 0 <= y < self.rows and 0 <= x < self.cols:
            return self.soil_layer.grid[y][x]
        return []

    def _next_cultivation_state(self, tile):
        cell = self._soil_cell(tile)
        if "X" not in cell:
            return "HOE"
        if "P" not in cell:
            return "PLANT"
        if "W" not in cell:
            return "WATER"
        return "DONE_TILE"

    def _is_living_crop(self, tile):
        return self._condition_for_tile(tile) in ("dry", "critical", "pest", "crow")

    def _enemy_retreat_tile(self, damaged_tile):
        if not self.farm_tiles:
            return (damaged_tile[0], damaged_tile[1] + 1)
        min_y = min(t[1] for t in self.farm_tiles)
        max_y = max(t[1] for t in self.farm_tiles)
        mid_y = (min_y + max_y) / 2
        retreat_x = max(1, min(self.cols - 2, damaged_tile[0]))
        if damaged_tile[1] <= mid_y:
            retreat_y = max(1, min_y - 2)
        else:
            retreat_y = min(self.rows - 2, max_y + 2)
        return (retreat_x, retreat_y)

    def _state_after_arrival(self, tile):
        condition = self._condition_for_tile(tile)
        if condition in ("dead", "replant"):
            return self._next_cultivation_state(tile)
        if condition in ("pest", "crow"):
            return "TREAT_PEST"
        return "WATER_RESCUE"

    def _finish_rescue_target(self, tile, condition):
        self.player.selected_tool = "water"
        self.player.status = "down_water"
        if hasattr(self.player, "watering"):
            self.player.watering.play()
        if condition == "critical":
            self.message = f"Cap cuu cay thieu nuoc tai {tile}"
        else:
            self.message = f"Tuoi phuc hoi cay kho tai {tile}"
        self.done_tiles.add(tile)
        self.current_target = None
        self.path = []
        self.state = "CHOOSE_TARGET"
        self.wait_time = 0.45

    def _finish_pest_target(self, tile, condition):
        self.player.status = "down_idle"
        if condition == "crow":
            self.message = f"Duoi qua va bao ve cay tai {tile}"
        else:
            self.message = f"Phun sinh hoc xu ly sau benh tai {tile}"
        self.done_tiles.add(tile)
        self.current_target = None
        self.path = []
        self.state = "CHOOSE_TARGET"
        self.wait_time = 0.45

    def _finish_cultivation_target(self, tile, message):
        self.message = message
        self.done_tiles.add(tile)
        self.current_target = None
        self.path = []
        self.state = "CHOOSE_TARGET"
        self.wait_time = 0.35

    def _condition_label(self, condition):
        labels = {
            "dry": "Cay kho -> tuoi phuc hoi",
            "critical": "Cay nguy cap -> cap cuu bang nuoc",
            "pest": "Sau benh sau bao -> phun sinh hoc",
            "crow": "Vet qua pha -> duoi qua va bao ve cay",
            "dead": "Cay chet -> don va trong lai",
            "replant": "Dat trong -> lap ke hoach gieo lai",
        }
        return labels.get(condition, "Dat trong -> gieo lai")

    def _target_label(self, tile):
        return self._condition_label(self._condition_for_tile(tile))

    # -------------------------------------------------------------- MODE 1
    def _bfs(self, start, goal):
        """Breadth-First Search â€” duyá»‡t theo lá»›p, khĂ´ng heuristic."""
        return self._search_path(start, goal, "BFS")

    # -------------------------------------------------------------- MODE 2
    def _astar(self, start, goal):
        """A* Search â€” f(n) = g(n) + h(n), heuristic Manhattan."""
        return self._search_path(start, goal, "A*")

    def _search_path(self, start, goal, algorithm_name=None):
        blocked = self._blocked_tiles()
        blocked.discard(start)
        blocked.discard(goal)
        name = algorithm_name or self.algorithm_name
        path, explored, fgh, stats = algorithms.find_path_by_algorithm(
            name, start, goal, blocked, self._neighbors, self._search_heuristic,
            self.counter, self._step_cost)
        self.astar_current_fgh = fgh
        stats["Algorithm"] = name
        if self.mode == 2 and goal in self.farm_tiles:
            g = self.dryness.get(goal, 50)
            h = self._heuristic(start, goal)
            self.astar_current_fgh = (g + h, g, h)
            stats["f(n)"] = f"{g + h}"
            stats["g(n)"] = f"{g}"
            stats["h(n)"] = f"{h}"
            stats["Dryness goal"] = f"{g}%"
        if self.mode == 1:
            self.bfs_explored |= explored
            stats["Nodes explored"] = len(self.bfs_explored)
        elif self.mode == 2:
            self.astar_explored = explored
        elif name == "A*":
            self.astar_explored = explored
        self.stats = stats
        return path

    def _search_first_target(self, start, goals):
        blocked = self._blocked_tiles()
        blocked.discard(start)
        for goal in goals:
            blocked.discard(goal)
        path, target, explored, fgh, stats = (
            algorithms.find_path_to_any_goal_by_algorithm(
                self.algorithm_name, start, goals, blocked,
                self._neighbors, self._search_heuristic, self.counter,
                self._step_cost))
        self.astar_current_fgh = fgh
        stats["Algorithm"] = self.algorithm_name
        stats["Target"] = f"{target}"
        if self.mode == 1:
            self.bfs_explored |= explored
            stats["Nodes explored"] = len(self.bfs_explored)
        elif self.mode == 2:
            self.astar_explored = explored
        self.stats = stats
        self.chosen_target_path = path if target is not None else None
        return target

    def _build_dfs_walk(self, start):
        blocked = self._blocked_tiles()
        blocked.discard(start)
        for goal in self.farm_tiles:
            blocked.discard(goal)

        visited = set()
        walk = [start]
        explored_order = []

        def visit(tile):
            visited.add(tile)
            explored_order.append(tile)
            for next_tile in self._dfs_traversal_neighbors(tile, blocked):
                if next_tile in visited:
                    continue
                walk.append(next_tile)
                visit(next_tile)
                walk.append(tile)

        visit(start)
        self.dfs_walk = walk
        self.dfs_walk_index = 0
        self.dfs_explored_order = explored_order

    def _dfs_traversal_choose(self, remaining, start):
        if self.dfs_walk is None:
            self._build_dfs_walk(start)

        if self.dfs_walk[self.dfs_walk_index] != start:
            try:
                self.dfs_walk_index = self.dfs_walk.index(
                    start, self.dfs_walk_index)
            except ValueError:
                self._build_dfs_walk(start)

        path = []
        remaining_set = set(remaining)
        for index in range(self.dfs_walk_index + 1, len(self.dfs_walk)):
            tile = self.dfs_walk[index]
            path.append(tile)
            if tile in remaining_set:
                self.dfs_walk_index = index
                self.chosen_target_path = path
                explored = set(self.dfs_walk[:index + 1])
                self.bfs_explored |= explored
                self.stats = {
                    "Nodes explored": len(self.bfs_explored),
                    "Path length": len(path),
                    "Stack max": len(self.dfs_explored_order),
                    "Algorithm": "DFS",
                    "Target": f"{tile}",
                }
                return tile

        self.chosen_target_path = None
        return None

    # -------------------------------------------------------------- MODE 3
    def _hill_score(self, tile, start):
        """HĂ m Ä‘Ă¡nh giĂ¡ cá»¥c bá»™: dryness cao = Æ°u tiĂªn, xa = trá»« Ä‘iá»ƒm."""
        return algorithms.hill_score(
            tile, start, self.dryness, self._heuristic)

    def _hill_climbing_choose(self, remaining, start):
        """Steepest-ascent hill climbing: chá»n neighbor tá»‘t nháº¥t liĂªn tá»¥c.

        1. Báº¯t Ä‘áº§u tá»« Ă´ gáº§n nháº¥t (initial state).
        2. XĂ©t cĂ¡c Ă´ ká» (neighbors) trong remaining.
        3. Chá»n Ă´ cĂ³ score cao nháº¥t. Náº¿u khĂ´ng tá»‘t hÆ¡n hiá»‡n táº¡i â†’ dá»«ng.
        """
        current_best, current_score, self.hc_scores, self.stats = (
            algorithms.local_search_choose(
                self.algorithm_name, remaining, start, self.dryness,
                self._heuristic))
        self.hc_current_score = current_score
        self.hc_best_neighbor_score = current_score
        return current_best

    # -------------------------------------------------------------- MODE 4
    def _expand_vision(self, center):
        """Má»Ÿ rá»™ng vĂ¹ng Ä‘Ă£ khĂ¡m phĂ¡ quanh center."""
        algorithms.expand_vision(
            center, self.vision_radius, self.rows, self.cols,
            self.hidden_blocked, self.explored_tiles,
            self.discovered_blocked)

    def _online_plan(self, start, goal):
        """Online Search: chá»‰ biáº¿t váº­t cáº£n Ä‘Ă£ khĂ¡m phĂ¡, phĂ¡t hiá»‡n thĂªm khi Ä‘i."""
        if goal in self.discovered_blocked:
            return []
        path = self._astar(start, goal)
        total_explored = len(self.explored_tiles)
        total_map = self.rows * self.cols
        self.stats = {
            "Explored tiles": f"{total_explored}",
            "Map coverage": f"{total_explored * 100 // max(total_map, 1)}%",
            "Replanned": f"{self.replan_count} lan",
            "Hidden blocks": f"{len(self.hidden_blocked - self.discovered_blocked)} chua biet",
        }
        return path

    # -------------------------------------------------------------- MODE 5
    def _solve_csp_crop_plan(self):
        """CSP Backtracking: gĂ¡n corn/tomato sao cho 2 Ă´ ká» khĂ¡c loáº¡i."""
        assignment, self.csp_steps, self.stats = (
            algorithms.solve_csp_crop_plan(self.farm_tiles))
        return assignment

    # -------------------------------------------------------------- MODE 6
    def _minimax_choose(self, remaining, player_start):
        """Minimax vá»›i Alpha-Beta pruning Ä‘Æ¡n giáº£n.

        Player (MAX) chá»n Ă´ cĂ³ giĂ¡ trá»‹ cao nháº¥t.
        Enemy (MIN) giáº£ láº­p chá»n Ă´ gáº§n mĂ¬nh nháº¥t Ä‘á»ƒ tranh.
        """
        best_tile, predicted_enemy_target, self.minimax_value, \
            self.alpha_beta_info, self.stats = algorithms.minimax_choose(
                remaining, player_start, self.enemy_tile,
                self.enemy_done_tiles, self._care_value, self._heuristic,
                self._is_living_crop)
        if (not self.enemy_target and not self.enemy_retreat_target
                and not self._enemy_is_destroying()):
            self.enemy_target = predicted_enemy_target
        return best_tile

    def _enemy_destroy_remaining(self):
        if not self.enemy_destroy_until:
            return 0.0
        remaining_ms = self.enemy_destroy_until - pygame.time.get_ticks()
        return max(0.0, remaining_ms / 1000)

    def _enemy_is_destroying(self):
        return self._enemy_destroy_remaining() > 0

    def _enemy_remaining_targets(self):
        return [t for t in self.farm_tiles
                if t not in self.done_tiles
                and t not in self.discovered_blocked
                and t not in self.enemy_done_tiles
                and t != self.current_target]

    def _choose_enemy_target(self):
        if not self.enemy_tile:
            return
        remaining = self._enemy_remaining_targets()
        if not remaining:
            self.enemy_target = None
            return
        self.enemy_target = min(
            remaining, key=lambda tile: self._heuristic(self.enemy_tile, tile))

    def _release_conflicting_locks(self):
        if self.mode != 6:
            return

        if (self.current_target is not None
                and self.enemy_target == self.current_target
                and not self._enemy_is_destroying()):
            self.enemy_target = None
            self.enemy_retarget_timer = 0.0

        if (self.enemy_target is not None
                and self.current_target == self.enemy_target
                and self.state in ("CHOOSE_TARGET", "MOVE")):
            self.current_target = None
            self.path = []
            self.state = "CHOOSE_TARGET"
            self.message = "Player re-plan vi qua da lock muc tieu"

    def _update_enemy_strategy(self, dt):
        if self.mode != 6:
            return
        if (self.enemy_target or self.enemy_retreat_target
                or self._enemy_is_destroying()):
            return
        if self.enemy_retarget_timer > 0:
            self.enemy_retarget_timer = max(0, self.enemy_retarget_timer - dt)
            if self.enemy_retarget_timer > 0:
                return
        self._choose_enemy_target()

    def _move_enemy_towards(self, dt):
        """Di chuyá»ƒn enemy dáº§n dáº§n tá»›i target."""
        if not self.enemy_tile:
            return
        if (self.enemy_target is not None
                and (self.enemy_target == self.current_target
                     or self.enemy_target in self.done_tiles
                     or self.enemy_target in self.enemy_done_tiles)
                and not self._enemy_is_destroying()):
            self.enemy_target = None
            self.enemy_retarget_timer = 0.0
            return
        target = self.enemy_target or self.enemy_retreat_target
        if not target:
            return

        if self.enemy_destroy_until:
            if not self._enemy_is_destroying() and self.enemy_target:
                damaged_tile = self.enemy_target
                self.enemy_done_tiles.add(damaged_tile)
                self.enemy_tile = damaged_tile
                self.enemy_target = None
                self.enemy_destroy_until = 0
                self.enemy_retarget_timer = self.ENEMY_RETARGET_DELAY
            return

        ex, ey = self.enemy_tile
        tx, ty = target
        step = self.ENEMY_MOVE_SPEED * dt

        dx = tx - ex
        dy = ty - ey
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.3:
            # Enemy "Ä‘áº¿n" Ă´ target
            if self.enemy_retreat_target and not self.enemy_target:
                self.enemy_tile = self.enemy_retreat_target
                self.enemy_retreat_target = None
                return
            self.enemy_destroy_until = (
                pygame.time.get_ticks() + int(self.ENEMY_DESTROY_TIME * 1000))
            return

        # Di chuyá»ƒn tá»«ng bÆ°á»›c nhá»
        if dist > 0:
            move_x = (dx / dist) * step
            move_y = (dy / dist) * step
            self.enemy_tile = (ex + move_x, ey + move_y)

    # -------------------------------------------------------- path dispatch
    def _path_for_mode(self, start, goal):
        if self.mode in (1, 2):
            return self._search_path(start, goal)
        if self.mode == 4:
            return self._online_plan(start, goal)
        return self._astar(start, goal)

    def _choose_target(self, remaining, start):
        if self.mode == 3:
            return self._hill_climbing_choose(remaining, start)
        if self.mode == 6:
            return self._minimax_choose(remaining, start)
        if self.mode == 1 and self.algorithm_name == "DFS":
            return self._dfs_traversal_choose(remaining, start)
        if self.mode == 2:
            return max(
                remaining,
                key=lambda tile: (
                    self.dryness.get(tile, 0),
                    -self._heuristic(start, tile),
                    tile))
        if self.mode in (1, 2):
            return self._search_first_target(start, remaining)

        # Mode 4,5: giu hanh vi demo rieng, chon muc tieu gan truoc.
        best = None
        for tile in remaining:
            path = self._path_for_mode(start, tile)
            if path or start == tile:
                distance = len(path)
                sort_key = (distance, tile)
                if best is None or sort_key < best[0]:
                    best = (sort_key, tile)
        return best[1] if best else None

    # ---------------------------------------------------- player direction
    def _set_player_direction_to(self, world_pos):
        """Di chuyá»ƒn player vá» world_pos. Tráº£ vá» True khi Ä‘Ă£ Ä‘áº¿n nÆ¡i."""
        delta = pygame.math.Vector2(world_pos) - self.player.pos
        # NgÆ°á»¡ng lá»›n hÆ¡n: 1 tile = 64px, player speed=200px/s
        # Chá»‰ cáº§n náº±m trong 14px (< 1 bÆ°á»›c dt=0.07s) lĂ  coi nhÆ° Ä‘áº¿n
        if delta.length() < 14:
            # Snap chĂ­nh xĂ¡c vĂ o center tile trĂ¡nh drift tĂ­ch lÅ©y
            self.player.pos.update(world_pos)
            self.player.rect.center = (round(world_pos.x), round(world_pos.y))
            self.player.hitbox.center = self.player.rect.center
            self.player.direction.update(0, 0)
            return True
        if abs(delta.x) > abs(delta.y):
            self.player.direction.update(1 if delta.x > 0 else -1, 0)
            self.player.status = "right" if delta.x > 0 else "left"
        else:
            self.player.direction.update(0, 1 if delta.y > 0 else -1)
            self.player.status = "down" if delta.y > 0 else "up"
        return False

    # ============================================================= UPDATE
    def update(self, dt):
        if self.mode == 4:
            self.fog_time += dt

        if not self.is_running:
            self.player.direction.update(0, 0)
            return

        if self.wait_time > 0:
            self.wait_time -= dt
            self.player.direction.update(0, 0)
            # Mode 6: enemy váº«n di chuyá»ƒn khi player chá»
            if self.mode == 6:
                self._update_enemy_strategy(dt)
                self._move_enemy_towards(dt)
            return

        if self.state == "DONE":
            self.player.direction.update(0, 0)
            if self.mode == 6:
                self._update_enemy_strategy(dt)
                self._move_enemy_towards(dt)
            return

        # --- Stuck detection: náº¿u Ä‘ang MOVE mĂ  position khĂ´ng Ä‘á»•i > 1.5s ---
        if self.state == "MOVE":
            cur_pos = (round(self.player.pos.x), round(self.player.pos.y))
            if not hasattr(self, '_stuck_pos'):
                self._stuck_pos = cur_pos
                self._stuck_timer = 0.0
            elif cur_pos == self._stuck_pos:
                self._stuck_timer += dt
                if self._stuck_timer > 1.5:
                    # Teleport player lĂªn center cá»§a next_tile, bá» qua tile nĂ y
                    if self.path:
                        snap_tile = self.path[0]
                        snap_world = self._tile_center(snap_tile)
                        self.player.pos.update(snap_world)
                        self.player.rect.center = (round(snap_world.x), round(snap_world.y))
                        self.player.hitbox.center = self.player.rect.center
                        self.path.pop(0)
                    self._stuck_pos = None
                    self._stuck_timer = 0.0
                    self.wait_time = 0.1
                    self.message = "Recover: skip tile bi ket"
                    return
            else:
                self._stuck_pos = cur_pos
                self._stuck_timer = 0.0

        # Mode 6: enemy di chuyá»ƒn liĂªn tá»¥c
        if self.mode == 6:
            self._release_conflicting_locks()
            self._update_enemy_strategy(dt)
            self._move_enemy_towards(dt)
            self._release_conflicting_locks()

        # --- CHOOSE TARGET ---
        if self.state == "CHOOSE_TARGET":
            remaining = [t for t in self.farm_tiles
                         if t not in self.done_tiles
                         and t not in self.discovered_blocked
                         and t not in self.enemy_done_tiles
                         and (self.mode != 6 or t != self.enemy_target)]
            if not remaining:
                self.state = "DONE"
                done_count = len(self.done_tiles)
                enemy_count = len(self.enemy_done_tiles)
                if self.mode == 6:
                    if self.enemy_tile:
                        enemy_tile = (
                            int(round(self.enemy_tile[0])),
                            int(round(self.enemy_tile[1])))
                        self.enemy_retreat_target = self._enemy_retreat_tile(enemy_tile)
                    self.message = (
                        f"Ket thuc! Player: {done_count}, "
                        f"Enemy: {enemy_count} o")
                else:
                    self.message = (
                        f"Hoan thanh khu {self.mode}: "
                        f"{self.algorithm_name}.")
                return

            start = self._world_to_tile(self.player.rect.center)

            # Mode 4: má»Ÿ rá»™ng táº§m nhĂ¬n
            if self.mode == 4:
                self._expand_vision(start)

            target = self._choose_target(remaining, start)
            if target is None:
                self.state = "DONE"
                self.message = "Khong tim duoc muc tieu hop le."
                return

            self.current_target = target
            if self.mode == 6 and self.enemy_target == self.current_target:
                self.current_target = None
                self.path = []
                self.state = "CHOOSE_TARGET"
                self.message = "Player re-plan vi qua da lock muc tieu"
                return
            if self.chosen_target_path is not None:
                self.path = self.chosen_target_path
                self.chosen_target_path = None
            else:
                self.path = self._path_for_mode(start, target)

            if not self.path and start != target:
                if self.mode == 4:
                    self.replan_count += 1
                    self.message = (
                        f"Online: o {target} bi chan, "
                        f"re-plan (lan {self.replan_count})")
                    self.wait_time = 0.4
                    return
                self.done_tiles.add(target)
                self.message = f"Bo qua {target}: khong co duong."
                return

            self.state = "MOVE"
            self.message = f"{self.algorithm_name}: di toi {target}"
            return

        # --- MOVE ---
        if self.state == "MOVE":
            if (self.mode == 6 and self.enemy_target is not None
                    and self.current_target == self.enemy_target):
                self.current_target = None
                self.path = []
                self.state = "CHOOSE_TARGET"
                self.message = "Player re-plan vi qua da lock muc tieu"
                return
            if not self.path:
                self.state = self._state_after_arrival(self.current_target)
                return

            next_tile = self.path[0]

            # Mode 4: phĂ¡t hiá»‡n váº­t cáº£n áº©n khi Ä‘i gáº§n
            if self.mode == 4:
                self._expand_vision(
                    self._world_to_tile(self.player.rect.center))
                if next_tile in self.discovered_blocked:
                    self.path = []
                    self.state = "CHOOSE_TARGET"
                    self.replan_count += 1
                    self.message = (
                        f"Online: phat hien vat can tai {next_tile}, "
                        f"re-plan (lan {self.replan_count})")
                    self.wait_time = 0.4
                    return

            reached = self._set_player_direction_to(
                self._tile_center(next_tile))
            if reached:
                self.path.pop(0)
                # Mode 4: tiáº¿p tá»¥c má»Ÿ rá»™ng táº§m nhĂ¬n khi di chuyá»ƒn
                if self.mode == 4:
                    self._expand_vision(next_tile)
            return

        # --- WATER_RESCUE / TREAT_PEST / HOE / PLANT / WATER ---
        current_tile = self._world_to_tile(self.player.rect.center)
        action_tile = self.current_target if self.current_target is not None else current_tile
        target_pos = self._tile_center(action_tile)
        self.player.status = "down_idle"
        self.player.direction.update(0, 0)

        if self.state == "WATER_RESCUE":
            self._finish_rescue_target(
                action_tile, self._condition_for_tile(action_tile))
            return

        if self.state == "TREAT_PEST":
            self._finish_pest_target(
                action_tile, self._condition_for_tile(action_tile))
            return

        if self.state == "DONE_TILE":
            self._finish_cultivation_target(
                action_tile, f"O {action_tile} da san sang, bo qua buoc thua")
            return

        if self.state == "HOE":
            condition = self._condition_for_tile(action_tile)
            if "X" in self._soil_cell(action_tile):
                self.message = f"Dat tai {action_tile} da duoc cuoc, bo qua cuoc"
            else:
                self.player.selected_tool = "hoe"
                self.soil_layer.get_hit(target_pos)
                if condition == "dead":
                    self.message = f"Don cay chet tai {action_tile}"
                else:
                    self.message = f"Cuoc dat tai {action_tile}"
            self.state = self._next_cultivation_state(action_tile)
            self.wait_time = 0.18 if self.state != "PLANT" else 0.30
            return

        if self.state == "PLANT":
            condition = self._condition_for_tile(action_tile)
            if "P" in self._soil_cell(action_tile):
                self.message = f"O {action_tile} da co cay, bo qua gieo hat"
            else:
                seed = self._seed_for_tile(action_tile)
                self.player.selected_seed = seed
                self.soil_layer.plant_seed(target_pos, seed)
                if condition == "dead":
                    self.message = f"Trong lai {seed} tai {action_tile}"
                else:
                    self.message = f"Gieo {seed} tai {action_tile}"
            self.state = self._next_cultivation_state(action_tile)
            self.wait_time = 0.18 if self.state != "WATER" else 0.30
            return

        if self.state == "WATER":
            if "W" not in self._soil_cell(action_tile):
                self.player.selected_tool = "water"
                self.soil_layer.water(target_pos)
                message = f"Tuoi nuoc tai {action_tile}"
            else:
                message = f"O {action_tile} da co nuoc, bo qua tuoi"
            self._finish_cultivation_target(action_tile, message)
            return

    # ============================================================= DRAW
    def draw(self, surface, offset):
        self._draw_map_overlays(surface, offset)
        self._draw_panel(surface)

    # ------------------------------------------------------ map overlays
    def _draw_map_overlays(self, surface, offset):
        """Váº½ thĂ´ng tin thuáº­t toĂ¡n lĂªn map (explored nodes, scores, fog, enemy...)."""

        for tile in self.farm_tiles:
            self._draw_task_tile_marker(surface, offset, tile)

        if self.mode in (1, 2, 3, 6):
            for tile in self.farm_tiles:
                if tile in self.enemy_done_tiles:
                    continue
                if tile in self.done_tiles:
                    self._draw_resolved_asset(surface, offset, tile)
                    continue
                condition = self._condition_for_tile(tile)
                asset_key = self._condition_asset(condition)
                self._blit_tile_asset(surface, offset, tile, asset_key, y_offset=-6)
                if condition == "pest":
                    self._blit_tile_asset(surface, offset, tile, "worm", y_offset=2)
                elif condition == "crow":
                    self._blit_tile_asset(surface, offset, tile, "black_worm", y_offset=2)

        # --- Mode 1: BFS explored nodes (xanh dÆ°Æ¡ng nháº¡t) ---
        if self.mode == 1 and self.bfs_explored:
            for tile in self.bfs_explored:
                world = self._tile_center(tile)
                sx = int(world.x - offset.x)
                sy = int(world.y - offset.y)
                s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4),
                                   pygame.SRCALPHA)
                s.fill(MODE_EXPLORE_FILLS[1])
                surface.blit(s, (sx - TILE_SIZE // 2 + 2,
                                 sy - TILE_SIZE // 2 + 2))

        # --- Mode 2: A* explored nodes (xanh lĂ¡ nháº¡t) ---
        if self.mode == 2 and self.astar_explored:
            for tile in self.astar_explored:
                world = self._tile_center(tile)
                sx = int(world.x - offset.x)
                sy = int(world.y - offset.y)
                s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4),
                                   pygame.SRCALPHA)
                s.fill(MODE_EXPLORE_FILLS[2])
                surface.blit(s, (sx - TILE_SIZE // 2 + 2,
                                 sy - TILE_SIZE // 2 + 2))

        # --- Mode 3: Hill Climbing scores trĂªn má»—i Ă´ ---
        if self.mode == 3 and self.hc_scores:
            font_sm = pygame.font.Font(None, 20)
            for tile, score in self.hc_scores.items():
                if tile in self.done_tiles:
                    continue
                world = self._tile_center(tile)
                sx = int(world.x - offset.x)
                sy = int(world.y - offset.y)
                # TĂ´ mĂ u theo score
                ratio = max(0, min(1, (score + 50) / 100))
                r = int(255 * (1 - ratio))
                g = int(255 * ratio)
                s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4),
                                   pygame.SRCALPHA)
                s.fill((r, g, 80, 60))
                surface.blit(s, (sx - TILE_SIZE // 2 + 2,
                                 sy - TILE_SIZE // 2 + 2))
                txt = font_sm.render(f"{score:.0f}", True, Colors.SCORE_TEXT)
                surface.blit(txt, (sx - 10, sy - 8))

        # --- Mode 4: Fog-of-war ---
        if self.mode == 4:
            fog = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
            fog.fill(Colors.FOG_DARK)
            center = (
                int(self.player.rect.centerx - offset.x),
                int(self.player.rect.centery - offset.y),
            )
            vision_px = int(self.vision_radius * TILE_SIZE)
            pygame.draw.circle(fog, Colors.FOG_EDGE,
                               center, vision_px + 34)
            pygame.draw.circle(fog, Colors.FOG_MID,
                               center, vision_px + 18)
            pygame.draw.circle(fog, Colors.FOG_CLEAR, center, vision_px)
            surface.blit(fog, (0, 0))
            self._draw_mist_effect(surface)

            # Váº½ Ă´ block Ä‘Ă£ phĂ¡t hiá»‡n
            for tile in self.discovered_blocked:
                self._blit_tile_asset(surface, offset, tile, "storm_debris")
                world = self._tile_center(tile)
                rect = pygame.Rect(0, 0, TILE_SIZE - 8, TILE_SIZE - 8)
                rect.center = (int(world.x - offset.x),
                                int(world.y - offset.y))
                font_sm = pygame.font.Font(None, 18)
                txt = font_sm.render("BLOCK", True, Colors.BLOCK_LABEL)
                surface.blit(txt, (rect.x + 4, rect.y + 20))

            # Váº½ Ă´ block áº©n chÆ°a phĂ¡t hiá»‡n (nháº¹, debug)
        # --- Mode 5: CSP assignment labels (C/T) ---
        if self.mode == 5 and self.seed_plan:
            font_sm = pygame.font.Font(None, 22)
            for tile, crop in self.seed_plan.items():
                world = self._tile_center(tile)
                sx = int(world.x - offset.x)
                sy = int(world.y - offset.y)
                label = "C" if crop == "corn" else "T"
                color = Colors.CSP_CORN if crop == "corn" else Colors.CSP_TOMATO
                # Ná»n
                s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4),
                                   pygame.SRCALPHA)
                bg = Colors.CSP_CORN_BG if crop == "corn" else Colors.CSP_TOMATO_BG
                s.fill(bg)
                surface.blit(s, (sx - TILE_SIZE // 2 + 2,
                                 sy - TILE_SIZE // 2 + 2))
                txt = font_sm.render(label, True, color)
                surface.blit(txt, (sx - 5, sy - 8))

        # --- Mode 6: Enemy + enemy path ---
        if self.mode == 6 and self.enemy_tile:
            world = self._tile_center(
                (int(round(self.enemy_tile[0])),
                 int(round(self.enemy_tile[1]))))
            ex = int(world.x - offset.x)
            ey = int(world.y - offset.y)
            # ThĂ¢n enemy
            crow = self.visual_assets.get("crow")
            if crow:
                crow_rect = crow.get_rect(center=(ex, ey - 4))
                surface.blit(crow, crow_rect)
            else:
                pygame.draw.circle(surface, Colors.ENEMY_FALLBACK, (ex, ey), 20)
                pygame.draw.circle(surface, Colors.ENEMY_FALLBACK_OUTLINE, (ex, ey), 20, 3)
            # Label
            font_sm = pygame.font.Font(None, 20)
            txt = font_sm.render("CROW", True, Colors.ENEMY)
            surface.blit(txt, (ex - 18, ey - 42))

            # Enemy target line
            if self.enemy_target:
                tw = self._tile_center(self.enemy_target)
                tx = int(tw.x - offset.x)
                ty = int(tw.y - offset.y)
                pygame.draw.line(surface, Colors.ENEMY,
                                 (ex, ey), (tx, ty), 2)
                pygame.draw.circle(surface, Colors.ENEMY,
                                   (tx, ty), 8, 2)

            # Ă” enemy Ä‘Ă£ chiáº¿m
            for tile in self.enemy_done_tiles:
                w = self._tile_center(tile)
                rect = pygame.Rect(0, 0, TILE_SIZE - 6, TILE_SIZE - 6)
                rect.center = (int(w.x - offset.x),
                                int(w.y - offset.y))
                pygame.draw.rect(surface, Colors.ENEMY_FALLBACK, rect, 3,
                                 border_radius=4)
                original_condition = self._condition_for_tile(tile)
                asset_key = self._condition_asset(original_condition)
                self._blit_tile_asset(surface, offset, tile, asset_key, y_offset=-6)
                self._blit_tile_asset(surface, offset, tile, "black_worm", y_offset=4)

        # --- ÄÆ°á»ng Ä‘i chung (vĂ ng) ---
        if self.path:
            points = []
            for tile in self.path:
                world = self._tile_center(tile)
                points.append((int(world.x - offset.x),
                                int(world.y - offset.y)))
            if len(points) >= 2:
                pygame.draw.lines(surface, Colors.PATH, False, points, 4)
            for p in points:
                pygame.draw.circle(surface, Colors.PATH, p, 5)

    # ------------------------------------------------------------ panel
    def _draw_panel(self, surface):
        """Draw the left control panel with algorithm selection and run buttons."""
        font = pygame.font.Font(None, 22)
        font_title = pygame.font.Font(None, 26)
        font_small = pygame.font.Font(None, 20)
        border_color = MODE_COLORS.get(self.mode, (200, 200, 200))

        panel_width = 390
        panel_height = 600 if self.mode in (1, 2, 3) else 360
        panel = pygame.Rect(20, 20, panel_width, panel_height)
        panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surf.fill(Colors.PANEL_BG)
        surface.blit(panel_surf, panel.topleft)
        pygame.draw.rect(surface, border_color, panel, 2, border_radius=8)

        self.algorithm_buttons = {}
        self.control_buttons = {}

        def draw_text(text, x, y, color=Colors.TEXT_PRIMARY, used_font=None):
            txt = (used_font or font).render(text, True, color)
            surface.blit(txt, (x, y))
            return y + txt.get_height() + 6

        def draw_button(key, rect, text, selected=False, danger=False):
            if danger:
                fill = (110, 52, 52, 220)
            elif selected:
                fill = (*border_color, 210)
            else:
                fill = (38, 42, 54, 230)
            text_color = (15, 18, 22) if selected else Colors.TEXT_PRIMARY
            pygame.draw.rect(surface, fill, rect, border_radius=6)
            pygame.draw.rect(surface, border_color, rect, 1, border_radius=6)
            label = font_small.render(text, True, text_color)
            surface.blit(label, label.get_rect(center=rect.center))
            return rect

        x = panel.x + 15
        y = panel.y + 14
        y = draw_text(f"KHU {self.mode}", x, y, border_color, font_title)
        y = draw_text(self.mode_name, x, y, Colors.TEXT_SUBTITLE, font_small)
        y += 4

        if self.mode in (1, 2, 3):
            y = draw_text("NHOM THUAT TOAN", x, y, Colors.TEXT_MUTED, font_small)
            y = draw_text(self.algorithm_group_name, x, y, border_color, font)
            y += 6
            y = draw_text("CHON THUAT TOAN", x, y, Colors.TEXT_MUTED, font_small)

            options = self.selectable_algorithms()
            button_w = 170
            button_h = 34
            gap = 8
            for index, algorithm_name in enumerate(options):
                col = index % 2
                row = index // 2
                rect = pygame.Rect(
                    x + col * (button_w + gap), y + row * (button_h + gap),
                    button_w, button_h)
                self.algorithm_buttons[algorithm_name] = draw_button(
                    algorithm_name, rect, algorithm_name,
                    selected=algorithm_name == self.algorithm_name)
            y += ((len(options) + 1) // 2) * (button_h + gap) + 8

            pygame.draw.line(surface, Colors.PANEL_SEPARATOR,
                             (x, y), (panel.right - 15, y))
            y += 12
            y = draw_text("DIEU KHIEN", x, y, Colors.TEXT_MUTED, font_small)
            start_rect = pygame.Rect(x, y, 108, 36)
            pause_rect = pygame.Rect(x + 116, y, 108, 36)
            reset_rect = pygame.Rect(x + 232, y, 108, 36)
            self.control_buttons["start"] = draw_button(
                "start", start_rect, "START", selected=self.is_running)
            self.control_buttons["pause"] = draw_button(
                "pause", pause_rect, "PAUSE", selected=not self.is_running)
            self.control_buttons["reset"] = draw_button(
                "reset", reset_rect, "RESET", danger=True)
            y += 50
        else:
            y = draw_text(f"Thuat toan: {self.algorithm_name}", x, y,
                          border_color, font)
            y += 8

        pygame.draw.line(surface, Colors.PANEL_SEPARATOR,
                         (x, y), (panel.right - 15, y))
        y += 12
        status = "Dang chay" if self.is_running else "Cho START"
        if self.state == "DONE":
            status = "Hoan thanh"
        y = draw_text(f"Trang thai: {status}", x, y, Colors.TEXT_PRIMARY, font)
        y = draw_text(self.message, x, y, Colors.TEXT_SUBTITLE, font_small)
        if self.current_target is not None and self._has_condition(self.current_target):
            y = draw_text(f"Xu ly: {self._target_label(self.current_target)}",
                          x, y, Colors.TEXT_WARNING, font_small)

        total = max(len(self.farm_tiles), 1)
        done = len(self.done_tiles)
        y += 4
        y = draw_text(f"Tien do: {done}/{len(self.farm_tiles)} nhiem vu",
                      x, y, Colors.TEXT_SUCCESS, font)
        bar_rect = pygame.Rect(x, y + 2, panel_width - 30, 9)
        pygame.draw.rect(surface, (40, 40, 60), bar_rect, border_radius=4)
        fill_w = int(bar_rect.width * done / total)
        if fill_w > 0:
            fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, fill_w, bar_rect.height)
            pygame.draw.rect(surface, border_color, fill_rect, border_radius=4)
        y += 26

        if self.mode == 6:
            enemy_count = len(self.enemy_done_tiles)
            y = draw_text(f"Enemy chiem: {enemy_count} o",
                          x, y, Colors.TEXT_WARNING, font_small)
            destroy_remaining = self._enemy_destroy_remaining()
            if destroy_remaining > 0:
                y = draw_text(f"Enemy pha cay: {destroy_remaining:.1f}s",
                              x, y, Colors.TEXT_WARNING, font_small)

        if self.stats:
            y = draw_text("THONG KE", x, y, Colors.TEXT_MUTED, font_small)
            for key, val in list(self.stats.items())[:7]:
                y = draw_text(f"{key}: {val}", x, y, Colors.TEXT_STAT, font_small)

        if self.mode in (1, 2, 3):
            hint = "Click nut hoac dung Q/E de doi thuat toan"
        else:
            hint = "Nhan phim 1-6 de doi khu nhiem vu"
        draw_text(hint, x, panel.bottom - 30, Colors.TEXT_MUTED, font_small)
