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
            "Gan 4 loai cay ma khong vi pham rang buoc ke nhau",
            "Ngo/ca chua va lua mi/ca rot khong duoc trong canh nhau"),
        6: ("NGAY 6 - KHU 6: HANG CAY CAN BAO VE", "Minimax / Alpha-Beta",
            "Bao ve cay khi doi thu cung toi pha: gia su doi thu choi toi uu",
            "AGRI-1 dau tri voi ke pha hoai tren cac o can bao ve"),
    }
    OBJECTIVE_INFO = {
        1: ("Khoi phuc toan bo khu dat", "o dat"),
        2: ("Cuu toan bo cay nguy cap", "cay"),
        3: ("Cham soc toan bo vuon cay", "cay"),
        4: ("Khao sat va xu ly toan bo khu suong", "o"),
        5: ("Hoan thanh toan bo so do gieo trong", "o trong"),
        6: ("Bao ve cang nhieu cay cang tot", "cay"),
    }

    ENEMY_MOVE_SPEED = 2.2
    ENEMY_DESTROY_TIME = 1.5
    ENEMY_RETARGET_DELAY = 0.08
    IDS_ITERATION_DELAY = 0.65
    IDSA_ITERATION_DELAY = 0.65

    # ------------------------------------------------------------------ init
    def __init__(self, player, soil_layer, collision_sprites, farm_tiles,
                 mode=2, hidden_blocks=None, enemy_spawn=None,
                 terrain_costs=None, extra_walkable_tiles=None,
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
        self.terrain_costs = dict(terrain_costs or {})
        self.extra_walkable_tiles = set(extra_walkable_tiles or [])
        self.rows = len(self.soil_layer.grid)
        self.cols = len(self.soil_layer.grid[0]) if self.rows else 0
        self.spawn_tile = self._world_to_tile(self.player.rect.center)
        self.walkable_tiles = self._build_walkable_tiles()

        self.state = "CHOOSE_TARGET"
        self.current_target = None
        self.path = []
        self.chosen_target_path = None
        self.full_garden_plan = []
        self.full_garden_index = 0
        self.full_garden_stats = {}
        self.stop_after_current_target = False
        self.dfs_walk = None
        self.dfs_walk_index = 0
        self.dfs_explored_order = []
        self.ids_depth_limit = 0
        self.ids_search_timer = 0.0
        self.ids_search_start = None
        self.ids_search_goals = set()
        self.ids_search_blocked = set()
        self.ids_total_expansions = 0
        self.ids_unique_explored = set()
        self.idsa_f_limit = 0
        self.idsa_search_timer = 0.0
        self.idsa_search_start = None
        self.idsa_search_goal = None
        self.idsa_search_blocked = set()
        self.idsa_total_expansions = 0
        self.idsa_unique_explored = set()
        self.done_tiles = set()
        self.wait_time = 0.0
        self.counter = count()
        self.message = f"Khu {mode}: {self.algorithm_name}"
        self.is_running = mode not in (1, 2, 3, 4)
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

        # Fixed mode-3 sensor map used by the original demo.
        if self.mode == 3 and self.farm_tiles:
            self.dryness = {
                (33, 24): 40,
                (33, 25): 43,
                (33, 26): 46,
                (32, 26): 49,
                (31, 26): 52,
                (30, 26): 55,
                (29, 26): 58,
                (29, 25): 68,
                (29, 24): 66,
                (30, 24): 65,
                (30, 23): 68,
                (30, 22): 71,
                (31, 22): 69,
                (32, 22): 67,
                (33, 22): 65,
            }
            for tile in (
                    (33, 23), (32, 23), (32, 24), (32, 25),
                    (31, 23), (31, 24), (31, 25),
                    (30, 25), (29, 22), (29, 23)):
                self.dryness[tile] = 5
        else:
            self.dryness = {tile: random.randint(30, 95) for tile in self.farm_tiles}

        self.hc_scores = {}  # tile -> score, Ä‘á»ƒ váº½ lĂªn map
        self.hc_current_score = 0
        self.hc_best_neighbor_score = 0
        self.anneal_temperature = 80.0
        self.tile_conditions = self._build_tile_conditions()
        self.danger_level = self._build_danger_levels()

        # --- Mode 4 (Online Search): fog-of-war ---
        self.hidden_blocked = set()
        self.discovered_blocked = set()
        self.explored_tiles = set()
        self.belief_worlds = []
        self.belief_unknown_tiles = set()
        self.belief_context = None
        self.vision_radius = 1.5
        self.replan_count = 0
        self.and_or_intended_tile = None
        self.and_or_actual_tile = None
        self.and_or_outcome_probability = 0
        self.and_or_outcome_label = ""
        self.and_or_policy = {}          # contingent plan: {state -> next_tile}
        self.and_or_visited = set()      # tiles already visited this navigation
        self.and_or_backtrack_count = 0
        self.and_or_backtrack_reason = ""
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
        defaults = {4: "Online A*", 5: "Backtrack", 6: "Minimax"}
        return defaults.get(mode, "BFS")

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
        self._clear_full_garden_plan()
        self.dfs_walk = None
        self.dfs_walk_index = 0
        self.dfs_explored_order = []
        self._reset_ids_search()
        self._reset_idsa_search()
        self.current_target = None
        self.state = "CHOOSE_TARGET"
        self.stop_after_current_target = False
        self.bfs_explored.clear()
        self.astar_explored.clear()
        self.hc_scores.clear()
        self.anneal_temperature = 80.0
        self.and_or_intended_tile = None
        self.and_or_actual_tile = None
        self.and_or_outcome_probability = 0
        self.and_or_outcome_label = ""
        self.and_or_policy = {}
        self.and_or_visited = set()
        self.and_or_backtrack_count = 0
        self.and_or_backtrack_reason = ""
        self.belief_worlds = []
        self.belief_unknown_tiles = set()
        self.belief_context = None
        self.is_running = False
        self.message = f"Da chon {algorithm_name}. Nhan START"

        if self.mode == 5:
            self.done_tiles.clear()
            self.seed_plan = self._solve_csp_crop_plan()
        elif self.mode == 4:
            self.done_tiles.clear()
            self.discovered_blocked.clear()
            self.explored_tiles.clear()
            self.belief_worlds = []
            self.belief_unknown_tiles = set()
            self.belief_context = None
            self.replan_count = 0
            spawn_pos = self._tile_center(self.spawn_tile)
            self.player.pos.update(spawn_pos)
            self.player.rect.center = (round(spawn_pos.x), round(spawn_pos.y))
            self.player.hitbox.center = self.player.rect.center
            self._expand_vision(self.spawn_tile)

        return True

    def _clear_full_garden_plan(self):
        self.full_garden_plan = []
        self.full_garden_index = 0
        self.full_garden_stats = {}
        self.chosen_target_path = None

    def _advance_full_garden_plan(self, tile):
        if not self.full_garden_plan:
            return
        if (self.full_garden_index < len(self.full_garden_plan)
                and self.full_garden_plan[self.full_garden_index] == tile):
            self.full_garden_index += 1
        while self.full_garden_index < len(self.full_garden_plan):
            planned = self.full_garden_plan[self.full_garden_index]
            if (planned not in self.done_tiles
                    and planned not in self.discovered_blocked
                    and planned not in self.enemy_done_tiles):
                break
            self.full_garden_index += 1
        self._apply_full_plan_stats()

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
        if self.mode not in (1, 4) or not self.farm_tiles:
            return None

        walkable = set(self.farm_tiles)
        walkable.add(self.spawn_tile)
        walkable.update(self.extra_walkable_tiles)
        walkable.update(self.terrain_costs.keys())

        if self.mode == 4:
            return walkable

        for target_x, target_y in self.farm_tiles:
            x, y = self.spawn_tile
            step_x = 1 if target_x >= x else -1
            while x != target_x:
                walkable.add((x, y))
                x += step_x
            step_y = 1 if target_y >= y else -1
            while y != target_y:
                walkable.add((x, y))
                y += step_y
            walkable.add((target_x, target_y))
        return walkable

    def _neighbors(self, tile, blocked):
        x, y = tile
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < self.cols and 0 <= ny < self.rows \
                    and (nx, ny) not in blocked \
                    and (self.walkable_tiles is None
                         or (nx, ny) in self.walkable_tiles):
                yield (nx, ny)

    def _belief_neighbors(self, tile, blocked, goal=None, risk_map=None):
        x, y = tile
        candidates = []

        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            nt = (nx, ny)

            if 0 <= nx < self.cols and 0 <= ny < self.rows \
                    and nt not in blocked \
                    and (self.walkable_tiles is None
                         or nt in self.walkable_tiles):
                risk = risk_map.get(nt, 0.0) if risk_map else 0.0
                known_penalty = 0 if nt in self.explored_tiles else 1
                h = self._heuristic(nt, goal) if goal else 0
                stable_tie = (
                    nt[0] * 73856093
                    ^ nt[1] * 19349663
                    ^ tile[0] * 83492791
                    ^ tile[1] * 2654435761
                    ^ (goal[0] * 97531 if goal else 0)
                    ^ (goal[1] * 314159 if goal else 0)
                ) & 0xffff
                candidates.append((risk, known_penalty, h, stable_tie, nt))

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        for _, _, _, _, nt in candidates:
            yield nt

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

    def _ucs_task_cost(self, tile):
        condition = self._condition_for_tile(tile)
        action_time = {
            "critical": 2.0,
            "dry": 2.0,
            "pest": 3.5,
            "crow": 3.5,
            "replant": 4.0,
            "dead": 5.0,
        }
        rescue_value = {
            "critical": 1.4,
            "pest": 1.0,
            "crow": 1.0,
            "dry": 0.4,
            "replant": 0.0,
            "dead": 0.0,
        }
        dryness = max(0, min(self.dryness.get(tile, 50), 100)) / 100.0
        living_pressure = 0.4 * dryness if condition in (
            "critical", "dry", "pest", "crow") else 0
        return max(
            0.5,
            action_time.get(condition, 2.5)
            - rescue_value.get(condition, 0)
            - living_pressure)

    def _step_cost(self, current, next_tile):
        terrain_cost = self.terrain_costs.get(next_tile, 0)
        if self.mode == 1 and next_tile in self.farm_tiles:
            return 1 + terrain_cost + self._ucs_task_cost(next_tile)
        return 1 + terrain_cost

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
            "storm_damaged_plant": "storm_damaged_plant.png",
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

    def _draw_crop_icon(self, surface, x, y, crop_type):
        shadow = pygame.Surface((20, 10), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (30, 20, 10, 80), (0, 0, 20, 8))
        surface.blit(shadow, (x - 10, y + 4))

        if crop_type == "tomato":
            pygame.draw.circle(surface, (220, 40, 40), (x, y), 10)
            pygame.draw.circle(surface, (255, 100, 100), (x - 3, y - 3), 3)
            pygame.draw.polygon(surface, (40, 160, 40), [
                (x, y - 8), (x - 5, y - 5), (x - 2, y - 6),
                (x, y - 3), (x + 2, y - 6), (x + 5, y - 5)])
            pygame.draw.line(surface, (30, 120, 30), (x, y - 8), (x, y - 12), 2)
        elif crop_type == "corn":
            pygame.draw.ellipse(surface, (50, 160, 50), (x - 9, y - 8, 18, 20))
            pygame.draw.ellipse(surface, (240, 200, 20), (x - 6, y - 12, 12, 22))
            for dy in (-6, -2, 2, 6):
                pygame.draw.line(surface, (180, 140, 0), (x - 5, y + dy), (x + 5, y + dy), 1)
        elif crop_type == "carrot":
            pygame.draw.polygon(surface, (250, 110, 10), [
                (x - 6, y - 4), (x + 6, y - 4), (x, y + 10)])
            pygame.draw.line(surface, (60, 180, 60), (x, y - 4), (x - 4, y - 12), 2)
            pygame.draw.line(surface, (60, 180, 60), (x, y - 4), (x + 4, y - 12), 2)
            pygame.draw.line(surface, (60, 180, 60), (x, y - 4), (x, y - 14), 2)
        elif crop_type == "wheat":
            pygame.draw.line(surface, (220, 180, 60), (x, y - 12), (x, y + 8), 2)
            for dy in range(-8, 6, 4):
                pygame.draw.ellipse(surface, (240, 200, 80), (x - 5, y + dy, 5, 6))
                pygame.draw.ellipse(surface, (240, 200, 80), (x + 1, y + dy - 2, 5, 6))

    def _build_tile_conditions(self):
        conditions = {}
        if self.mode == 4:
            condition_cycle = ("dry", "critical", "dry", "pest", "dry")
            return {
                tile: condition_cycle[index % len(condition_cycle)]
                for index, tile in enumerate(self.farm_tiles)
            }
        if self.mode == 5:
            return {tile: "replant" for tile in self.farm_tiles}

        for index, tile in enumerate(self.farm_tiles):
            dryness = self.dryness.get(tile, 50)
            if dryness >= 88 and self.mode != 3:
                conditions[tile] = "dead"
            elif self.mode == 2 and dryness >= 70:
                conditions[tile] = "critical"
            elif self.mode == 2 and index % 4 == 0:
                conditions[tile] = "pest"
            elif dryness >= 65:
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

    def _build_danger_levels(self):
        priorities = {
            "dry": 4,
            "pest": 3,
            "crow": 3,
            "critical": 2,
            "dead": 1,
            "replant": 1,
        }
        levels = {}
        for tile in self.farm_tiles:
            levels[tile] = priorities.get(self._condition_for_tile(tile), 0)
        return levels

    def _care_value(self, tile):
        return self._condition_priority(tile) + self.dryness.get(tile, 50) / 10.0

    def _condition_asset(self, condition):
        if condition == "dead":
            return "dead_plant"
        if condition == "crow":
            return "crow_damage"
        if condition == "critical":
            return "storm_damaged_plant"
        if condition in ("pest", "replant"):
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
        self._advance_full_garden_plan(tile)
        self.current_target = None
        self.path = []
        self.state = "DONE" if self.stop_after_current_target else "CHOOSE_TARGET"
        self.stop_after_current_target = False
        self.wait_time = 0.45

    def _finish_pest_target(self, tile, condition):
        self.player.status = "down_idle"
        if condition == "crow":
            self.message = f"Duoi qua va bao ve cay tai {tile}"
        else:
            self.message = f"Phun sinh hoc xu ly sau benh tai {tile}"
        self.done_tiles.add(tile)
        self._advance_full_garden_plan(tile)
        self.current_target = None
        self.path = []
        self.state = "DONE" if self.stop_after_current_target else "CHOOSE_TARGET"
        self.stop_after_current_target = False
        self.wait_time = 0.45

    def _finish_cultivation_target(self, tile, message):
        self.message = message
        self.done_tiles.add(tile)
        self._advance_full_garden_plan(tile)
        self.current_target = None
        self.path = []
        self.state = "DONE" if self.stop_after_current_target else "CHOOSE_TARGET"
        self.stop_after_current_target = False
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

    def _target_summary(self, tile):
        labels = {
            "dry": "Cay kho",
            "critical": "Cay nguy cap",
            "pest": "Cay sau benh",
            "crow": "Cay bi qua pha",
            "dead": "Cay chet",
            "replant": "O can gieo",
        }
        condition = self._condition_for_tile(tile)
        return f"{tile} - {labels.get(condition, 'O can xu ly')}"

    def _objective_snapshot(self):
        """Return one progress format shared by every game mode."""
        total = len(self.farm_tiles)
        completed = len(self.done_tiles)
        blocked = len(self.discovered_blocked) if self.mode == 4 else 0
        lost = len(self.enemy_done_tiles) if self.mode == 6 else 0
        resolved = min(total, completed + blocked + lost)
        objective, unit = self.OBJECTIVE_INFO.get(
            self.mode, ("Hoan thanh toan bo khu vuc", "nhiem vu"))

        if self.state == "IDS_SEARCH":
            current = (
                f"IDS depth {self.ids_depth_limit} "
                f"tu {self.ids_search_start}")
        elif self.state == "IDSA_SEARCH":
            current = (
                f"IDSA f-limit {self.idsa_f_limit} "
                f"toi {self.idsa_search_goal}")
        elif self.current_target is None:
            current = "Chua chon"
        elif self._has_condition(self.current_target):
            current = self._target_summary(self.current_target)
        else:
            current = f"O {self.current_target}"

        if self.state == "DONE":
            current = "Da hoan tat"

        return {
            "title": objective,
            "unit": unit,
            "total": total,
            "completed": completed,
            "blocked": blocked,
            "lost": lost,
            "resolved": resolved,
            "remaining": max(0, total - resolved),
            "current": current,
        }

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
            g = self.danger_level.get(goal, 1)
            h = self._heuristic(start, goal)
            self.astar_current_fgh = (g + h, g, h)
            stats["f(n)"] = f"{g + h}"
            stats["g(n)"] = f"{g}"
            stats["h(n)"] = f"{h}"
            stats["Color priority"] = f"{g}"
            stats["Dryness goal"] = f"{self.dryness.get(goal, 50)}%"
        if self.mode == 1:
            self.bfs_explored = set(explored)
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
                self._neighbors, self.counter, self._step_cost))
        self.astar_current_fgh = fgh
        stats["Algorithm"] = self.algorithm_name
        stats["Target"] = f"{target}"
        if self.mode == 1:
            self.bfs_explored = set(explored)
        elif self.mode == 2:
            self.astar_explored = explored
        self.stats = stats
        self.chosen_target_path = path if target is not None else None
        return target

    def _reset_ids_search(self):
        self.ids_depth_limit = 0
        self.ids_search_timer = 0.0
        self.ids_search_start = None
        self.ids_search_goals = set()
        self.ids_search_blocked = set()
        self.ids_total_expansions = 0
        self.ids_unique_explored = set()

    def _begin_ids_search(self, start, goals):
        self._reset_ids_search()
        self.ids_search_start = start
        self.ids_search_goals = set(goals)
        self.ids_search_blocked = self._blocked_tiles()
        self.ids_search_blocked.discard(start)
        for goal in goals:
            self.ids_search_blocked.discard(goal)
        self.state = "IDS_SEARCH"
        self.player.direction.update(0, 0)
        self.message = f"IDS: bat dau lai tu {start}, depth limit 0"

    def _update_ids_search(self, dt):
        self.player.direction.update(0, 0)
        if self.ids_search_timer > 0:
            self.ids_search_timer = max(0, self.ids_search_timer - dt)
            return

        path, target, explored, cutoff = (
            algorithms.depth_limited_search_to_any_goal(
                self.ids_search_start,
                self.ids_search_goals,
                self.ids_search_blocked,
                self._neighbors,
                self.ids_depth_limit))

        # Show only this iteration so every larger limit visibly restarts.
        self.bfs_explored = set(explored)
        self.ids_unique_explored.update(explored)
        self.ids_total_expansions += len(explored)
        self.stats = {
            "Algorithm": "IDS",
            "Depth limit": self.ids_depth_limit,
            "Restart from": f"{self.ids_search_start}",
            "This iteration": len(explored),
            "Total expansions": self.ids_total_expansions,
            "Unique explored": len(self.ids_unique_explored),
            "Target": f"{target}",
        }

        if target is not None:
            self.current_target = target
            self.path = path
            self.state = "MOVE"
            self.message = (
                f"IDS tim thay {target} o depth {self.ids_depth_limit}")
            self.ids_search_timer = 0.0
            return

        if not cutoff:
            self.state = "DONE"
            self.message = "IDS: khong con node de duyet"
            return

        finished_limit = self.ids_depth_limit
        self.ids_depth_limit += 1
        self.ids_search_timer = self.IDS_ITERATION_DELAY
        self.message = (
            f"IDS limit {finished_limit} chua thay; "
            f"quay lai {self.ids_search_start}, tang len "
            f"{self.ids_depth_limit}")

    def _reset_idsa_search(self):
        self.idsa_f_limit = 0
        self.idsa_search_timer = 0.0
        self.idsa_search_start = None
        self.idsa_search_goal = None
        self.idsa_search_blocked = set()
        self.idsa_total_expansions = 0
        self.idsa_unique_explored = set()

    def _begin_idsa_search(self, start, goal):
        self._reset_idsa_search()
        self.idsa_search_start = start
        self.idsa_search_goal = goal
        self.idsa_f_limit = 1
        self.idsa_search_blocked = self._blocked_tiles()
        self.idsa_search_blocked.discard(start)
        self.idsa_search_blocked.discard(goal)
        self.current_target = goal
        self.state = "IDSA_SEARCH"
        self.player.direction.update(0, 0)
        self.message = (
            f"IDSA: bat dau lai tu {start}, f-limit "
            f"{self.idsa_f_limit}")

    def _update_idsa_search(self, dt):
        self.player.direction.update(0, 0)
        if self.idsa_search_timer > 0:
            self.idsa_search_timer = max(0, self.idsa_search_timer - dt)
            return

        next_limit, path, explored, hit_depth_guard = (
            algorithms.idastar_iteration(
                self.idsa_search_start,
                self.idsa_search_goal,
                self.idsa_search_blocked,
                self._neighbors,
                self._search_heuristic,
                self.idsa_f_limit))

        self.astar_explored = set(explored)
        self.idsa_unique_explored.update(explored)
        self.idsa_total_expansions += len(explored)
        self.stats = {
            "Algorithm": "IDSA",
            "f limit": self.idsa_f_limit,
            "Restart from": f"{self.idsa_search_start}",
            "This iteration": len(explored),
            "Total expansions": self.idsa_total_expansions,
            "Unique explored": len(self.idsa_unique_explored),
            "Target": f"{self.idsa_search_goal}",
        }
        if hit_depth_guard:
            self.stats["Stopped by guard"] = "yes"

        if path is not None:
            self.path = path
            self.state = "MOVE"
            self.message = (
                f"IDSA tim thay {self.idsa_search_goal} voi "
                f"f-limit {self.idsa_f_limit}")
            self.idsa_search_timer = 0.0
            return

        if next_limit == algorithms.INF:
            self.state = "DONE"
            self.message = "IDSA: khong con node de duyet"
            return

        finished_limit = self.idsa_f_limit
        self.idsa_f_limit += 1
        self.idsa_search_timer = self.IDSA_ITERATION_DELAY
        self.message = (
            f"IDSA f-limit {finished_limit} chua thay; "
            f"quay lai {self.idsa_search_start}, tang len "
            f"{self.idsa_f_limit}")

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
                self.bfs_explored = explored
                self.stats = {
                    "Nodes explored": len(explored),
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
        """Local state value: higher dryness means higher priority."""
        return algorithms.hill_score(
            tile, start, self.dryness, self._heuristic)

    def _local_scores(self):
        scores = {
            tile: self._hill_score(tile, self.spawn_tile)
            for tile in self.farm_tiles
        }
        self.hc_scores = scores
        return scores

    def _nearest_remaining(self, remaining, start):
        return min(remaining, key=lambda tile: (self._heuristic(start, tile), tile))

    def _remaining_neighbors(self, remaining, center):
        return [
            tile for tile in remaining
            if abs(tile[0] - center[0]) + abs(tile[1] - center[1]) == 1
        ]

    def _set_local_stats(self, algorithm, target, score, extra=None):
        stats = {
            "Local algorithm": algorithm,
            "Current score": f"{score:.0f}",
            "Target": f"{target}" if target is not None else "None",
            "Dryness": (
                f"{self.dryness.get(target, 0)}%"
                if target is not None else "0%"),
        }
        if extra:
            stats.update(extra)
        self.stats = stats
        self.hc_current_score = score
        self.hc_best_neighbor_score = score

    def _hill_climbing_step_choose(self, remaining, start):
        """Hill Climbing tung buoc: gan nhat truoc, sau do leo sang lang gieng tot hon."""
        scores = self._local_scores()

        if start not in self.dryness:
            target = self._nearest_remaining(remaining, start)
            current_score = scores.get(target, 0)
            self._set_local_stats("Hill Climbing", target, current_score, {
                "Hill steps": 0,
                "Trace": [target],
                "Start rule": "Nearest target first",
            })
            return target

        current_score = scores.get(start, 0)
        neighbors = self._remaining_neighbors(remaining, start)
        better_neighbors = [
            tile for tile in neighbors
            if scores.get(tile, 0) > current_score
        ]
        if not better_neighbors:
            self._set_local_stats("Hill Climbing", None, current_score, {
                "Hill steps": 0,
                "Trace": [start],
                "Stop rule": "No better neighbor",
                "Remaining tiles": len(remaining),
                "Coverage": f"{len(self.done_tiles)}/{len(self.farm_tiles)}",
                "Suggestion": "Use Restart Hill for full coverage",
            })
            self.state = "DONE"
            return None

        target = max(
            better_neighbors,
            key=lambda tile: (scores.get(tile, 0), -self._heuristic(start, tile), tile))
        target_score = scores.get(target, 0)
        self._set_local_stats("Hill Climbing", target, target_score, {
            "Hill steps": 1,
            "Trace": [start, target],
            "Stop rule": "Move to best better neighbor",
        })
        return target

    def _climb_from_tile(self, start_tile, remaining, scores):
        current = start_tile
        trace = [current]
        current_score = scores.get(current, 0)
        while True:
            neighbors = [
                tile for tile in self._remaining_neighbors(remaining, current)
                if tile != current and scores.get(tile, 0) > current_score
            ]
            if not neighbors:
                return current, current_score, trace
            current = max(
                neighbors,
                key=lambda tile: (scores.get(tile, 0), -self._heuristic(current, tile), tile))
            current_score = scores.get(current, 0)
            trace.append(current)

    def _restart_hill_choose(self, remaining, start):
        """Restart Hill: leo nhu Hill, neu ket thi khoi dong lai o muc tieu khac."""
        scores = self._local_scores()
        if start not in self.dryness:
            target = self._nearest_remaining(remaining, start)
            self._set_local_stats("Restart Hill", target, scores.get(target, 0), {
                "Start rule": "Nearest target first",
                "Restarts": 0,
                "Trace": [target],
            })
            return target

        current_score = scores.get(start, 0)
        better_neighbors = [
            tile for tile in self._remaining_neighbors(remaining, start)
            if scores.get(tile, 0) > current_score
        ]
        if better_neighbors:
            target = max(
                better_neighbors,
                key=lambda tile: (scores.get(tile, 0),
                                  -self._heuristic(start, tile), tile))
            self._set_local_stats("Restart Hill", target, scores.get(target, 0), {
                "Restarts": 0,
                "Trace": [start, target],
                "Rule": "Continue current hill",
            })
            return target

        target = max(
            remaining,
            key=lambda tile: (scores.get(tile, 0), -self._heuristic(start, tile), tile))
        self._set_local_stats("Restart Hill", target, scores.get(target, 0), {
            "Restarts": 1,
            "Trace": [target],
            "Rule": "Restart after local optimum",
        })
        return target

    def _local_beam_step_choose(self, remaining, start, beam_width=3):
        """Local Beam: giu k ung vien, sinh tat ca lang gieng, chon k tot nhat."""
        scores = self._local_scores()
        if start not in self.dryness:
            beam = sorted(
                remaining,
                key=lambda tile: (self._heuristic(start, tile), -scores.get(tile, 0), tile)
            )[:beam_width]
        else:
            seed = self._remaining_neighbors(remaining, start)
            if not seed:
                seed = list(remaining)
            beam = sorted(
                seed,
                key=lambda tile: (-scores.get(tile, 0), self._heuristic(start, tile), tile)
            )[:beam_width]

        children = []
        for tile in beam:
            children.extend(self._remaining_neighbors(remaining, tile))
        pool = set(beam) | set(children)
        next_beam = sorted(
            pool,
            key=lambda tile: (-scores.get(tile, 0), self._heuristic(start, tile), tile)
        )[:max(1, min(beam_width, len(pool)))]
        target = next_beam[0] if next_beam else None
        target_score = scores.get(target, 0) if target is not None else 0
        self._set_local_stats("Local Beam", target, target_score, {
            "Beam width": beam_width,
            "Beam": f"{next_beam}",
            "Generated": len(children),
            "Rule": "Best of global successor beam",
        })
        return target

    def _annealing_step_choose(self, remaining, start):
        """Simulated Annealing: chon lang gieng, chap nhan buoc te hon theo nhiet do."""
        scores = self._local_scores()
        if start not in self.dryness:
            target = self._nearest_remaining(remaining, start)
            self._set_local_stats("Annealing", target, scores.get(target, 0), {
                "Start rule": "Nearest target first",
                "Temperature": f"{self.anneal_temperature:.1f}",
            })
            return target

        neighbors = self._remaining_neighbors(remaining, start)
        if not neighbors:
            self._set_local_stats("Annealing", None, scores.get(start, 0), {
                "Stop rule": "No neighbor state",
                "Temperature": f"{self.anneal_temperature:.1f}",
            })
            self.state = "DONE"
            return None

        current_score = scores.get(start, 0)
        ordered = list(neighbors)
        random.shuffle(ordered)
        accepted = None
        accepted_delta = 0
        accepted_worse = False
        for candidate in ordered:
            delta = scores.get(candidate, 0) - current_score
            if delta >= 0:
                accepted = candidate
                accepted_delta = delta
                break
            probability = math.exp(delta / max(self.anneal_temperature, 0.001))
            if random.random() < probability:
                accepted = candidate
                accepted_delta = delta
                accepted_worse = True
                break

        self.anneal_temperature = max(0.1, self.anneal_temperature * 0.90)
        if accepted is None:
            self._set_local_stats("Annealing", None, current_score, {
                "Rejected": len(ordered),
                "Temperature": f"{self.anneal_temperature:.1f}",
                "Stop rule": "No accepted neighbor",
            })
            self.state = "DONE"
            return None

        self._set_local_stats("Annealing", accepted, scores.get(accepted, 0), {
            "Delta": f"{accepted_delta:.0f}",
            "Accepted worse": "yes" if accepted_worse else "no",
            "Temperature": f"{self.anneal_temperature:.1f}",
        })
        return accepted

    def _local_search_choose(self, remaining, start):
        """Chon muc tieu bang thuat toan local search dang duoc chon."""
        if self.algorithm_name == "Hill Climbing":
            return self._hill_climbing_step_choose(remaining, start)
        if self.algorithm_name == "Restart Hill":
            return self._restart_hill_choose(remaining, start)
        if self.algorithm_name == "Local Beam":
            return self._local_beam_step_choose(remaining, start)
        if self.algorithm_name == "Annealing":
            return self._annealing_step_choose(remaining, start)

        current_best, current_score, self.hc_scores, self.stats = (
            algorithms.local_search_choose(
                self.algorithm_name, remaining, start, self.dryness,
                self._heuristic))
        trace = self.stats.get("Trace")
        if self.algorithm_name == "Hill Climbing" and trace:
            self.chosen_target_path = list(trace)
            self.stop_after_current_target = True
        self.hc_current_score = current_score
        self.hc_best_neighbor_score = current_score
        return current_best

    # -------------------------------------------------------------- MODE 4
    def _expand_vision(self, center):
        """Má»Ÿ rá»™ng vĂ¹ng Ä‘Ă£ khĂ¡m phĂ¡ quanh center."""
        before_explored = set(self.explored_tiles)
        before_blocked = set(self.discovered_blocked)
        algorithms.expand_vision(
            center, self.vision_radius, self.rows, self.cols,
            self.hidden_blocked, self.explored_tiles,
            self.discovered_blocked)
        observed_tiles = self.explored_tiles - before_explored
        observed_blocked = self.discovered_blocked - before_blocked
        observed_free = observed_tiles - self.discovered_blocked
        if self.belief_worlds:
            self.belief_worlds = algorithms.update_belief_worlds(
                self.belief_worlds, observed_free, observed_blocked)
            self.belief_unknown_tiles -= observed_free
            self.belief_unknown_tiles -= observed_blocked

    def _online_plan(self, start, goal):
        """Online Search: chá»‰ biáº¿t váº­t cáº£n Ä‘Ă£ khĂ¡m phĂ¡, phĂ¡t hiá»‡n thĂªm khi Ä‘i."""
        if goal in self.discovered_blocked:
            return []

        if self.algorithm_name in (
                "Belief A*", "Belief Init-Goal A*", "AND-OR Search"):
            blocked = self._blocked_tiles()
            blocked.discard(start)
            blocked.discard(goal)
            initial_belief = {
                (x, y)
                for y in range(self.rows)
                for x in range(self.cols)
                if (x, y) not in self.explored_tiles
                and (self.walkable_tiles is None
                     or (x, y) in self.walkable_tiles)
            }
            if self.algorithm_name == "AND-OR Search":
                search_depth = min(
                    12, max(5, self._heuristic(start, goal) + 2))
                path, policy, explored, belief_stats = algorithms.and_or_search(
                    start, goal, blocked, initial_belief, self._neighbors,
                    self._search_heuristic, self.counter,
                    max_depth=search_depth
                )
                self.and_or_policy = policy
                if not path:
                    path, explored, fallback_stats = algorithms.belief_astar(
                        start, goal, blocked, initial_belief,
                        self._neighbors, self._search_heuristic, self.counter)
                    belief_stats.update({
                        "Contingent fallback": "Intended branch; rebuild next state",
                        "Fallback path": fallback_stats.get("Path length", 0),
                    })
            elif self.algorithm_name == "Belief Init-Goal A*":
                possible_starts = {start}
                possible_starts.update(
                    tile for tile in self._neighbors(start, blocked)
                    if tile not in self.explored_tiles
                )
                possible_goals = {goal}
                possible_goals.update(self._neighbors(goal, blocked))
                path, explored, _, chosen_goal, belief_stats = (
                    algorithms.belief_init_goal_astar(
                        possible_starts, possible_goals, blocked,
                        initial_belief, self._neighbors,
                        self._search_heuristic, self.counter,
                        preferred_start=start
                    )
                )
                if chosen_goal is not None and chosen_goal != goal:
                    if goal in self._neighbors(chosen_goal, blocked):
                        path.append(goal)
                        belief_stats["Path length"] = len(path)
                        belief_stats["Resolved goal"] = goal
            else:
                selected_unknowns = algorithms.select_relevant_unknown_tiles(
                    initial_belief, start, goal, self._search_heuristic,
                    limit=6)
                belief_context = (
                    start, goal, frozenset(selected_unknowns)
                )
                if self.belief_context != belief_context:
                    self.belief_context = belief_context
                    self.belief_unknown_tiles = set(selected_unknowns)
                    self.belief_worlds = algorithms.generate_belief_worlds(
                        selected_unknowns, max_hidden=2)
                risk_map = algorithms.belief_risk_map(
                    self.belief_worlds, self.belief_unknown_tiles)
                belief_neighbors = lambda tile, blocked_set: (
                    self._belief_neighbors(tile, blocked_set, goal, risk_map)
                )
                path, explored, belief_stats = algorithms.belief_world_astar(
                    start, goal, blocked, self.belief_worlds, risk_map,
                    self.belief_unknown_tiles, belief_neighbors,
                    self._search_heuristic, self.counter
                )
            self.astar_explored = explored
        else:
            path = self._astar(start, goal)
            belief_stats = {}

        explored_in_area = self.explored_tiles & (
            set(self.farm_tiles) | {self.spawn_tile})
        total_explored = len(explored_in_area)
        total_map = len(self.farm_tiles) + 1
        self.stats = {
            "Algorithm": self.algorithm_name,
            "Current target": f"{goal}",
            "Explored tiles": f"{total_explored}",
            "Map coverage": f"{total_explored * 100 // max(total_map, 1)}%",
            "Replanned": f"{self.replan_count} lan",
            "Blocks found": (
                f"{len(self.discovered_blocked)}/{len(self.hidden_blocked)}"),
            "Unknown blocks": len(
                self.hidden_blocked - self.discovered_blocked),
        }
        if self.algorithm_name == "Online A*":
            self.stats["Online policy"] = "Plan known map; patch on discovery"
        elif self.algorithm_name == "Belief A*":
            self.stats["Online policy"] = "Possible worlds over hidden blocks"
        elif self.algorithm_name == "Belief Init-Goal A*":
            self.stats["Online policy"] = "Belief over start and goal"
        else:
            self.stats["Online policy"] = "Plan for success/failure outcomes"
            self.stats["Backtracking"] = "logical rollback on failed branch"
            self.stats["Backtracks"] = self.and_or_backtrack_count
        self.stats.update(belief_stats)
        return path

    def _sample_and_or_outcome(self, current, intended):
        dx = intended[0] - current[0]
        dy = intended[1] - current[1]
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
        intended_direction = (dx, dy)
        outcomes = [intended_direction]
        outcomes.extend(
            direction for direction in directions
            if direction != intended_direction)
        probabilities = (40, 20, 20, 20)
        chosen = random.choices(outcomes, weights=probabilities, k=1)[0]
        probability = probabilities[outcomes.index(chosen)]
        actual = (current[0] + chosen[0], current[1] + chosen[1])

        blocked = (
            actual not in self.walkable_tiles
            or actual in self.hidden_blocked
            or actual in self.discovered_blocked)
        if blocked:
            if actual in self.hidden_blocked:
                self.discovered_blocked.add(actual)
            return current, probability, f"Blocked at {actual}; stay"
        if actual == intended:
            return actual, probability, "Correct direction"
        return actual, probability, "Wrong direction; continue policy"

    def _and_or_policy_path(self, start, goal, max_steps=50):
        """Extract a path from start to goal by following self.and_or_policy."""
        path = []
        current = start
        seen = {start} | self.and_or_visited
        policy = self.and_or_policy
        for _ in range(max_steps):
            if current == goal or current not in policy:
                break
            nxt = policy[current]
            if nxt is None or nxt in seen:
                break
            path.append(nxt)
            seen.add(nxt)
            current = nxt
        return path

    def _and_or_replan_from_current(self, reason, wait_time=0.18):
        current = self._world_to_tile(self.player.rect.center)
        self.replan_count += 1
        self._clear_full_garden_plan()
        self.and_or_visited = {current}
        self.and_or_intended_tile = None
        self.and_or_actual_tile = None
        self.path = self._online_plan(current, self.current_target)
        if not self.path and current != self.current_target:
            self.state = "CHOOSE_TARGET"
        self.stats.update({
            "Contingency": f"soft replan: {reason}",
            "Replan from": f"{current}",
            "Visited states": len(self.and_or_visited),
        })
        self.message = (
            f"AND-OR: {reason}, re-plan mem lan {self.replan_count}")
        self.wait_time = wait_time

    def _and_or_logical_backtrack(self, reason, wait_time=0.18):
        self.and_or_backtrack_count += 1
        self.and_or_backtrack_reason = reason
        self._and_or_replan_from_current(reason, wait_time)
        self.stats.update({
            "Backtracks": self.and_or_backtrack_count,
            "Backtrack reason": reason,
            "Backtrack type": "logical rollback",
        })
        self.message = (
            f"AND-OR backtrack #{self.and_or_backtrack_count}: {reason}")

    def _and_or_reset_navigation(self):
        """Clear navigation state when changing target."""
        self.and_or_visited.clear()
        self.and_or_policy = {}

    def _finish_and_or_outcome(self):
        intended = self.and_or_intended_tile
        actual = self.and_or_actual_tile
        probability = self.and_or_outcome_probability
        label = self.and_or_outcome_label
        self.and_or_intended_tile = None
        self.and_or_actual_tile = None

        self.stats.update({
            "Intended move": f"{intended}",
            "Actual outcome": f"{actual}",
            "Outcome probability": f"{probability}%",
            "Outcome": label,
            "Visited states": len(self.and_or_visited),
        })
        self._expand_vision(actual)

        # Always evaluate the tile the robot actually reached first,
        # whether the move followed the intended direction or drifted.
        if actual == self.current_target and actual not in self.done_tiles:
            self.path = []
            self._and_or_reset_navigation()
            self.state = self._state_after_arrival(actual)
            self.stats.update({
                "Observed tile": f"{actual}",
                "Observed action": "Current target reached",
            })
            self.message = f"AND-OR xet o {actual}: dung muc tieu, xu ly cay"
            self.wait_time = 0.18
            return

        # ---- SUCCESS: reached intended tile ----
        if actual == intended:
            self.and_or_visited.add(actual)
            self.path.pop(0)
            return

        # ---- BLOCKED: stayed in place ----
        if label.startswith("Blocked"):
            self._and_or_logical_backtrack(f"Bi chan tai {actual}", 0.25)
            return

        # ---- DRIFT: ended up on a different tile ----

        if actual in self.farm_tiles and actual not in self.done_tiles:
            previous_target = self.current_target
            self.current_target = actual
            self.path = []
            self._and_or_reset_navigation()
            self.state = self._state_after_arrival(actual)
            self.stats.update({
                "Observed tile": f"{actual}",
                "Observed action": "Other plant reached; handle now",
                "Previous target": f"{previous_target}",
            })
            self.message = (
                f"AND-OR xet o {actual}: lech sang cay khac, xu ly cay nay")
            self.wait_time = 0.18
            return

        # Drifted to a tile we already visited -> logical backtrack.
        if actual in self.and_or_visited:
            self._and_or_logical_backtrack(
                f"Quay lai o da tham {actual}", 0.18)
            return

        # New tile — try following contingent policy.
        self.and_or_visited.add(actual)
        contingency_path = self._and_or_policy_path(
            actual, self.current_target)
        if contingency_path:
            self.path = contingency_path
            self.stats.update({
                "Contingency": "followed policy",
                "Policy state": f"{actual}",
                "Policy size": len(self.and_or_policy),
            })
            self.message = (
                f"AND-OR drift {probability}%: {label}; "
                f"theo ke hoach du phong")
            self.wait_time = 0.15
            return

        # New tile but no policy covers it -> logical backtrack.
        self._and_or_logical_backtrack(
            f"Khong co ke hoach cho {actual}", 0.18)

    # -------------------------------------------------------------- MODE 5

    def _solve_csp_crop_plan(self):
        """CSP Backtracking: gĂ¡n corn/tomato sao cho 2 Ă´ ká» khĂ¡c loáº¡i."""
        assignment, self.csp_steps, self.stats = algorithms.solve_csp_crop_plan(
            self.farm_tiles, algorithm=self.algorithm_name
        )
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

    def _full_plan_enabled(self):
        return self.mode in (1, 2, 3, 4, 5)

    def _apply_full_plan_stats(self):
        if not self.full_garden_plan:
            return
        completed = len(self.done_tiles)
        total = len(self.farm_tiles)
        stats = dict(self.stats or {})
        stats.update(self.full_garden_stats)
        stats["Algorithm"] = self.algorithm_name
        stats["Planning mode"] = "Full garden search"
        stats["Plan target"] = (
            f"{min(self.full_garden_index + 1, len(self.full_garden_plan))}/"
            f"{len(self.full_garden_plan)}")
        stats["Completed"] = f"{completed}/{total}"
        self.stats = stats

    def _build_full_garden_plan(self, remaining, start):
        self._clear_full_garden_plan()
        if not remaining:
            return []

        if self.mode == 3:
            plan, scores, stats = algorithms.build_local_search_full_plan(
                self.algorithm_name, remaining, start, self.dryness,
                self._heuristic)
            self.hc_scores = scores
            self.hc_current_score = (
                scores.get(plan[0], 0) if plan else 0)
            self.hc_best_neighbor_score = self.hc_current_score
            self.full_garden_plan = list(plan)
            self.full_garden_stats = stats
        elif self.mode in (1, 2):
            blocked = self._blocked_tiles()
            blocked.discard(start)
            for goal in remaining:
                blocked.discard(goal)
            plan, explored, fgh, stats = (
                algorithms.build_full_garden_plan_by_algorithm(
                    self.algorithm_name, start, remaining, blocked,
                    self._neighbors, self._search_heuristic, self.counter,
                    self._step_cost))
            self.full_garden_plan = list(plan)
            self.full_garden_stats = stats
            self.astar_current_fgh = fgh
            if self.mode == 1:
                self.bfs_explored = set(explored)
            else:
                self.astar_explored = set(explored)
        elif self.mode == 4:
            known_blocked = self._blocked_tiles()
            known_blocked.discard(start)
            for goal in remaining:
                known_blocked.discard(goal)
            plan, explored, fgh, stats = (
                algorithms.build_full_garden_plan_by_algorithm(
                    "A*", start, remaining, known_blocked,
                    self._neighbors, self._search_heuristic, self.counter,
                    self._step_cost))
            self.full_garden_plan = list(plan)
            self.full_garden_stats = stats
            self.full_garden_stats["Algorithm"] = self.algorithm_name
            self.full_garden_stats["Online policy"] = (
                "Full plan over known map; rebuild on discovery")
            self.astar_current_fgh = fgh
            self.astar_explored = set(explored)
        elif self.mode == 5:
            plan = []
            unvisited = list(remaining)
            current = start
            while unvisited:
                target = min(
                    unvisited,
                    key=lambda tile: (self._heuristic(current, tile), tile))
                plan.append(target)
                unvisited.remove(target)
                current = target
            self.full_garden_plan = plan
            self.full_garden_stats = {
                "Algorithm": self.algorithm_name,
                "Planning mode": "Full garden search",
                "Plan targets": len(plan),
                "CSP route": "Nearest seed-plan order",
            }

        self._apply_full_plan_stats()
        return self.full_garden_plan

    def _next_full_garden_target(self, remaining, start):
        if not self._full_plan_enabled():
            return self._choose_target(remaining, start)

        remaining_set = set(remaining)
        if not self.full_garden_plan:
            self._build_full_garden_plan(remaining, start)

        while self.full_garden_index < len(self.full_garden_plan):
            target = self.full_garden_plan[self.full_garden_index]
            if target in remaining_set:
                self.chosen_target_path = None
                self._apply_full_plan_stats()
                return target
            self.full_garden_index += 1

        if remaining_set:
            # The old plan was exhausted or invalidated by online discovery.
            self._build_full_garden_plan(remaining, start)
            if not self.full_garden_plan:
                return None
            return self._next_full_garden_target(remaining, start)
        return None

    def _choose_target(self, remaining, start):
        if self.mode == 3:
            return self._local_search_choose(remaining, start)
        if self.mode == 6:
            return self._minimax_choose(remaining, start)
        if self.mode == 4 and self.algorithm_name == "AND-OR Search":
            if self.current_target in remaining:
                self.chosen_target_path = self._online_plan(
                    start, self.current_target)
                self.stats.update({
                    "Target rule": "AND-OR continue current target",
                    "Selected target": f"{self.current_target}",
                })
                return self.current_target
        if self.mode == 1:
            return self._search_first_target(start, remaining)
        if self.mode == 2:
            return max(
                remaining,
                key=lambda tile: (
                    self.danger_level.get(tile, 1),
                    -self._heuristic(start, tile),
                    tile))
        # Mode 4,5: giu hanh vi demo rieng, chon muc tieu gan truoc.
        best = None
        for tile in remaining:
            path = self._path_for_mode(start, tile)
            if path or start == tile:
                distance = len(path)
                sort_key = (distance, tile)
                if best is None or sort_key < best[0]:
                    best = (sort_key, tile)
        if best is None:
            return None
        target = best[1]
        self.stats.update({
            "Target rule": "Nearest reachable target",
            "Selected target": f"{target}",
            "Selection reason": (
                f"Shortest planned path = {best[0][0]}; "
                "coordinate breaks ties"),
        })
        return target

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

        if self.state == "IDS_SEARCH":
            self._update_ids_search(dt)
            return

        if self.state == "IDSA_SEARCH":
            self._update_idsa_search(dt)
            return

        # --- Stuck detection: náº¿u Ä‘ang MOVE mĂ  position khĂ´ng Ä‘á»•i > 1.5s ---
        if (self.state == "MOVE"
                and not (self.mode == 4
                         and self.algorithm_name == "AND-OR Search")):
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

            target = self._next_full_garden_target(remaining, start)
            if target is None:
                self.state = "DONE"
                if self.mode == 3:
                    self.message = (
                        f"{self.algorithm_name} dung: "
                        "khong co buoc local hop le tiep theo.")
                else:
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
                    self._clear_full_garden_plan()
                    self.message = (
                        f"Online: o {target} bi chan, "
                        f"re-plan (lan {self.replan_count})")
                    self.wait_time = 0.4
                    return
                self.done_tiles.add(target)
                self._advance_full_garden_plan(target)
                self.message = f"Bo qua {target}: khong co duong."
                return

            self.state = "MOVE"
            self._apply_full_plan_stats()
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
                    self._clear_full_garden_plan()
                    self.message = (
                        f"Online: phat hien vat can tai {next_tile}, "
                        f"re-plan (lan {self.replan_count})")
                    self.wait_time = 0.4
                    return

            if self.mode == 4 and self.algorithm_name == "AND-OR Search":
                current_tile = self._world_to_tile(self.player.rect.center)

                # Normal AND-OR forward movement with stochastic drift.
                if self.and_or_actual_tile is None:
                    actual, probability, label = self._sample_and_or_outcome(
                        current_tile, next_tile)
                    self.and_or_intended_tile = next_tile
                    self.and_or_actual_tile = actual
                    self.and_or_outcome_probability = probability
                    self.and_or_outcome_label = label
                    self.stats.update({
                        "Intended move": f"{next_tile}",
                        "Sampled outcome": f"{actual}",
                        "Outcome probability": f"{probability}%",
                    })
                reached = self._set_player_direction_to(
                    self._tile_center(self.and_or_actual_tile))
                if reached:
                    self._finish_and_or_outcome()
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
            if self.mode == 4 and tile not in self.explored_tiles:
                continue
            self._draw_task_tile_marker(surface, offset, tile)

        if self.mode in (1, 2, 3, 4, 6):
            for tile in self.farm_tiles:
                if tile in self.enemy_done_tiles:
                    continue
                if self.mode == 4 and tile in self.hidden_blocked:
                    continue
                if self.mode == 4 and tile not in self.explored_tiles:
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

        # --- Mode 3: Local Search scores trĂªn má»—i Ă´ ---
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
            for tile, crop in self.seed_plan.items():
                world = self._tile_center(tile)
                sx = int(world.x - offset.x)
                sy = int(world.y - offset.y)
                # Ná»n
                s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4),
                                   pygame.SRCALPHA)
                s.fill((90, 60, 40, 80))
                pygame.draw.rect(s, (120, 80, 50, 150), s.get_rect(), 2,
                                 border_radius=6)
                surface.blit(s, (sx - TILE_SIZE // 2 + 2,
                                 sy - TILE_SIZE // 2 + 2))
                self._draw_crop_icon(surface, sx, sy, crop)

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
        selectable_modes = (1, 2, 3, 4, 5)
        panel_height = 680 if self.mode in selectable_modes else 430
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

        if self.mode in selectable_modes:
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
        objective = self._objective_snapshot()
        status = "Dang chay" if self.is_running else "Cho START"
        if self.state == "DONE":
            status = "Hoan thanh"
        y = draw_text(f"MUC TIEU: {objective['title']}",
                      x, y, Colors.TEXT_PRIMARY, font)
        y = draw_text(f"Trang thai: {status}", x, y, Colors.TEXT_SUBTITLE,
                      font_small)
        y = draw_text(
            f"TIEN DO: {objective['resolved']}/{objective['total']} "
            f"{objective['unit']} da giai quyet",
            x, y, Colors.TEXT_SUCCESS, font)
        bar_rect = pygame.Rect(x, y + 2, panel_width - 30, 9)
        pygame.draw.rect(surface, (40, 40, 60), bar_rect, border_radius=4)
        total = max(objective["total"], 1)
        fill_w = int(bar_rect.width * objective["resolved"] / total)
        if fill_w > 0:
            fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, fill_w, bar_rect.height)
            pygame.draw.rect(surface, border_color, fill_rect, border_radius=4)
        y += 20

        y = draw_text(f"DANG XU LY: {objective['current']}",
                      x, y, Colors.TEXT_WARNING, font_small)

        if self.full_garden_plan:
            y = draw_text(
                f"PLAN: {len(self.full_garden_plan)} target | "
                f"dang xu ly {min(self.full_garden_index + 1, len(self.full_garden_plan))}/"
                f"{len(self.full_garden_plan)}",
                x, y, Colors.TEXT_STAT, font_small)
            y = draw_text(
                f"Hoan thanh: {objective['completed']}/{objective['total']} o",
                x, y, Colors.TEXT_SUCCESS, font_small)

        if self.mode == 4 and objective["blocked"]:
            y = draw_text(
                f"Da phat hien: {objective['blocked']} o bi chan",
                x, y, Colors.TEXT_WARNING, font_small)

        if self.mode == 6:
            y = draw_text(
                f"Bao ve: {objective['completed']} | "
                f"Bi pha: {objective['lost']} | "
                f"Con lai: {objective['remaining']}",
                x, y, Colors.TEXT_WARNING, font_small)
            destroy_remaining = self._enemy_destroy_remaining()
            if destroy_remaining > 0:
                y = draw_text(f"Enemy pha cay: {destroy_remaining:.1f}s",
                              x, y, Colors.TEXT_WARNING, font_small)

        if self.stats:
            y = draw_text("THONG KE", x, y, Colors.TEXT_MUTED, font_small)
            stats_bottom = panel.bottom - 42
            for key, val in list(self.stats.items())[:7]:
                if y + font_small.get_height() + 6 > stats_bottom:
                    break
                y = draw_text(f"{key}: {val}", x, y, Colors.TEXT_STAT, font_small)

        if self.mode in (1, 2, 3):
            hint = "Click nut hoac dung Q/E de doi thuat toan"
        else:
            hint = "Nhan phim 1-6 de doi khu nhiem vu"
        draw_text(hint, x, panel.bottom - 30, Colors.TEXT_MUTED, font_small)
