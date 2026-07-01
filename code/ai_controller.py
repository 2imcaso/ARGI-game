from collections import deque
from itertools import count
import heapq
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
            "Khong cung crop; corn-tomato va wheat-carrot cung bi cam"),
        6: ("NGAY 6 - KHU 6: DAU TRUONG BAO VE VUON", "Adversarial Search",
            "AGRI-1 va qua vao vuon tu hai phia doi xung",
            "MAX bao ve; MIN pha huy; CHANCE mo phong rui ro"),
    }
    OBJECTIVE_INFO = {
        1: ("Khoi phuc toan bo khu dat", "o dat"),
        2: ("Cuu toan bo cay nguy cap", "cay"),
        3: ("Cham soc toan bo vuon cay", "cay"),
        4: ("Khao sat va xu ly toan bo khu suong", "o"),
        5: ("Hoan thanh toan bo so do gieo trong", "o trong"),
        6: ("Sua cay bi qua pha truoc khi bi pha huy", "cay"),
    }

    MODE6_ACTION_TIME = 1.0
    MODE6_RESOLVE_TIME = 0.0
    MODE6_EXPECTIMAX_CHANCE_INTERVAL = 2.0
    ENEMY_DESTROY_TIME = MODE6_ACTION_TIME
    ENEMY_RETARGET_DELAY = 0.08
    MODE6_ENEMY_SPEED_SCALE = 0.85
    MODE6_ROBOT_SPEED_SCALE = 0.85
    MODE6_PLAN_COOLDOWN = 0.2
    MODE6_INTERRUPT_VALUE_MARGIN = 35
    BAT_FRAME_SIZE = 32
    BAT_ANIMATION_MS = 90
    LIGHTNING_EFFECT_MS = 900
    LIGHTNING_EFFECT_SCALE = 10
    IDS_STEP_DELAY = 0.06
    IDS_ITERATION_DELAY = 0.65
    IDSA_ITERATION_DELAY = 0.65

    # ------------------------------------------------------------------ init

    # ------------------------------------------------------------------
    # lifecycle / public controls
    # ------------------------------------------------------------------

    def __init__(self, player, soil_layer, collision_sprites, farm_tiles,
                 mode=2, hidden_blocks=None, enemy_spawn=None,
                 terrain_costs=None, extra_walkable_tiles=None,
                 entry_tile=None,
                 selected_algorithm=None):
        self.player = player
        self.mode6_player_base_speed = self.player.speed
        self.soil_layer = soil_layer
        self.collision_sprites = collision_sprites
        self.player.auto_control = True

        self.mode = mode
        info = self.MODE_INFO.get(mode, self.MODE_INFO[2])
        self.mode_name = info[0]
        self.algorithm_group_name = info[1]
        self.algorithm_name = selected_algorithm or self._default_algorithm(mode)
        if (self.mode == 4
                and self.algorithm_name in (
                    "Belief Init-Goal A*", "Belief BFS", "Belief A*")):
            self.algorithm_name = "Belief-State BFS"
        if (self.mode == 4
                and self.algorithm_name == "Belief-State BFS"
                and self.algorithm_name not in algorithms.ONLINE_ALGORITHMS):
            self.algorithm_name = self._default_algorithm(mode)
        self.mode_desc = info[2]
        self.difficulty_desc = info[3] if len(info) > 3 else ""
        self.farm_tiles = list(farm_tiles)
        self.terrain_costs = dict(terrain_costs or {})
        self.extra_walkable_tiles = set(extra_walkable_tiles or [])
        self.rows = len(self.soil_layer.grid)
        self.cols = len(self.soil_layer.grid[0]) if self.rows else 0
        self.spawn_tile = self._world_to_tile(self.player.rect.center)
        self.mode4_entry_tile = entry_tile
        self.walkable_tiles = self._build_walkable_tiles()

        self.state = "CHOOSE_TARGET"
        self.current_target = None
        self.path = []
        self.chosen_target_path = None
        self.full_garden_plan = []
        self.full_mode2_walk_path = []
        self.full_garden_leg_paths = {}
        self.full_garden_leg_stats = {}
        self.full_garden_index = 0
        self.full_garden_stats = {}
        self.mode1_ucs_tile_costs = {}
        self.mode1_ucs_path_length_total = 0
        self.mode1_ucs_current_leg_length = 0
        self.mode1_ucs_extra_cost_total = 0
        self.mode1_ucs_current_leg_extra_cost = 0
        self.mode2_total_g = 0
        self.mode2_current_leg_g = 0
        self.mode2_total_weighted_g = 0.0
        self.mode2_current_leg_weighted_g = 0.0
        self.mode2_prev_dir = None
        self.mode2_current_leg_start = None
        self.mode2_current_leg_target = None
        self.mode2_current_step_stats = []
        self.mode2_current_step_index = 0
        self.stop_after_current_target = False
        self.dfs_walk = None
        self.dfs_explored_order = []
        self.dfs_stack_max = 0
        self.ids_depth_limit = 0
        self.ids_search_timer = 0.0
        self.ids_search_start = None
        self.ids_search_goals = set()
        self.ids_search_blocked = set()
        self.ids_pending_depth_increment = False
        self.ids_stack = []
        self.ids_reached = set()
        self.ids_frontier = set()
        self.ids_came_from = {}
        self.ids_iteration_found = []
        self.ids_iteration_cutoff = False
        self.ids_iteration_expansions = 0
        self.ids_total_expansions = 0
        self.ids_unique_explored = set()
        self.ids_found_targets = []
        self.ids_found_paths = {}
        self.ids_found_depths = {}
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
        self.is_running = False
        if not self.is_running:
            self.message = "Chon thuat toan roi nhan START"
        self.algorithm_buttons = {}
        self.control_buttons = {}
        self.visual_assets = self._load_visual_assets()
        self.fog_time = 0.0

        # --- Thá»‘ng kĂª hiá»ƒn thá»‹ chung ---
        self.stats = {}
        self.algorithm_elapsed_seconds = {}

        # --- Mode 1 (BFS): lÆ°u explored nodes ---
        self.bfs_explored = set()

        # --- Mode 2 (A*): lÆ°u explored + f/g/h hiá»‡n táº¡i ---
        self.astar_explored = set()
        self.astar_current_fgh = (0, 0, 0)  # (f, g, h) má»›i nháº¥t

        # Sensor map: khong ep diem theo toa do/route.
        # Mode 3 se tinh score bang LOAI CAY + Manhattan trong _hill_score.
        # Dryness chi de hien thi/tham khao, khong dung de dan duong Mode 3.
        self.dryness = {
            tile: random.randint(30, 95)
            for tile in self.farm_tiles
        }

        self.hc_scores = {}  # tile -> score, Ä‘á»ƒ váº½ lĂªn map
        self.hc_current_score = 0
        self.hc_best_neighbor_score = 0
        self.anneal_temperature = 80.0
        self.tile_conditions = self._build_tile_conditions()
        self.mode6_crop_profiles = {
            tile: dict(algorithms.MODE6_TREE_STATUS.get(
                condition, algorithms.MODE6_TREE_STATUS["dry"]))
            for tile, condition in self.tile_conditions.items()
        } if self.mode == 6 else {}
        self.mode6_lightning_rod_tile = (
            (max(tile[0] for tile in self.farm_tiles) + 1,
             min(tile[1] for tile in self.farm_tiles))
            if self.mode == 6 and self.farm_tiles
            else None
        )
        self.danger_level = self._build_danger_levels()

        # --- Mode 4 (Online Search): fog-of-war ---
        self.hidden_blocked = set()
        self.discovered_blocked = set()
        self.explored_tiles = set()
        self.known_free = set()
        self.belief_worlds = []
        self.belief_unknown_tiles = set()
        self.belief_context = None
        self.belief_worlds_before_observation = 0
        self.belief_worlds_after_observation = 0
        self.belief_inconsistent = False
        self.mode4_current_frontier = None
        self.belief_max_hidden = 4
        self._last_risk_map = {}
        self.frontier_tiles = set()
        self.belief_state = {}
        self.mode4_phase = "IDLE"
        self.vision_radius = 1.5
        self.replan_count = 0
        self.mode4_completed_g = 0.0
        self.mode4_active_step = None
        self._mode4_online_astar_reset()
        self._mode4_online_bfs_reset()
        self._mode4_belief_bfs_reset()
        self.and_or_intended_tile = None
        self.and_or_actual_tile = None
        self.and_or_outcome_probability = 0
        self.and_or_outcome_label = ""
        self.and_or_seen_targets = frozenset()
        self.and_or_visited = set()      # tiles already visited this navigation
        self.and_or_resume_target = None
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
            self._init_mode4_belief()
            start_tile = self._world_to_tile(self.player.rect.center)
            self._expand_vision(start_tile)

        # --- Mode 5 (CSP): backtracking gĂ¡n crop ---
        self.seed_plan = {}
        self.csp_steps = []  # lưu các bước backtracking
        # CSP animation state
        self.csp_phase = "idle"          # idle | analyze | replay | ready
        self.csp_analyze_timer = 0.0
        self.csp_analyze_duration = 1.8  # thời gian phase "phân tích"
        self.csp_replay_index = 0
        self.csp_replay_timer = 0.0
        self.csp_replay_speed = 0.06     # giây/bước
        self.csp_display = {}            # {tile: (crop, state)} state=try|assign|backtrack
        self.csp_domains = {}            # {tile: [crop, ...]} domain hien tai khi replay CSP
        self.csp_flash_timers = {}       # {tile: thời gian flash đỏ còn lại}
        self.csp_analyze_pairs = []      # cặp ô highlight phase analyze
        self.csp_analyze_pair_index = 0
        self.csp_analyze_pair_timer = 0.0
        # --- CSP robot-walk: robot di den tung o trong luc replay ---
        self.csp_walk_target = None      # tile dang di den
        self.csp_walk_path = []          # duong di den tile do
        self.csp_walk_state = "idle"     # idle | moving | acting
        self.csp_walk_act_timer = 0.0    # thoi gian dung lai lam viec
        self.csp_walk_act_duration = 0.0
        self.csp_current_var = None      # bien dang xu ly
        self.csp_current_value = None    # gia tri dang thu
        self.csp_current_event = None    # try/assign/backtrack
        self.csp_backtrack_flash = 0.0   # thoi gian flash backtrack con lai
        self.csp_speech = ""             # speech bubble text
        self.csp_speech_timer = 0.0
        self.csp_speech_tile = None      # tile ma bubble dang gan vao (de ve tai tile, khong theo robot)
        self.csp_speech_pending = None   # (text, duration) hoan lai toi khi robot den tile
        self.csp_conflict_tiles = set()  # cac o dang xung dot (highlight do)
        self.csp_conflict_timer = 0.0
        self.csp_assigned = {}           # {tile: crop} da assign chinh thuc
        self.csp_seeded_tiles = set()    # cac o da thuc su gieo+tuoi xong (dung de to xanh)
        self.csp_backtracks_live = 0     # dem backtrack realtime

        # --- Mode 6 (Minimax): robot Ä‘á»‘i thá»§ ---
        self.enemy_tile = None
        self.enemy_spawn_tile = tuple(enemy_spawn) if enemy_spawn else None
        self.enemy_target = None
        self.enemy_retreat_target = None
        self.enemy_path = []
        self.enemy_done_tiles = set()
        self.enemy_destroy_until = 0
        self.enemy_retarget_timer = 0.0
        self.mode6_fix_until = 0
        self.mode6_fix_target = None
        self.minimax_value = 0
        self.alpha_beta_info = {
            "alpha": "-\u221e", "beta": "+\u221e", "pruned": 0}
        self.mode6_tree_details = {}
        self.mode6_chance_event = ""
        self.mode6_chance_probability = 0.0
        self.mode6_random_chance_elapsed = 0.0
        self.mode6_turn = 0
        self.mode6_move_total = 0
        self.mode6_move_done = 0
        self.mode6_enemy_move_total = 0
        self.mode6_enemy_move_done = 0
        self.mode6_phase = "PLAN_STEP"
        self.mode6_robot_action = "STAY"
        self.mode6_enemy_action = "STAY"
        self.mode6_robot_step_target = None
        self.mode6_enemy_step_target = None
        self.mode6_robot_target = None
        self.mode6_enemy_target = None
        self.mode6_robot_path = []
        self.mode6_enemy_path = []
        self.mode6_robot_prev_tile = None
        self.mode6_enemy_prev_tile = None
        self.mode6_robot_step_start = None
        self.mode6_enemy_step_start = None
        self.mode6_enemy_threat_tile = None
        self.mode6_enemy_threat_turns = 0
        self.mode6_enemy_forbidden_tile = None
        self.mode6_robot_repair_tile = None
        self.mode6_robot_repair_turns = 0
        self.mode6_resolve_elapsed = 0.0
        self.mode6_plan_cooldown = 0.0
        self.mode6_last_plan_ms = 0
        self.mode6_robot_hold_elapsed = 0.0
        self.mode6_enemy_hold_elapsed = 0.0
        self.mode6_robot_hold_required = False
        self.mode6_enemy_hold_required = False
        self.mode6_robot_committed_tile = None
        self.mode6_pending_actor = None
        self.mode6_pending_target = None
        self.mode6_pending_event = None
        self.mode6_loop_positions = []
        self.mode6_loop_count = 0
        self.mode6_shared_target = None
        self.mode6_shared_target_count = 0
        self.mode6_enemy_aim_target = None
        self.mode6_enemy_aim_count = 0
        self.mode6_lightning_effects = []
        if self.mode == 6:
            if enemy_spawn is not None:
                # DĂ¹ng spawn cá»‘ Ä‘á»‹nh tá»« Level (gĂ³c pháº£i hĂ ng dÆ°á»›i)
                self.enemy_tile = tuple(enemy_spawn)
            elif self.farm_tiles:
                # Fallback: tĂ­nh tá»« farm tiles
                xs = [t[0] for t in self.farm_tiles]
                ys = [t[1] for t in self.farm_tiles]
                self.enemy_tile = (max(xs) + 3, max(ys) + 2)
            self._reset_mode6_runtime(reset_positions=False)


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
        self._reset_current_algorithm_timer()
        self.message = f"Khu {self.mode}: {self.algorithm_name}"
        self.stats = {}
        self.path = []
        self.chosen_target_path = None
        self._clear_full_garden_plan()
        self.dfs_walk = None
        self.dfs_explored_order = []
        self.dfs_stack_max = 0
        self._reset_ids_search()
        self._reset_idsa_search()
        self.current_target = None
        self.state = "CHOOSE_TARGET"
        self.mode1_ucs_path_length_total = 0
        self.mode1_ucs_current_leg_length = 0
        self.mode1_ucs_extra_cost_total = 0
        self.mode1_ucs_current_leg_extra_cost = 0
        self.mode2_total_g = 0
        self.mode2_current_leg_g = 0
        self.mode2_total_weighted_g = 0.0
        self.mode2_current_leg_weighted_g = 0.0
        self.mode2_prev_dir = None
        self.mode2_current_leg_start = None
        self.mode2_current_leg_target = None
        self.mode2_current_step_stats = []
        self.mode2_current_step_index = 0
        self.stop_after_current_target = False
        self.bfs_explored.clear()
        self.astar_explored.clear()
        self.hc_scores.clear()
        self.anneal_temperature = 80.0
        self.and_or_intended_tile = None
        self.and_or_actual_tile = None
        self.and_or_outcome_probability = 0
        self.and_or_outcome_label = ""
        self.and_or_visited = set()
        self.belief_worlds = []
        self.belief_unknown_tiles = set()
        self.belief_context = None
        self.is_running = False
        self.message = f"Da chon {algorithm_name}. Nhan START"

        if self.mode == 5:
            self.done_tiles.clear()
            # Chi tao nghiem khi bat dau replay. Nhu vay man hinh idle
            # khong con cam san dap an da tinh truoc.
            self.seed_plan = {}
            self.csp_steps = []
            self.csp_phase = "idle"
            self.csp_replay_index = 0
            self.csp_replay_timer = 0.0
            self.csp_display = {}
            self.csp_domains = {}
            self.csp_flash_timers = {}
            self.csp_analyze_pairs = []
            self.csp_analyze_pair_index = 0
            self.csp_analyze_pair_timer = 0.0
            self.csp_walk_target = None
            self.csp_walk_path = []
            self.csp_walk_state = "idle"
            self.csp_walk_act_timer = 0.0
            self.csp_current_var = None
            self.csp_current_value = None
            self.csp_current_event = None
            self.csp_backtrack_flash = 0.0
            self.csp_speech = ""
            self.csp_speech_timer = 0.0
            self.csp_speech_tile = None
            self.csp_speech_pending = None
            self.csp_conflict_tiles = set()
            self.csp_conflict_timer = 0.0
            self.csp_assigned = {}
            self.csp_seeded_tiles = set()
            self.csp_backtracks_live = 0
            self.message = f"Da chon {algorithm_name}. Nhan START de xem ke hoach"
        elif self.mode == 4:
            self.done_tiles.clear()
            self.discovered_blocked.clear()
            self.explored_tiles.clear()
            self.known_free = set()
            self.belief_worlds = []
            self.belief_unknown_tiles = set()
            self.belief_context = None
            self.belief_worlds_before_observation = 0
            self.belief_worlds_after_observation = 0
            self.belief_inconsistent = False
            self.frontier_tiles = set()
            self.belief_state = {}
            self.mode4_phase = "EXPLORE"
            self.replan_count = 0
            self.mode4_completed_g = 0.0
            self.mode4_active_step = None
            self._mode4_online_astar_reset()
            self._mode4_online_bfs_reset()
            self._mode4_belief_bfs_reset()
            self.and_or_seen_targets = frozenset()
            self.and_or_visited = set()
            self.and_or_resume_target = None
            spawn_pos = self._tile_center(self.spawn_tile)
            self.player.pos.update(spawn_pos)
            self.player.rect.center = (round(spawn_pos.x), round(spawn_pos.y))
            self.player.hitbox.center = self.player.rect.center
            self._init_mode4_belief()
            self._expand_vision(self.spawn_tile)
        elif self.mode == 6:
            self._reset_mode6_runtime(reset_positions=True)

        return True


    def _reset_mode6_runtime(self, reset_positions=True):
        self.player.speed = self.mode6_player_base_speed
        self._reset_current_algorithm_timer()
        self.done_tiles.clear()
        self.enemy_done_tiles.clear()
        self.current_target = None
        self.enemy_target = None
        self.enemy_retreat_target = None
        self.path = []
        self.enemy_path = []
        self.chosen_target_path = None
        self.enemy_destroy_until = 0
        self.enemy_retarget_timer = 0.0
        self.mode6_fix_until = 0
        self.mode6_fix_target = None
        self.minimax_value = 0
        self.alpha_beta_info = {
            "alpha": "-\u221e", "beta": "+\u221e", "pruned": 0}
        self.mode6_tree_details = {}
        self.mode6_chance_event = ""
        self.mode6_chance_probability = 0.0
        self.mode6_random_chance_elapsed = 0.0
        self.mode6_turn = 0
        self.mode6_move_total = 0
        self.mode6_move_done = 0
        self.mode6_enemy_move_total = 0
        self.mode6_enemy_move_done = 0
        self.mode6_phase = "PLAN_STEP"
        self.mode6_robot_action = "STAY"
        self.mode6_enemy_action = "STAY"
        self.mode6_robot_step_target = None
        self.mode6_enemy_step_target = None
        self.mode6_robot_target = None
        self.mode6_enemy_target = None
        self.mode6_robot_path = []
        self.mode6_enemy_path = []
        self.mode6_robot_prev_tile = None
        self.mode6_enemy_prev_tile = None
        self.mode6_robot_step_start = None
        self.mode6_enemy_step_start = None
        self.mode6_enemy_threat_tile = None
        self.mode6_enemy_threat_turns = 0
        self.mode6_enemy_forbidden_tile = None
        self.mode6_robot_repair_tile = None
        self.mode6_robot_repair_turns = 0
        self.mode6_resolve_elapsed = 0.0
        self.mode6_plan_cooldown = 0.0
        self.mode6_last_plan_ms = 0
        self.mode6_robot_hold_elapsed = 0.0
        self.mode6_enemy_hold_elapsed = 0.0
        self.mode6_robot_hold_required = False
        self.mode6_enemy_hold_required = False
        self.mode6_robot_committed_tile = None
        self.mode6_pending_actor = None
        self.mode6_pending_target = None
        self.mode6_pending_event = None
        self.mode6_loop_positions = []
        self.mode6_loop_count = 0
        self.mode6_shared_target = None
        self.mode6_shared_target_count = 0
        self.mode6_enemy_aim_target = None
        self.mode6_enemy_aim_count = 0
        self.mode6_lightning_effects = []
        self.state = "PLAN_STEP"
        self.stats = {}
        if reset_positions:
            spawn_pos = self._tile_center(self.spawn_tile)
            self.player.pos.update(spawn_pos)
            self.player.rect.center = (round(spawn_pos.x), round(spawn_pos.y))
            self.player.hitbox.center = self.player.rect.center
            self.player.direction.update(0, 0)
        if self.enemy_spawn_tile is not None:
            self.enemy_tile = self.enemy_spawn_tile


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
        if self.mode == 5 and self.csp_phase == "idle":
            self.is_running = True
            self._begin_csp_analyze()
            return
        if self.mode == 6:
            self.state = "PLAN_STEP"
            self.mode6_phase = "PLAN_STEP"
        self.is_running = True
        self.message = f"Dang chay {self.algorithm_name}"


    def pause(self):
        self.is_running = False
        if self.mode == 6:
            self.player.speed = self.mode6_player_base_speed
        self.player.direction.update(0, 0)
        self.message = "Tam dung"


    def _algorithm_timer_key(self):
        return (self.mode, self.algorithm_name)


    def _reset_current_algorithm_timer(self):
        self.algorithm_elapsed_seconds[self._algorithm_timer_key()] = 0.0


    def _current_algorithm_elapsed(self):
        return self.algorithm_elapsed_seconds.get(
            self._algorithm_timer_key(), 0.0)


    def _format_elapsed_time(self, seconds):
        total = max(0, int(seconds))
        minutes, secs = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


    def _tick_algorithm_timer(self, dt):
        if not self.is_running or self.state == "DONE":
            return
        key = self._algorithm_timer_key()
        self.algorithm_elapsed_seconds[key] = (
            self.algorithm_elapsed_seconds.get(key, 0.0) + max(0.0, dt))
        if isinstance(self.stats, dict):
            self.stats["Run time"] = self._format_elapsed_time(
                self.algorithm_elapsed_seconds[key])


    def handle_panel_click(self, pos):
        for algorithm_name, rect in self.algorithm_buttons.items():
            if rect.collidepoint(pos):
                self.set_algorithm(algorithm_name)
                return "algorithm"

        for action, rect in self.control_buttons.items():
            if not rect.collidepoint(pos):
                continue
            if getattr(self, "manual_control", False) and action in ("start", "pause"):
                return None
            if action == "start":
                self.start()
                return "start"
            if action == "pause":
                self.pause()
                return "pause"
            if action == "reset":
                return "reset"
            if action == "manual":
                return "manual"
        return None
    # -------------------------------------------------------- helpers chung

    # ------------------------------------------------------------------
    # shared helpers / map state
    # ------------------------------------------------------------------

    def _clear_full_garden_plan(self):
        self.full_garden_plan = []
        self.full_mode2_walk_path = []
        self.full_garden_leg_paths = {}
        self.full_garden_leg_stats = {}
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
        if self.mode not in (1, 4, 6) or not self.farm_tiles:
            return None

        walkable = set(self.farm_tiles)
        walkable.add(self.spawn_tile)
        walkable.update(self.extra_walkable_tiles)
        walkable.update(self.terrain_costs.keys())

        if self.mode in (4, 6):
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


    def _is_adjacent_tile(self, a, b):
        return self._heuristic(a, b) == 1


    def _path_is_contiguous(self, start, path):
        current = start
        for tile in path:
            if not self._is_adjacent_tile(current, tile):
                return False
            current = tile
        return True


    @staticmethod
    def _heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


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
            "golden_plant": "golden_plant.png",
            "healthy_plant": "healthy_plant.png",
            "dead_plant": "dead_plant.png",
            "storm_debris": "storm_debris.png",
            "worm": "worm.png",
            "black_worm": "black_worm.png",
            "bug": "bug.png",
            "crow": "crow.png",
            "bat": "bat.gif",
            "lightning": "lightining2.gif",
            "lightning_rod": "lightning_rod.png",
            "crow_damage": "crow_damage.png",
            "crow_wilted_plant": "crow_wilted_plant.png",
        }
        assets = {}
        for key, filename in asset_names.items():
            path = os.path.join(asset_dir, filename)
            if os.path.exists(path):
                image = pygame.image.load(path).convert_alpha()
                assets[key] = pygame.transform.scale(image, (TILE_SIZE, TILE_SIZE))
        lightning = assets.get("lightning")
        if lightning is not None:
            effect_size = (
                lightning.get_width() * self.LIGHTNING_EFFECT_SCALE,
                lightning.get_height() * self.LIGHTNING_EFFECT_SCALE,
            )
            assets["lightning_effect"] = pygame.transform.scale(
                lightning, effect_size)
        bat_animations = self._load_bat_animations(asset_dir)
        assets.update(bat_animations)
        rock_frames = self._load_rock_frames(asset_dir)
        if rock_frames:
            assets["rocks"] = rock_frames
        return assets


    def _load_rock_frames(self, asset_dir):
        path = os.path.join(asset_dir, "Rocks.png")
        if not os.path.exists(path):
            return []

        sheet = pygame.image.load(path).convert_alpha()
        frame_size = 16
        frames = []
        for y in range(0, sheet.get_height(), frame_size):
            for x in range(0, sheet.get_width(), frame_size):
                if x + frame_size > sheet.get_width() or y + frame_size > sheet.get_height():
                    continue
                frame = sheet.subsurface(
                    pygame.Rect(x, y, frame_size, frame_size)).copy()
                image = pygame.transform.scale(
                    frame, (TILE_SIZE - 14, TILE_SIZE - 14))
                surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                surf.blit(image, image.get_rect(
                    midbottom=(TILE_SIZE // 2, TILE_SIZE - 4)))
                frames.append(surf)
        return frames


    def _load_bat_animations(self, asset_dir):
        path = os.path.join(asset_dir, "bat_animations.png")
        if not os.path.exists(path):
            return {}

        sheet = pygame.image.load(path).convert_alpha()
        frame_size = self.BAT_FRAME_SIZE
        animations = {"bat_fly_frames": [], "bat_attack_frames": []}
        row_keys = ["bat_fly_frames", "bat_attack_frames"]
        for y in range(0, sheet.get_height(), frame_size):
            row_index = y // frame_size
            if row_index >= len(row_keys):
                break
            frames = animations[row_keys[row_index]]
            for x in range(0, sheet.get_width(), frame_size):
                if x + frame_size > sheet.get_width() or y + frame_size > sheet.get_height():
                    continue
                frame = pygame.Surface((frame_size, frame_size), pygame.SRCALPHA)
                frame.blit(sheet, (0, 0), pygame.Rect(x, y, frame_size, frame_size))
                if frame.get_bounding_rect().width == 0:
                    continue
                frames.append(pygame.transform.scale(frame, (TILE_SIZE, TILE_SIZE)))
        return {key: frames for key, frames in animations.items() if frames}


    def _enemy_is_attacking(self):
        if self._enemy_is_destroying():
            return True
        if self.mode != 6 or self.algorithm_name == "Expectimax":
            return False

        enemy_tile = self._mode6_actor_tile(self.enemy_tile)
        if enemy_tile is None:
            return False
        if (self.state == "MOVE_STEP"
                and self.mode6_enemy_hold_required):
            return True
        if self.mode6_enemy_threat_tile == enemy_tile:
            return True
        if self.state == "RESOLVE_STEP":
            robot_tile = self._world_to_tile(self.player.rect.center)
            return (
                enemy_tile != robot_tile
                and enemy_tile in set(self._mode6_remaining_tiles()))
        return False


    def _enemy_image(self):
        frame_key = (
            "bat_attack_frames"
            if self._enemy_is_attacking()
            else "bat_fly_frames")
        bat_frames = self.visual_assets.get(frame_key)
        if bat_frames:
            frame_index = (
                pygame.time.get_ticks() // self.BAT_ANIMATION_MS
            ) % len(bat_frames)
            return bat_frames[frame_index]
        return self.visual_assets.get("bat") or self.visual_assets.get("crow")


    def _blit_tile_asset(self, surface, offset, tile, asset_key, y_offset=0):
        image = self.visual_assets.get(asset_key)
        if image is None:
            return
        if isinstance(image, list):
            if not image:
                return
            image = image[(tile[0] * 7 + tile[1] * 11) % len(image)]
        world = self._tile_center(tile)
        rect = image.get_rect(center=(int(world.x - offset.x),
                                      int(world.y - offset.y + y_offset)))
        surface.blit(image, rect)


    def _trigger_lightning_effect(self, tile):
        if tile is None:
            return
        expires_at = pygame.time.get_ticks() + self.LIGHTNING_EFFECT_MS
        self.mode6_lightning_effects.append((tile, expires_at))


    def _draw_mode6_lightning_rod(self, surface, offset):
        if (self.mode != 6
                or self.algorithm_name not in ("Expectimax", "Expectiminimax")
                or self.mode6_lightning_rod_tile is None):
            return
        self._blit_tile_asset(
            surface, offset, self.mode6_lightning_rod_tile, "lightning_rod")


    def _draw_lightning_effects(self, surface, offset):
        image = self.visual_assets.get("lightning_effect")
        if not self.mode6_lightning_effects:
            return

        now = pygame.time.get_ticks()
        active_effects = []
        for tile, expires_at in self.mode6_lightning_effects:
            if now >= expires_at:
                continue
            active_effects.append((tile, expires_at))
            if image is None:
                continue
            world = self._tile_center(tile)
            rect = image.get_rect(center=(
                int(world.x - offset.x),
                int(world.y - offset.y - 24)))
            surface.blit(image, rect)
        self.mode6_lightning_effects = active_effects


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

        # Mode 3: chi xep LOAI CAY/HINH ANH CAY.
        # Diem cua Mode 3 duoc tinh trong _hill_score theo:
        # dry=40, critical=30, pest=20, dead=10, sau do + Manhattan.
        if self.mode == 3:
            mode3_layout = {
                # Layout toi uu theo cong thuc:
                # score = diem_loai_cay + Manhattan(current, tile)
                # dry=40, critical=30, pest=20, dead=10.
                #
                # Voi Hill Climbing dang di xuong (< current_score),
                # robot se co xu huong di theo chuoi chinh:
                # (33,22) dry -> (33,23) critical ->
                # (32,23) pest -> (32,24) dead.
                #
                # Cac o ke ben canh duoc xep khong thap hon de tranh robot
                # re ngang qua o dead/pest qua som.

                # Hang tren: y = 22
                (29, 22): "dead",
                (30, 22): "dry",
                (31, 22): "critical",
                (32, 22): "dry",
                (33, 22): "dry",

                # Hang 2: y = 23
                (29, 23): "dry",
                (30, 23): "dry",
                (31, 23): "critical",
                (32, 23): "critical",
                (33, 23): "dry",

                # Hang 3: y = 24
                (29, 24): "critical",
                (30, 24): "dry",
                (31, 24): "pest",
                (32, 24): "pest",
                (33, 24): "dry",

                # Hang 4: y = 25
                (29, 25): "pest",
                (30, 25): "critical",
                (31, 25): "dry",
                (32, 25): "dead",
                (33, 25): "pest",

                # Hang duoi: y = 26
                (29, 26): "dry",
                (30, 26): "dead",
                (31, 26): "critical",
                (32, 26): "pest",
                (33, 26): "dry",
            }
            return {
                tile: mode3_layout.get(tile, "dry")
                for tile in self.farm_tiles
            }

        if self.mode == 4:
            condition_cycle = ("dry", "critical", "dry", "pest", "dry")
            return {
                tile: condition_cycle[index % len(condition_cycle)]
                for index, tile in enumerate(self.farm_tiles)
            }
        if self.mode == 5:
            return {tile: "replant" for tile in self.farm_tiles}
        if self.mode == 6:
            min_x = min(tile[0] for tile in self.farm_tiles)
            min_y = min(tile[1] for tile in self.farm_tiles)
            status_grid = (
                ("dry", "critical", "pest", "dead", "pest"),
                ("pest", "dead", "dry", "golden", "critical"),
                ("critical", "pest", "golden", "dry", "pest"),
                ("dead", "golden", "pest", "critical", "dry"),
                ("pest", "dry", "critical", "pest", "dead"),
            )
            return {
                tile: status_grid[tile[1] - min_y][tile[0] - min_x]
                for tile in self.farm_tiles
            }

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


    def _build_danger_levels(self):
        # Kept for older HUD/code paths. For Mode 2 test version, smaller
        # means higher priority: dry=0, critical=1, pest/crow=2, dead=3.
        priorities = {
            "dry": 0,
            "critical": 1,
            "pest": 2,
            "crow": 2,
            "dead": 3,
            "replant": 3,
        }
        levels = {}
        for tile in self.farm_tiles:
            levels[tile] = priorities.get(self._condition_for_tile(tile), 3)
        return levels


    def _condition_asset(self, condition):
        if condition == "dead":
            return "dead_plant"
        if condition == "critical":
            return "storm_damaged_plant"
        if condition == "dry":
            return "dry_plant"
        if condition == "pest":
            return "healthy_plant"
        return "dry_plant"


    def _mode6_condition_asset(self, condition):
        if condition == "dry":
            return "dry_plant"
        if condition == "critical":
            return "storm_damaged_plant"
        if condition == "pest":
            return "urgent_plant"
        if condition == "dead":
            return "dead_plant"
        if condition == "golden":
            return "golden_plant"
        return "dry_plant"


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


    def _state_after_arrival(self, tile):
        if self.mode == 6:
            return "FIX_CROW"
        if self.mode == 4:
            self.mode4_phase = "RESCUE"
            self._update_belief_state()
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
        self._finish_mode1_ucs_leg(tile)
        self._finish_mode2_leg_g(tile)
        self._refresh_mode3_scores_after_action(tile)
        self._advance_full_garden_plan(tile)
        if self.mode == 4:
            self.mode4_phase = "EXPLORE"
            self._update_belief_state()
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
        self._finish_mode1_ucs_leg(tile)
        self._finish_mode2_leg_g(tile)
        self._refresh_mode3_scores_after_action(tile)
        self._advance_full_garden_plan(tile)
        if self.mode == 4:
            self.mode4_phase = "EXPLORE"
            self._update_belief_state()
        self.current_target = None
        self.path = []
        self.state = "DONE" if self.stop_after_current_target else "CHOOSE_TARGET"
        self.stop_after_current_target = False
        self.wait_time = 0.45


    def _finish_cultivation_target(self, tile, message):
        self.message = message
        self.done_tiles.add(tile)
        if self.mode == 5:
            self.csp_seeded_tiles.add(tile)
        self._finish_mode1_ucs_leg(tile)
        self._finish_mode2_leg_g(tile)
        self._refresh_mode3_scores_after_action(tile)
        self._advance_full_garden_plan(tile)
        if self.mode == 4:
            self.mode4_phase = "EXPLORE"
            self._update_belief_state()
        self.current_target = None
        self.path = []
        self.state = "DONE" if self.stop_after_current_target else "CHOOSE_TARGET"
        self.stop_after_current_target = False
        self.wait_time = 0.35


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

    # ------------------------------------------------------------------
    # mode 1 and mode 2 path search
    # ------------------------------------------------------------------

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
        prev_dir = self.mode2_prev_dir if name == "A*_v2" else None
        path, explored, fgh, stats = algorithms.find_path_by_algorithm(
            name, start, goal, blocked, self._neighbors, self._search_heuristic,
            self.counter, self._step_cost, prev_dir_override=prev_dir)
        self.astar_current_fgh = fgh
        stats["Algorithm"] = name
        if self.mode == 2 and goal in self.farm_tiles:
            g_leg = len(path)
            self.mode2_current_leg_g = g_leg
            self.mode2_current_leg_target = goal
            if name == "A*_v2":
                weighted_f, real_g, h_val = fgh
                display_g = self.mode2_total_g + real_g
                weighted_g_leg = weighted_f - h_val
                self.mode2_current_leg_weighted_g = weighted_g_leg
                self._apply_mode2_hud_stats(
                    stats, start, goal, fgh, display_g,
                    weighted_display=True)
                stats["g leg"] = real_g
                stats["g weighted leg"] = f"{weighted_g_leg:.1f}"
                stats["g completed"] = self.mode2_total_g
                stats["g weighted completed"] = (
                    f"{self.mode2_total_weighted_g:.1f}")
            elif name == "Greedy":
                stats.pop("f(n)", None)
                stats.pop("g(n)", None)
                stats["Priority"] = "min h(n)"
                self._apply_mode2_hud_stats(stats, start, goal)
                stats["g leg"] = g_leg
                stats["g completed"] = self.mode2_total_g
            else:
                # For A*/IDSA, display cumulative travel to this target.
                display_g = self.mode2_total_g + g_leg
                self._apply_mode2_hud_stats(
                    stats, start, goal, self.astar_current_fgh, display_g)
                stats["g leg"] = g_leg
                stats["g completed"] = self.mode2_total_g
        if self.mode == 1:
            self.bfs_explored = set(explored)
        elif self.mode == 2:
            self.astar_explored = explored
        elif name in ("A*", "A*_v2"):
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
        self.ids_pending_depth_increment = False
        self.ids_stack = []
        self.ids_reached = set()
        self.ids_frontier = set()
        self.ids_came_from = {}
        self.ids_iteration_found = []
        self.ids_iteration_cutoff = False
        self.ids_iteration_expansions = 0
        self.ids_total_expansions = 0
        self.ids_unique_explored = set()
        self.ids_found_targets = []
        self.ids_found_paths = {}
        self.ids_found_depths = {}


    def _reset_ids_iteration_stack(self):
        self.ids_stack = [(self.ids_search_start, 0)]
        self.ids_reached = {self.ids_search_start}
        self.ids_frontier = {self.ids_search_start}
        self.ids_came_from = {}
        self.ids_iteration_found = []
        self.ids_iteration_cutoff = False
        self.ids_iteration_expansions = 0
        self.bfs_explored = set()


    def _begin_ids_search(self, start, goals):
        self.ids_depth_limit = 0
        self.ids_search_timer = 0.0
        self.ids_search_start = start
        self.ids_search_goals = set(goals)
        self.ids_search_blocked = self._blocked_tiles()
        self.ids_pending_depth_increment = False
        self.ids_search_blocked.discard(start)
        for goal in self.ids_search_goals:
            self.ids_search_blocked.discard(goal)
        self.ids_total_expansions = 0
        self.ids_unique_explored = set()
        self.ids_found_targets = []
        self.ids_found_paths = {}
        self.ids_found_depths = {}
        self._reset_ids_iteration_stack()
        self.current_target = None
        self.path = []
        self.state = "IDS_SEARCH"
        self.player.direction.update(0, 0)
        self.message = f"IDS bat dau depth 0 tu {start}"
        self.stats = {
            "Algorithm": "IDS",
            "Depth limit": self.ids_depth_limit,
            "Restart from": f"{self.ids_search_start}",
            "This iteration": 0,
            "Total expansions": self.ids_total_expansions,
            "Unique explored": len(self.ids_unique_explored),
            "Target": "None",
        }


    def _next_ids_found_target(self, remaining=None):
        remaining_set = set(remaining) if remaining is not None else None
        for target in self.ids_found_targets:
            if target in self.done_tiles:
                continue
            if target in self.discovered_blocked:
                continue
            if target in self.enemy_done_tiles:
                continue
            if remaining_set is not None and target not in remaining_set:
                continue
            return target
        return None


    def _move_to_ids_target(self, target):
        start = self._world_to_tile(self.player.rect.center)
        if start == target:
            path = []
        elif start == self.ids_search_start and target in self.ids_found_paths:
            path = list(self.ids_found_paths[target])
        else:
            blocked = self._blocked_tiles()
            blocked.discard(start)
            blocked.discard(target)
            path, _, _ = algorithms.bfs(start, target, blocked, self._neighbors)
        self.current_target = target
        self.path = path
        self.state = "MOVE"
        self.stats.update({
            "Target": f"{target}",
            "Path length": len(path),
            "Found depth": self.ids_found_depths.get(target, "-"),
        })
        self.message = (
            f"IDS den {target} "
            f"(tim thay o depth {self.ids_found_depths.get(target, '-')})")


    def _update_ids_search(self, dt):
        self.player.direction.update(0, 0)
        if self.ids_search_timer > 0:
            self.ids_search_timer = max(0, self.ids_search_timer - dt)
            if self.ids_search_timer == 0 and self.ids_pending_depth_increment:
                self.ids_depth_limit += 1
                self.ids_pending_depth_increment = False
                self._reset_ids_iteration_stack()
                self.stats.update({
                    "Depth limit": self.ids_depth_limit,
                    "This iteration": 0,
                    "Target": "None",
                    "Found this depth": 0,
                })
                self.message = (
                    f"IDS bat dau depth {self.ids_depth_limit} "
                    f"tu {self.ids_search_start}")
            return
        if self.ids_pending_depth_increment:
            self.ids_depth_limit += 1
            self.ids_pending_depth_increment = False
            self._reset_ids_iteration_stack()
            self.stats.update({
                "Depth limit": self.ids_depth_limit,
                "This iteration": 0,
                "Target": "None",
                "Found this depth": 0,
            })
            self.message = (
                f"IDS bat dau depth {self.ids_depth_limit} "
                f"tu {self.ids_search_start}")
            return

        available_goals = (
            set(self.ids_search_goals)
            - set(self.done_tiles)
            - set(self.discovered_blocked)
            - set(self.enemy_done_tiles))
        if not available_goals and not self.ids_stack:
            target = self._next_ids_found_target()
            if target is not None:
                self._move_to_ids_target(target)
                return
            self.state = "DONE"
            self.message = "IDS: da xu ly het muc tieu"
            return

        if self.ids_stack:
            current, depth = self.ids_stack.pop()
            self.ids_frontier.discard(current)
            self.bfs_explored.add(current)
            self.ids_unique_explored.add(current)
            self.ids_total_expansions += 1
            self.ids_iteration_expansions += 1

            if (current in available_goals
                    and current not in self.ids_found_targets
                    and current not in self.ids_iteration_found):
                self.ids_iteration_found.append(current)
                self.ids_found_targets.append(current)
                self.ids_found_depths[current] = self.ids_depth_limit
                self.ids_found_paths[current] = algorithms.reconstruct_path(
                    self.ids_came_from, current)

            neighbor_list = list(self._neighbors(
                current, self.ids_search_blocked))
            if depth >= self.ids_depth_limit:
                for next_tile in neighbor_list:
                    if (next_tile not in self.ids_reached
                            and next_tile not in self.ids_frontier):
                        self.ids_iteration_cutoff = True
                        break
            else:
                for next_tile in neighbor_list:
                    if (next_tile not in self.ids_reached
                            and next_tile not in self.ids_frontier):
                        self.ids_reached.add(next_tile)
                        self.ids_came_from[next_tile] = current
                        self.ids_stack.append((next_tile, depth + 1))
                        self.ids_frontier.add(next_tile)

            self.stats = {
                "Algorithm": "IDS",
                "Depth limit": self.ids_depth_limit,
                "Restart from": f"{self.ids_search_start}",
                "Current node": f"{current}",
                "Node depth": depth,
                "Stack size": len(self.ids_stack),
                "This iteration": self.ids_iteration_expansions,
                "Total expansions": self.ids_total_expansions,
                "Unique explored": len(self.ids_unique_explored),
                "Target": (
                    f"{self.ids_iteration_found[0]}"
                    if self.ids_iteration_found else "None"),
                "Found this depth": len(self.ids_iteration_found),
            }
            self.message = (
                f"IDS depth {self.ids_depth_limit}: "
                f"pop {current} o depth {depth}, "
                f"stack con {len(self.ids_stack)}")
            self.ids_search_timer = self.IDS_STEP_DELAY
            return

        self.stats = {
            "Algorithm": "IDS",
            "Depth limit": self.ids_depth_limit,
            "Restart from": f"{self.ids_search_start}",
            "This iteration": self.ids_iteration_expansions,
            "Total expansions": self.ids_total_expansions,
            "Unique explored": len(self.ids_unique_explored),
            "Target": (
                f"{self.ids_iteration_found[0]}"
                if self.ids_iteration_found else "None"),
            "Found this depth": len(self.ids_iteration_found),
            "Stack size": 0,
        }

        target = self._next_ids_found_target()
        if target is not None:
            self.ids_pending_depth_increment = self.ids_iteration_cutoff
            self.ids_search_timer = 0.0
            self._move_to_ids_target(target)
            return

        if not self.ids_iteration_cutoff:
            self.state = "DONE"
            self.message = "IDS: khong con node de duyet"
            return

        finished_limit = self.ids_depth_limit
        self.ids_pending_depth_increment = True
        self.ids_search_timer = self.IDS_ITERATION_DELAY
        self.message = (
            f"IDS limit {finished_limit} da duyet xong; "
            f"chuan bi depth {finished_limit + 1}")


    def _reset_idsa_search(self):
        self.idsa_f_limit = 0
        self.idsa_search_timer = 0.0
        self.idsa_search_start = None
        self.idsa_search_goal = None
        self.idsa_search_blocked = set()
        self.idsa_total_expansions = 0
        self.idsa_unique_explored = set()


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
        current_node = self.idsa_search_start
        priority, distance_h, condition_cost, h_val = self._mode2_h_parts(
            current_node, self.idsa_search_goal)
        g_leg = len(path) if path is not None else 0
        g_val = self.mode2_total_g + g_leg
        f_val = g_val + h_val
        self.stats = {
            "Algorithm": "IDSA",
            "f(n)": f"{f_val}",
            "g(n)": f"{g_val}",
            "h(n)": f"{h_val}",
            "h distance": f"{distance_h}",
            "h condition": (
                f"2 * {priority} = {condition_cost} "
                f"({self._condition_for_tile(self.idsa_search_goal)})"),
            "h formula": "Manhattan + 2 * condition",
            "f limit": self.idsa_f_limit,
            "g leg": g_leg,
            "g completed": self.mode2_total_g,
            "g formula": "+1 per move",
            "Restart from": f"{self.idsa_search_start}",
            "This iteration": len(explored),
            "Total expansions": self.idsa_total_expansions,
            "Unique explored": len(self.idsa_unique_explored),
            "Target": f"{self.idsa_search_goal}",
        }
        if hit_depth_guard:
            self.stats["Stopped by guard"] = "yes"

        if path is not None:
            self.mode2_current_leg_g = len(path)
            self.mode2_current_leg_target = self.idsa_search_goal
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
        self.idsa_f_limit = next_limit
        self.idsa_search_timer = self.IDSA_ITERATION_DELAY
        self.message = (
            f"IDSA f-limit {finished_limit} chua thay; "
            f"quay lai {self.idsa_search_start}, tang len "
            f"{self.idsa_f_limit}")


    def _mode2_condition_h(self, tile):
        """Heuristic bonus for Mode 2 target condition.

        Smaller value = higher priority because Greedy/A* use min-heap:
            dry      = 0
            critical = 1
            pest     = 2
            dead     = 3

        This is intentionally separate from Mode 3 score.
        """
        priorities = {
            "dry": 0,
            "critical": 1,
            "pest": 2,
            "crow": 2,
            "dead": 3,
            "replant": 3,
        }
        return priorities.get(self._condition_for_tile(tile), 3)


    def _search_heuristic(self, a, b):
        """Heuristic used by Greedy, A* and IDA*.

        Mode 2 uses the formula requested for informed search:

            h(n) = Manhattan(current_node, target) + 2 * condition(target)

        Important: ``a`` is the current expanded node, not the original
        spawn point. Therefore A* and IDA* recalculate h(n) for every node
        they expand. ``b`` is the current target/goal being evaluated.
        """
        if self.mode in (2, 4) and b in self.farm_tiles:
            return self._heuristic(a, b) + 2 * self._mode2_condition_h(b)
        return self._heuristic(a, b)


    def _mode2_h_parts(self, current_node, target):
        priority = self._mode2_condition_h(target)
        distance = self._heuristic(current_node, target)
        condition_cost = 2 * priority
        return priority, distance, condition_cost, distance + condition_cost


    @staticmethod
    def _step_direction(from_tile, to_tile):
        dx = to_tile[0] - from_tile[0]
        dy = to_tile[1] - from_tile[1]
        if dx > 0:
            return "R"
        if dx < 0:
            return "L"
        if dy > 0:
            return "D"
        if dy < 0:
            return "U"
        return None


    def _count_turns_in_path(self, path, start=None, initial_dir=None):
        turns = 0
        previous_dir = initial_dir
        for index in range(len(path)):
            from_tile = start if index == 0 else path[index - 1]
            if from_tile is None:
                continue
            direction = self._step_direction(from_tile, path[index])
            if (previous_dir is not None and direction is not None
                    and direction != previous_dir):
                turns += 1
            previous_dir = direction
        return turns


    def _apply_mode2_hud_stats(
            self, stats, current_node, target, fgh=None, display_g=None,
            weighted_display=False):
        priority, distance, condition_cost, h = self._mode2_h_parts(
            current_node, target)
        condition = self._condition_for_tile(target)

        if fgh is not None and self.algorithm_name == "A*_v2":
            weighted_f, real_g, h_val = fgh
            if weighted_display and display_g is not None:
                weighted_g = (
                    self.mode2_total_weighted_g + weighted_f - h_val)
                stats["f(n)"] = f"{weighted_g + h_val:.1f}"
                stats["g(n)"] = (
                    f"{display_g} + turns*{algorithms.TURN_PENALTY:.1f}")
            else:
                stats["f(n)"] = f"{weighted_f:.1f}"
                stats["g(n)"] = (
                    f"{real_g} + turns*{algorithms.TURN_PENALTY:.1f}")
            stats["g steps"] = real_g
            stats["h(n)"] = f"{h_val:.1f}"
        elif fgh is not None and self.algorithm_name != "Greedy":
            _, g_val, _ = fgh
            if display_g is not None:
                g_val = display_g
            h_val = h
            f_val = g_val + h_val
            stats["f(n)"] = f"{f_val}"
            stats["g(n)"] = f"{g_val}"
            stats["h(n)"] = f"{h_val}"
        else:
            stats["h(n)"] = f"{h}"

        stats["h distance"] = f"{distance}"
        stats["h condition"] = (
            f"2 * {priority} = {condition_cost} ({condition})")
        stats["h formula"] = "Manhattan + 2 * condition"
        stats["g formula"] = (
            f"steps + turns*{algorithms.TURN_PENALTY:.1f}"
            if self.algorithm_name == "A*_v2" else "+1 per move")
        return stats


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


    def _mode1_ucs_condition_cost(self, tile):
        if tile in self.done_tiles:
            return 0
        condition_costs = {
            "dry": 0,
            "critical": 1,
            "pest": 2,
            "crow": 2,
            "dead": 3,
            "replant": 3,
        }
        return condition_costs.get(self._condition_for_tile(tile), 0)


    def _prepare_mode1_ucs_costs(self):
        tiles = set(self.walkable_tiles)
        tiles.update(self.farm_tiles)
        tiles.update(self.terrain_costs)
        self.mode1_ucs_tile_costs = {}
        for tile in tiles:
            terrain_cost = 5 if tile in self.terrain_costs else 0
            condition_cost = 0
            if tile in self.farm_tiles:
                condition_cost = self._mode1_ucs_condition_cost(tile)
            self.mode1_ucs_tile_costs[tile] = {
                "step": 1,
                "terrain": terrain_cost,
                "condition": condition_cost,
                "total": 1 + terrain_cost + condition_cost,
            }


    def _mode1_ucs_step_cost(self, current, next_tile):
        costs = self.mode1_ucs_tile_costs.get(next_tile)
        if costs is None:
            terrain_cost = 5 if next_tile in self.terrain_costs else 0
            condition_cost = 0
            if next_tile in self.farm_tiles:
                condition_cost = self._mode1_ucs_condition_cost(next_tile)
            return 1 + terrain_cost + condition_cost
        return costs["total"]


    def _mode1_ucs_path_extra_cost(self, path):
        extra_cost = 0
        for tile in path:
            costs = self.mode1_ucs_tile_costs.get(tile)
            if costs is None:
                terrain_cost = 5 if tile in self.terrain_costs else 0
                condition_cost = 0
                if tile in self.farm_tiles:
                    condition_cost = self._mode1_ucs_condition_cost(tile)
                extra_cost += terrain_cost + condition_cost
            else:
                extra_cost += costs["terrain"] + costs["condition"]
        return extra_cost


    def _step_cost(self, current, next_tile):
        # In Mode 2, A*/IDA* g(n) must be the real number of moves:
        # each move adds exactly +1. Terrain/condition belongs to h(n),
        # not to g(n), so f(n) remains easy to explain: f = steps + h.
        if self.mode == 2:
            return 1

        if self.mode == 1 and self.algorithm_name == "UCS":
            return self._mode1_ucs_step_cost(current, next_tile)

        terrain_cost = self.terrain_costs.get(next_tile, 0)
        if self.mode == 1 and next_tile in self.farm_tiles:
            return 1 + terrain_cost + self._ucs_task_cost(next_tile)
        return 1 + terrain_cost


    def _finish_mode1_ucs_leg(self, tile):
        if self.mode != 1 or self.algorithm_name != "UCS":
            return
        if tile != self.current_target:
            return
        self.mode1_ucs_path_length_total += self.mode1_ucs_current_leg_length
        self.mode1_ucs_extra_cost_total += (
            self.mode1_ucs_current_leg_extra_cost)
        self.mode1_ucs_current_leg_length = 0
        self.mode1_ucs_current_leg_extra_cost = 0


    def _finish_mode2_leg_g(self, tile):
        if self.mode != 2 or tile != self.mode2_current_leg_target:
            return
        self.mode2_total_g += self.mode2_current_leg_g
        self.mode2_current_leg_g = 0
        if self.algorithm_name == "A*_v2":
            self.mode2_total_weighted_g += (
                self.mode2_current_leg_weighted_g)
            self.mode2_current_leg_weighted_g = 0.0
        self.mode2_current_leg_target = None
        self.mode2_current_leg_start = None
        self.mode2_current_step_stats = []
        self.mode2_current_step_index = 0


    def _mode2_priority_path_to_any_goal(self, start, remaining):
        """Mode 2: BFS-style search with a priority queue.

        This is intentionally simple and stable:
        - expand only 4-direction neighbors from _neighbors();
        - store came_from parent for every node;
        - reconstruct a tile-by-tile path;
        - never jump directly to a target.

        Priority:
        - Greedy: priority = h(n)
        - A*/IDSA: priority = g(n) + h(n)

        h(n) is computed from the node currently being expanded to the best
        remaining target:
            h(n) = Manhattan(n, target) + 2 * condition(target)
        """
        goals = {
            tile for tile in remaining
            if tile not in self.done_tiles
            and tile not in self.discovered_blocked
            and tile not in self.enemy_done_tiles
        }
        if not goals:
            return None, [], set(), (0, 0, 0), {
                "Algorithm": self.algorithm_name,
                "Path length": 0,
            }

        if start in goals:
            target = start
            priority, distance, condition_cost, h_val = self._mode2_h_parts(
                start, target)
            stats = {
                "Algorithm": self.algorithm_name,
                "Target": f"{target}",
                "Path length": 0,
                "Nodes explored": 1,
                "h(n)": f"{h_val}",
                "h distance": f"{distance}",
                "h condition": (
                    f"2 * {priority} = {condition_cost} "
                    f"({self._condition_for_tile(target)})"),
                "h formula": "Manhattan + 2 * condition",
                "Search style": "BFS + priority queue",
            }
            return target, [], {start}, (h_val, 0, h_val), stats

        blocked = self._blocked_tiles()
        blocked.discard(start)
        for goal in goals:
            blocked.discard(goal)

        def best_h_from(node):
            # Smaller tuple wins. This is recalculated from the current node,
            # not from the original spawn.
            best = None
            for target in goals:
                priority, distance, condition_cost, h_val = self._mode2_h_parts(
                    node, target)
                key = (h_val, priority, distance, target, condition_cost)
                if best is None or key < best:
                    best = key
            return best

        start_h, _, _, _, _ = best_h_from(start)
        open_set = []
        heapq.heappush(
            open_set,
            (start_h, start_h, 0, next(self.counter), start)
        )
        came_from = {}
        g_score = {start: 0}
        closed = set()
        explored_order = []
        max_frontier = 1

        while open_set:
            _, _, _, _, current = heapq.heappop(open_set)
            current_g = g_score[current]
            if current in closed:
                continue
            closed.add(current)
            explored_order.append(current)

            if current in goals:
                target = current
                path = self._reconstruct(came_from, current)
                priority, distance, condition_cost, h_start = self._mode2_h_parts(
                    start, target)
                g_val = len(path)
                f_val = g_val + h_start
                fgh = (f_val, g_val, h_start)
                stats = {
                    "Algorithm": self.algorithm_name,
                    "Target": f"{target}",
                    "Path length": len(path),
                    "Nodes explored": len(explored_order),
                    "Frontier max": max_frontier,
                    "Search style": "BFS + priority queue",
                    "Parent rule": "came_from reconstructs each step",
                    "f(n)": f"{f_val}",
                    "g(n)": f"{g_val}",
                    "h(n)": f"{h_start}",
                    "h distance": f"{distance}",
                    "h condition": (
                        f"2 * {priority} = {condition_cost} "
                        f"({self._condition_for_tile(target)})"),
                    "h formula": "Manhattan + 2 * condition",
                    "g formula": "+1 per move",
                }
                if self.algorithm_name == "Greedy":
                    stats.pop("f(n)", None)
                    stats.pop("g(n)", None)
                    stats["Priority"] = "min h(n)"
                elif self.algorithm_name == "IDSA":
                    stats["Note"] = "IDSA visualized with priority frontier"
                return target, path, set(explored_order), fgh, stats

            for next_tile in self._neighbors(current, blocked):
                tentative_g = current_g + 1
                if tentative_g >= g_score.get(next_tile, float("inf")):
                    continue
                came_from[next_tile] = current
                g_score[next_tile] = tentative_g
                h_val, condition_priority, distance, _, _ = best_h_from(next_tile)
                if self.algorithm_name == "Greedy":
                    frontier_priority = h_val
                else:
                    frontier_priority = tentative_g + h_val
                heapq.heappush(
                    open_set,
                    (
                        frontier_priority,
                        h_val,
                        -tentative_g,
                        next(self.counter),
                        next_tile,
                    )
                )
                max_frontier = max(max_frontier, len(open_set))

        return None, [], set(explored_order), (0, 0, 0), {
            "Algorithm": self.algorithm_name,
            "Path length": 0,
            "Nodes explored": len(explored_order),
            "Search style": "BFS + priority queue",
            "Result": "No reachable target",
        }


    def _mode2_plan_one_leg(self, parent, target, prev_dir_override=None):
        """Plan one Mode-2 leg from parent to target.

        The search algorithm still owns the path.  The only change is that
        h(n) is evaluated with the expanded node n and this leg's target:

            Greedy priority = h(n)
            A*/IDA* priority = g(n) + h(n)
            g(n) = +1 for every move
            h(n) = Manhattan(n,target) + 2 * condition(target)
        """
        blocked = self._blocked_tiles()
        blocked.discard(parent)
        blocked.discard(target)
        return algorithms.find_path_by_algorithm(
            self.algorithm_name, parent, target, blocked,
            self._neighbors, self._search_heuristic,
            self.counter, self._step_cost,
            prev_dir_override=prev_dir_override)


    def _mode2_set_leg_runtime_stats(
            self, parent, target, path, explored, fgh, sort_key,
            step_stats=None):
        g_leg = len(path)
        self.mode2_current_leg_g = g_leg
        self.mode2_current_leg_start = parent
        self.mode2_current_leg_target = target
        self.mode2_current_step_index = 0
        if self.algorithm_name == "A*_v2":
            weighted_f, _, h_val = fgh
            self.mode2_current_leg_weighted_g = weighted_f - h_val
            if path:
                previous_tile = parent if len(path) == 1 else path[-2]
                self.mode2_prev_dir = self._step_direction(
                    previous_tile, path[-1])
        else:
            self.mode2_current_leg_weighted_g = 0.0
        if step_stats is None:
            self.mode2_current_step_stats = self._mode2_build_step_stats(
                parent, target, path, len(explored), sort_key)
        else:
            self.mode2_current_step_stats = [dict(stats) for stats in step_stats]
        self.astar_current_fgh = fgh
        self.astar_explored = set(explored)
        self._mode2_apply_precomputed_step_stats(0)


    def _mode2_build_step_stats(
            self, parent, target, path, explored_count, sort_key,
            completed_g=None, completed_weighted_g=None):
        if completed_g is None:
            completed_g = self.mode2_total_g
        if completed_weighted_g is None:
            completed_weighted_g = self.mode2_total_weighted_g
        nodes = [parent] + list(path)
        result = []
        priority, distance, condition_cost, h = self._mode2_h_parts(
            parent, target)
        g_val = completed_g + len(path)
        f_val = g_val + h
        turns = self._count_turns_in_path(
            path, parent, self.mode2_prev_dir)
        weighted_g = (
            completed_weighted_g + len(path)
            + turns * algorithms.TURN_PENALTY)
        for step_index, node in enumerate(nodes):
            next_node = nodes[step_index + 1] if step_index + 1 < len(nodes) else target
            stats = {
                "Algorithm": self.algorithm_name,
                "Target": f"{target}",
                "f(n)": (
                    f"{weighted_g + h:.1f}"
                    if self.algorithm_name == "A*_v2" else f"{f_val}"),
                "g(n)": (
                    f"{g_val} + {turns}*{algorithms.TURN_PENALTY:.1f}"
                    if self.algorithm_name == "A*_v2" else f"{g_val}"),
                "h(n)": f"{h}",
                "h distance": f"{distance}",
                "h condition": (
                    f"2 * {priority} = {condition_cost} "
                    f"({self._condition_for_tile(target)})"),
                "h formula": "Manhattan + 2 * condition",
                "Current node": f"{node}",
                "Next node": f"{next_node}",
                "Path length": len(path),
                "Path remaining": max(0, len(path) - step_index),
                "Nodes explored": explored_count,
                "g completed": completed_g,
                "g leg": len(path),
                "Target rule": "min score from current parent node",
                "Selected target": f"{target}",
                "Parent node": f"{parent}",
                "score(target)": sort_key[0],
                "Tie-break": "score, condition, Manhattan, tile",
            }
            if self.algorithm_name == "A*_v2":
                stats["g steps"] = g_val
                stats["g weighted"] = f"{weighted_g:.1f}"
                stats["Turns"] = turns
                stats["Turn penalty"] = algorithms.TURN_PENALTY
            if self.algorithm_name == "Greedy":
                stats.pop("f(n)", None)
                stats.pop("g(n)", None)
                stats["Priority"] = "min h(n)"
            result.append(stats)
        return result


    def _mode2_apply_precomputed_step_stats(self, step_index):
        if not self.mode2_current_step_stats:
            return
        step_index = max(
            0, min(step_index, len(self.mode2_current_step_stats) - 1))
        self.mode2_current_step_index = step_index
        stats = dict(self.mode2_current_step_stats[step_index])
        f_val = stats.get("f(n)")
        g_val = stats.get("g(n)")
        h_val = stats.get("h(n)")
        if f_val is not None and g_val is not None and h_val is not None:
            try:
                self.astar_current_fgh = (
                    int(f_val), int(g_val), int(h_val))
            except (ValueError, TypeError):
                try:
                    self.astar_current_fgh = (
                        float(f_val), float(g_val), float(h_val))
                except (ValueError, TypeError):
                    self.astar_current_fgh = (0, 0, 0)
        self.stats = stats


    def _build_mode2_frontier_full_plan(self, remaining, start):
        """Build the whole Mode-2 route with a priority BFS frontier.

        The robot executes the saved route later.  This planner is the only
        place where Mode 2 decides target order and path legs.
        """
        self._clear_full_garden_plan()

        remaining_set = {
            tile for tile in remaining
            if tile in self.farm_tiles
            and tile not in self.done_tiles
            and tile not in self.discovered_blocked
            and tile not in self.enemy_done_tiles
        }
        if not remaining_set:
            return []

        blocked = self._blocked_tiles()
        blocked.discard(start)
        for tile in remaining_set:
            blocked.discard(tile)

        plan = []
        planned = set()
        full_walk_path = []
        leg_paths = {}
        leg_stats = {}
        explored_all = set()
        frontier = []
        queued = set()
        current = start
        total_g = self.mode2_total_g
        total_weighted_g = self.mode2_total_weighted_g
        planning_prev_dir = self.mode2_prev_dir
        last_fgh = (0, 0, 0)
        max_frontier = 0

        def nearest_remaining_distance(candidate):
            open_targets = remaining_set - planned
            if not open_targets:
                return 0
            return min(self._heuristic(candidate, target)
                       for target in open_targets)

        def push_frontier_neighbors(center):
            nonlocal max_frontier
            x, y = center
            for candidate in (
                    (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if candidate not in remaining_set:
                    continue
                if candidate in planned or candidate in queued:
                    continue
                if candidate in blocked:
                    continue
                if candidate not in self.farm_tiles:
                    continue

                condition_priority = self._mode2_condition_h(candidate)
                g_val = total_g + len(full_walk_path) + 1
                h_val = (
                    nearest_remaining_distance(candidate)
                    + 2 * condition_priority)
                if self.algorithm_name == "Greedy":
                    heapq.heappush(
                        frontier,
                        (
                            h_val,
                            condition_priority,
                            candidate,
                            next(self.counter),
                            center,
                        )
                    )
                else:
                    priority = g_val + h_val
                    heapq.heappush(
                        frontier,
                        (
                            priority,
                            h_val,
                            -g_val,
                            condition_priority,
                            candidate,
                            next(self.counter),
                            center,
                        )
                    )
                queued.add(candidate)
            max_frontier = max(max_frontier, len(frontier))

        def push_next_frontier(center):
            push_frontier_neighbors(center)

        def save_leg(parent, target, path, explored, leg_fgh=None):
            nonlocal current, last_fgh, total_weighted_g, planning_prev_dir
            if path and not self._path_is_contiguous(parent, path):
                return False

            completed_g = total_g + len(full_walk_path)
            priority, distance, _, h_before = self._mode2_h_parts(
                parent, target)
            g_leg = len(path)
            if self.algorithm_name == "A*_v2":
                turns = self._count_turns_in_path(
                    path, parent, planning_prev_dir)
                weighted_leg = (
                    g_leg + turns * algorithms.TURN_PENALTY)
                fgh = (weighted_leg + h_before, g_leg, h_before)
            else:
                weighted_leg = 0.0
                fgh = (completed_g + g_leg + h_before,
                       completed_g + g_leg,
                       h_before)
            sort_key = (h_before, priority, distance, target)
            step_stats = self._mode2_build_step_stats(
                parent, target, path, len(explored), sort_key, completed_g,
                total_weighted_g)

            plan.append(target)
            planned.add(target)
            full_walk_path.extend(path)
            leg_paths[target] = (parent, list(path), fgh, h_before, g_leg)
            leg_stats[target] = (
                parent, list(path), set(explored), fgh, sort_key, step_stats)
            explored_all.update(explored)
            last_fgh = fgh
            if self.algorithm_name == "A*_v2":
                total_weighted_g += weighted_leg
                if path:
                    previous_tile = parent if len(path) == 1 else path[-2]
                    planning_prev_dir = self._step_direction(
                        previous_tile, path[-1])
            current = target
            return True

        while len(planned) < len(remaining_set):
            if current in remaining_set and current not in planned:
                if not save_leg(current, current, [], {current}):
                    break
                push_next_frontier(current)
                continue

            if not frontier:
                target, path, explored, _, _ = (
                    self._mode2_priority_path_to_any_goal(
                        current, remaining_set - planned))
                if target is None:
                    break
                if not save_leg(current, target, path, explored):
                    break
                push_next_frontier(target)
                continue

            if self.algorithm_name == "Greedy":
                _, _, target, _, _ = heapq.heappop(frontier)
            else:
                _, _, _, _, target, _, _ = heapq.heappop(frontier)
            queued.discard(target)
            if target in planned or target not in remaining_set:
                continue
            if target in blocked or target not in self.farm_tiles:
                continue

            if self._is_adjacent_tile(current, target):
                path, explored = [target], {current, target}
                leg_fgh = None
            else:
                path, explored, leg_fgh, _ = self._mode2_plan_one_leg(
                    current, target,
                    prev_dir_override=(
                        planning_prev_dir
                        if self.algorithm_name == "A*_v2" else None))
                if not path and current != target:
                    continue

            if not save_leg(
                    current, target, path, explored, leg_fgh=leg_fgh):
                continue
            push_next_frontier(target)

        self.full_garden_plan = list(plan)
        self.full_mode2_walk_path = list(full_walk_path)
        self.full_garden_leg_paths = leg_paths
        self.full_garden_leg_stats = leg_stats
        self.full_garden_index = 0
        self.full_garden_stats = {
            "Algorithm": self.algorithm_name,
            "Planning mode": "Mode 2 frontier full traversal",
            "Plan targets": len(plan),
            "Plan travel g": len(full_walk_path),
            "Walk path length": len(full_walk_path),
            "Frontier max": max_frontier,
            "Frontier rule": "push only 4-neighbor farm tiles",
            "Priority": (
                "h(n)" if self.algorithm_name == "Greedy"
                else "weighted g(n) + h(n)"
                if self.algorithm_name == "A*_v2"
                else "f(n) = g(n) + h(n)"),
            "Tie-break": (
                "h, condition, tile"
                if self.algorithm_name == "Greedy"
                else "priority, h, -g, condition, tile"),
            "h formula": "Manhattan(candidate, nearest remaining) + 2 * condition(candidate)",
            "g formula": "planned real steps + 1",
            "Path source": "precomputed full walk path",
        }
        self.astar_current_fgh = last_fgh
        self.astar_explored = set(explored_all)
        self._apply_full_plan_stats()
        return self.full_garden_plan


    # ------------------------------------------------------------------
    # mode 3 local search
    # ------------------------------------------------------------------

    def _dfs_candidates(self, candidate_tiles=None):
        if candidate_tiles is None:
            return {
                tile for tile in self.farm_tiles
                if tile not in self.done_tiles
                and tile not in self.discovered_blocked
                and tile not in self.enemy_done_tiles
            }
        return set(candidate_tiles)


    def _dfs_entry_path(self, start, candidates, blocked):
        """Find a route to the first tree without adding connectors to DFS."""
        if start in candidates:
            return [], start

        queue = deque([start])
        parent = {}
        seen = {start}

        while queue:
            current = queue.popleft()
            for next_tile in self._neighbors(current, blocked):
                if next_tile in seen:
                    continue
                seen.add(next_tile)
                parent[next_tile] = current
                if next_tile in candidates:
                    path = [next_tile]
                    while path[-1] != start:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path[1:], next_tile
                queue.append(next_tile)
        return [], None


    @staticmethod
    def _dfs_tree_path(current, target, parent):
        """Build an adjacent route between two consecutive popped nodes."""
        ancestors = {current}
        cursor = current
        while cursor in parent:
            cursor = parent[cursor]
            ancestors.add(cursor)

        down_branch = []
        cursor = target
        while cursor not in ancestors:
            down_branch.append(cursor)
            cursor = parent[cursor]
        common_ancestor = cursor

        path = []
        cursor = current
        while cursor != common_ancestor:
            cursor = parent[cursor]
            path.append(cursor)
        path.extend(reversed(down_branch))
        return path


    def _build_dfs_walk(self, start, candidate_tiles=None):
        candidates = self._dfs_candidates(candidate_tiles)
        blocked = self._blocked_tiles()
        blocked.discard(start)
        for tile in candidates:
            blocked.discard(tile)

        entry_path, dfs_start = self._dfs_entry_path(
            start, candidates, blocked)
        if dfs_start is None:
            self.dfs_walk = [start]
            self.dfs_explored_order = []
            self.dfs_stack_max = 0
            return

        stack = [dfs_start]
        reached = {dfs_start}
        parent = {}
        explored_order = []
        stack_max = 1

        while stack:
            current = stack.pop()
            explored_order.append(current)

            # Match algorithms.dfs(): push left, right, up, down and let
            # stack.pop() select the newest reachable neighbor.
            for next_tile in self._neighbors(current, blocked):
                if next_tile not in candidates:
                    continue
                if next_tile in reached:
                    continue
                reached.add(next_tile)
                parent[next_tile] = current
                stack.append(next_tile)
                stack_max = max(stack_max, len(stack))

        walk = [start, *entry_path]
        current = dfs_start
        for next_popped in explored_order[1:]:
            walk.extend(self._dfs_tree_path(current, next_popped, parent))
            current = next_popped

        self.dfs_walk = walk
        self.dfs_explored_order = explored_order
        self.dfs_stack_max = stack_max

    # -------------------------------------------------------------- MODE 3

    def _mode3_turn_penalty(self):
        """Penalty tang dan sau moi cay da chon/xu ly.

        Muc dich: neu 2 cay cung loai nam ke nhau, o tiep theo van co
        the co score nho hon current score de robot di tiep.
        """
        if self.mode != 3:
            return 0
        # +1 de sau khi xu ly cay dau tien, o ung vien tiep theo bi tru 2.
        # Vi Manhattan cua o ke ben la +1, tru 2 se tao chenhlech -1.
        return len(self.done_tiles) + 1


    def _hill_score(self, tile, start):
        """Mode 3 score = fixed score by tree type + Manhattan - turn penalty.

        dry      = 40
        critical = 30
        pest     = 20
        dead     = 10

        candidate_score = type_score + Manhattan(start, tile) - turn_penalty
        current_score   = type_score + 0

        turn_penalty tang sau moi cay da xu ly, giup 2 cay cung loai nam ke
        nhau van co the tao score giam dan de robot tiep tuc di.
        """
        type_scores = {
            "dry": 40,
            "critical": 30,
            "pest": 20,
            "dead": 10,
        }
        condition = self._condition_for_tile(tile)
        base_score = type_scores.get(condition, 0)
        score = base_score + self._heuristic(start, tile)

        # Chi tru penalty cho cac ung vien chua xu ly, khong tru vao o hien tai.
        # Neu tru ca current va candidate thi hieu ung se bi triet tieu.
        if self.mode == 3 and tile != start and tile not in self.done_tiles:
            score -= self._mode3_turn_penalty()

        return score


    def _local_scores(self, start=None, remaining=None):
        """Tinh lai score theo vi tri hien tai moi buoc.

        Mode 3 score duoc tinh lai moi buoc theo:
        type_score + Manhattan(start, tile) - turn_penalty.
        turn_penalty tang theo so cay da xu ly de giup robot di tiep
        khi gap 2 cay cung loai nam ke nhau.
        """
        start = start or self._world_to_tile(self.player.rect.center)
        tiles = list(remaining) if remaining is not None else list(self.farm_tiles)
        if start in self.farm_tiles and start not in tiles:
            tiles.append(start)

        scores = {
            tile: self._hill_score(tile, start)
            for tile in tiles
        }
        self.hc_scores = scores
        return scores


    def _refresh_mode3_scores_after_action(self, current_tile):
        """Cap nhat lai score ngay sau khi xu ly xong 1 cay Mode 3."""
        if self.mode != 3:
            return
        remaining = [
            tile for tile in self.farm_tiles
            if tile not in self.done_tiles
            and tile not in self.discovered_blocked
            and tile not in self.enemy_done_tiles
        ]
        if remaining:
            self._local_scores(current_tile, remaining)
        else:
            self.hc_scores = {}
        stats = dict(self.stats or {})
        stats["Completed"] = f"{len(self.done_tiles)}/{len(self.farm_tiles)}"
        stats["Score refreshed"] = f"after {current_tile}"
        self.stats = stats


    def _nearest_remaining(self, remaining, start):
        return min(remaining, key=lambda tile: (self._heuristic(start, tile), tile))


    def _remaining_neighbors(self, remaining, center):
        return [
            tile for tile in remaining
            if abs(tile[0] - center[0]) + abs(tile[1] - center[1]) == 1
        ]


    def _set_local_stats(self, algorithm, target, score, extra=None):
        condition = self._condition_for_tile(target) if target is not None else "none"
        stats = {
            "Local algorithm": algorithm,
            "Current score": f"{score:.0f}",
            "Target": f"{target}" if target is not None else "None",
            "Target type": condition,
            "Score formula": "type + Manhattan - turn",
        }
        if extra:
            stats.update(extra)
        self.stats = stats
        self.hc_current_score = score
        self.hc_best_neighbor_score = score


    def _hill_climbing_step_choose(self, remaining, start):
        """Hill Climbing bien the di xuong: chon lang gieng co score nho hon."""
        scores = self._local_scores(start, remaining)

        if start not in self.dryness:
            target = self._nearest_remaining(remaining, start)
            current_score = scores.get(target, 0)
            self._set_local_stats("Hill Climbing", target, current_score, {
                "Hill steps": 0,
                "Trace": [target],
                "Start rule": "Nearest target first",
                "Turn penalty": f"-{self._mode3_turn_penalty()} candidate",
            })
            return target

        current_score = self._hill_score(start, start)
        neighbors = self._remaining_neighbors(remaining, start)
        lower_neighbors = [
            tile for tile in neighbors
            if scores.get(tile, 0) < current_score
        ]
        if not lower_neighbors:
            self._set_local_stats("Hill Climbing", None, current_score, {
                "Hill steps": 0,
                "Trace": [start],
                "Stop rule": "No lower neighbor",
                "Remaining tiles": len(remaining),
                "Coverage": f"{len(self.done_tiles)}/{len(self.farm_tiles)}",
                "Suggestion": "No adjacent tile has a lower score",
            })
            self.state = "DONE"
            return None

        # Test variant: di xuong nhanh.
        # lower_neighbors chi gom cac o co score nho hon current_score.
        # min o day chon o co score thap nhat trong cac lang gieng thap hon.
        target = min(
            lower_neighbors,
            key=lambda tile: (scores.get(tile, 0), self._heuristic(start, tile), tile))
        target_score = scores.get(target, 0)
        self._set_local_stats("Hill Climbing", target, target_score, {
            "Hill steps": 1,
            "Trace": [start, target],
            "Stop rule": "Move to lowest lower neighbor",
            "Turn penalty": f"-{self._mode3_turn_penalty()} candidate",
        })
        return target


    def _restart_hill_choose(self, remaining, start):
        """Restart Hill: di xuong nhu Hill, neu ket thi khoi dong lai o tile co score cao nhat (xa nhat)."""
        scores = self._local_scores(start, remaining)
        if start not in self.dryness:
            target = self._nearest_remaining(remaining, start)
            self._set_local_stats("Restart Hill", target, scores.get(target, 0), {
                "Start rule": "Nearest target first",
                "Restarts": 0,
                "Trace": [target],
            })
            return target

        current_score = self._hill_score(start, start)
        lower_neighbors = [
            tile for tile in self._remaining_neighbors(remaining, start)
            if scores.get(tile, 0) < current_score
        ]
        if lower_neighbors:
            target = min(
                lower_neighbors,
                key=lambda tile: (scores.get(tile, 0),
                                  self._heuristic(start, tile), tile))
            self._set_local_stats("Restart Hill", target, scores.get(target, 0), {
                "Restarts": 0,
                "Trace": [start, target],
                "Rule": "Continue downhill",
            })
            return target

        # Bi ket cuc bo: restart tai tile co score cao nhat (chua xu ly)
        target = max(
            remaining,
            key=lambda tile: (scores.get(tile, 0), -self._heuristic(start, tile), tile))
        self._set_local_stats("Restart Hill", target, scores.get(target, 0), {
            "Restarts": 1,
            "Trace": [target],
            "Rule": "Restart at highest unvisited tile",
        })
        return target


    def _local_beam_step_choose(self, remaining, start, beam_width=3):
        """Local Beam: giu k ung vien, sinh tat ca lang gieng, chon k tot nhat."""
        scores = self._local_scores(start, remaining)
        if start not in self.dryness:
            beam = sorted(
                remaining,
                key=lambda tile: (self._heuristic(start, tile), scores.get(tile, 0), tile)
            )[:beam_width]
        else:
            seed = self._remaining_neighbors(remaining, start)
            if not seed:
                seed = list(remaining)
            beam = sorted(
                seed,
                key=lambda tile: (scores.get(tile, 0), self._heuristic(start, tile), tile)
            )[:beam_width]

        children = []
        for tile in beam:
            children.extend(self._remaining_neighbors(remaining, tile))
        pool = set(beam) | set(children)
        next_beam = sorted(
            pool,
            key=lambda tile: (scores.get(tile, 0), self._heuristic(start, tile), tile)
        )[:max(1, min(beam_width, len(pool)))]
        target = next_beam[0] if next_beam else None
        target_score = scores.get(target, 0) if target is not None else 0
        self._set_local_stats("Local Beam", target, target_score, {
            "Beam width": beam_width,
            "Beam": f"{next_beam}",
            "Generated": len(children),
            "Rule": "Lowest score in successor beam",
        })
        return target


    def _annealing_step_choose(self, remaining, start):
        """Simulated Annealing: chon lang gieng, chap nhan buoc te hon theo nhiet do."""
        scores = self._local_scores(start, remaining)
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

        current_score = self._hill_score(start, start)
        ordered = list(neighbors)
        random.shuffle(ordered)
        accepted = None
        accepted_delta = 0
        accepted_worse = False
        for candidate in ordered:
            delta = scores.get(candidate, 0) - current_score
            # Downhill: delta < 0 means score decreased = improvement
            if delta <= 0:
                accepted = candidate
                accepted_delta = delta
                break
            # Accept worse (higher score) with probability e^(-delta/T)
            probability = math.exp(-delta / max(self.anneal_temperature, 0.001))
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

    # ------------------------------------------------------------------
    # mode 4 online / belief / AND-OR search
    # ------------------------------------------------------------------

    def _draw_belief_minimap(self, surface):
        if not self.rows or not self.cols:
            return

        max_size = 170
        cell = max(3, min(8, max_size // max(self.rows, self.cols)))
        pad = 8
        map_w = self.cols * cell
        map_h = self.rows * cell
        panel = pygame.Rect(0, 0, map_w + pad * 2, map_h + pad * 2)
        panel.topright = (surface.get_width() - 18, 18)

        bg = pygame.Surface(panel.size, pygame.SRCALPHA)
        bg.fill((12, 14, 18, 210))
        surface.blit(bg, panel.topleft)
        pygame.draw.rect(surface, (235, 235, 210), panel, 1)

        risk_map = self._last_risk_map or {}
        if self.walkable_tiles is None:
            navigable = {
                (x, y)
                for y in range(self.rows)
                for x in range(self.cols)
            }
        else:
            navigable = set(self.walkable_tiles)

        for y in range(self.rows):
            for x in range(self.cols):
                tile = (x, y)
                if tile not in navigable:
                    color = (38, 42, 48)
                elif tile in self.discovered_blocked:
                    color = (18, 20, 24)
                elif tile in self.explored_tiles:
                    color = (218, 232, 205)
                else:
                    risk = max(0.0, min(1.0, float(risk_map.get(tile, 0.0))))
                    color = (255, int(220 - 130 * risk), int(55 - 35 * risk))

                rect = pygame.Rect(
                    panel.x + pad + x * cell,
                    panel.y + pad + y * cell,
                    cell,
                    cell)
                pygame.draw.rect(surface, color, rect)

        target = self.current_target
        if target is None and self.path:
            target = self.path[-1]
        if target is not None:
            tx, ty = target
            if 0 <= tx < self.cols and 0 <= ty < self.rows:
                rect = pygame.Rect(
                    panel.x + pad + tx * cell,
                    panel.y + pad + ty * cell,
                    cell,
                    cell)
                pygame.draw.rect(surface, (255, 245, 70), rect)

        robot = self._world_to_tile(self.player.rect.center)
        rx, ry = robot
        if 0 <= rx < self.cols and 0 <= ry < self.rows:
            rect = pygame.Rect(
                panel.x + pad + rx * cell,
                panel.y + pad + ry * cell,
                cell,
                cell)
            pygame.draw.rect(surface, (65, 235, 95), rect)


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


    def _compute_frontier_tiles(self):
        if self.mode != 4:
            return set()

        frontier = set()
        known_free = set(self.known_free or self.explored_tiles)
        unknown_tiles = self._mode4_unknown_tiles()
        blocked = self._blocked_tiles()
        for tile in known_free:
            if tile in blocked:
                continue
            for neighbor in self._neighbors(tile, blocked):
                if neighbor in unknown_tiles:
                    frontier.add(tile)
                    break
        return frontier


    def _mode4_initial_known_free(self):
        known = {self.spawn_tile}
        if self.mode4_entry_tile is not None:
            known.add(self.mode4_entry_tile)
        return {
            tile for tile in known
            if (self.walkable_tiles is None or tile in self.walkable_tiles)
            and tile not in self.hidden_blocked
        }


    def _mode4_unknown_tiles(self):
        candidates = set(self.farm_tiles)
        return candidates - self.known_free - self.discovered_blocked


    def _init_mode4_belief(self):
        if self.mode != 4:
            return
        self.known_free = self._mode4_initial_known_free()
        self.explored_tiles.update(self.known_free)
        self.discovered_blocked.intersection_update(set(self.farm_tiles))
        self.belief_unknown_tiles = self._mode4_unknown_tiles()
        self.belief_worlds = algorithms.generate_belief_worlds(
            self.belief_unknown_tiles, max_hidden=self.belief_max_hidden)
        self.belief_worlds_before_observation = len(self.belief_worlds)
        self.belief_worlds_after_observation = len(self.belief_worlds)
        self.belief_inconsistent = False
        self.belief_context = "persistent"
        self._last_risk_map = algorithms.belief_risk_map(
            self.belief_worlds, self.belief_unknown_tiles)


    def _rebuild_mode4_belief_from_observations(self):
        self.belief_unknown_tiles = self._mode4_unknown_tiles()
        remaining_hidden_budget = max(
            0, self.belief_max_hidden - len(self.discovered_blocked))
        worlds = algorithms.generate_belief_worlds(
            self.belief_unknown_tiles, max_hidden=remaining_hidden_budget)
        known_blocked = set(self.discovered_blocked)
        self.belief_worlds = [
            world for world in worlds
            if world.isdisjoint(self.known_free)
            and known_blocked.isdisjoint(self.known_free)
        ]
        self.belief_worlds_after_observation = len(self.belief_worlds)
        self.belief_inconsistent = not self.belief_worlds
        self._last_risk_map = algorithms.belief_risk_map(
            self.belief_worlds, self.belief_unknown_tiles)


    def _update_belief_state(self):
        if self.mode != 4:
            return

        self.frontier_tiles = self._compute_frontier_tiles()
        unknown_tiles = self._mode4_unknown_tiles()
        self.belief_state = {
            "explored": len(self.explored_tiles),
            "known_free": len(self.known_free),
            "unknown": len(unknown_tiles),
            "frontier": len(self.frontier_tiles),
            "discovered_blocked": len(self.discovered_blocked),
            "hidden_remaining": len(
                self.hidden_blocked - self.discovered_blocked),
            "belief_worlds": len(self.belief_worlds),
            "belief_before": self.belief_worlds_before_observation,
            "belief_after": self.belief_worlds_after_observation,
            "inconsistent": self.belief_inconsistent,
            "replans": self.replan_count,
            "phase": self.mode4_phase,
        }


    def _belief_bfs_plan(self, start, goal, blocked):
        path, explored, bfs_stats = algorithms.belief_state_bfs(
            start, goal, blocked, self.belief_worlds,
            self.belief_unknown_tiles, self._neighbors)
        risky_steps = sum(tile in self.belief_unknown_tiles for tile in path)
        stats = dict(bfs_stats)
        stats.update({
            "Algorithm": self.algorithm_name,
            "Current node": f"{start}",
            "Next node": f"{path[0] if path else start}",
            "Target rule": "belief-state BFS to current target",
            "Neighbor rule": "push only 4 adjacent neighbors",
            "Risky steps": risky_steps,
        })
        return path, explored, (0, len(path), 0), stats


    def _choose_mode4_target(self, remaining, start):
        if self.algorithm_name in ("Belief-State BFS", "Belief BFS", "Belief A*"):
            target, path, explored, fgh, stats = (
                self._mode4_belief_state_target(start, remaining))
        elif self.algorithm_name in (
                "Online A*", "Online BFS", "Belief BFS", "Belief A*"):
            if self.algorithm_name in ("Online BFS", "Belief BFS", "Belief A*"):
                target, path, explored, fgh, stats = (
                    self._mode4_online_bfs_frontier_target(start, remaining))
            else:
                target, path, explored, fgh, stats = (
                    self._mode4_online_astar_frontier_target(start, remaining))
        elif self.algorithm_name == "AND-OR Search":
            all_seen_targets = self._and_or_seen_targets_now()
            seen_targets = self._and_or_planning_targets(
                start, all_seen_targets)
            self.and_or_seen_targets = seen_targets
            resume_target = self.and_or_resume_target
            if self._and_or_target_valid(resume_target):
                target = resume_target
                self.and_or_resume_target = None
                target_rule = "resume original target after drift rescue"
            elif seen_targets:
                target = min(
                    seen_targets,
                    key=lambda tile: (
                        self._mode2_h_parts(start, tile)[3],
                        self._mode2_condition_h(tile),
                        self._heuristic(start, tile),
                        tile))
                target_rule = "same h(n)=distance+2*condition as Mode 2 A*"
            else:
                target = None

            if target is not None:
                priority, distance, condition_cost, h_val = (
                    self._mode2_h_parts(start, target))
                path, explored, fgh, stats = [], {start}, (0, 0, 0), {
                    "Algorithm": self.algorithm_name,
                    "Selected target": f"{target}",
                    "Seen targets": len(seen_targets),
                    "Seen targets total": len(all_seen_targets),
                    "Planned seen targets": len(seen_targets),
                    "Remaining seen targets": len(seen_targets),
                    "Target rule": target_rule,
                    "h(n)": f"{h_val}",
                    "h distance": f"{distance}",
                    "condition": f"{priority} -> +{condition_cost}",
                    "OR node": "choose A* target",
                    "AND outcomes": "70 correct / 10 / 10 / 10 drift",
                }
            else:
                target, path, explored, fgh, stats = (
                    None, [], set(), (0, 0, 0), {
                        "Algorithm": self.algorithm_name,
                        "Selected target": "None",
                        "Seen targets": 0,
                        "Target rule": "Wait for visible/seen targets",
                    })
        else:
            target, path, explored, fgh, stats = (
                self._mode4_priority_queue_astar_to_any_goal(
                    start, remaining))
        self.chosen_target_path = None
        if target is None:
            self.stats = stats
            return None
        if self.algorithm_name in (
                "Online A*", "Online BFS",
                "Belief-State BFS", "Belief BFS", "Belief A*"):
            self.chosen_target_path = list(path)
        self.astar_explored = set(explored)
        self.astar_current_fgh = fgh
        self.mode4_phase = "EXPLORE"
        base_stats = {
            "Algorithm": self.algorithm_name,
            "Selected target": f"{target}",
            "Target rule": (
                "heapq.heappop(frontier)"
                if self.algorithm_name in (
                    "Online A*", "Online BFS",
                    "Belief-State BFS", "Belief BFS", "Belief A*")
                else "target selected only when (node == target) "
                     "is popped from PQ"),
            "Path source": (
                "online frontier + algorithm core"
                if self.algorithm_name in (
                    "Online A*", "Online BFS",
                    "Belief-State BFS", "Belief BFS", "Belief A*")
                else "planner runs per algorithm"),
        }
        if self.algorithm_name in (
                "Online A*", "Online BFS",
                "Belief-State BFS", "Belief BFS", "Belief A*"):
            if self.algorithm_name in (
                    "Online BFS", "Belief-State BFS",
                    "Belief BFS", "Belief A*"):
                priority_keys = (
                    "Algorithm", "Selected target", "Current tile",
                    "Next node", "Frontier size", "Frontier max",
                    "Frontier rule", "Target rule", "Neighbor rule",
                    "Path source", "Path length", "Nodes explored",
                    "Queue max", "Queued farms",
                )
            else:
                priority_keys = (
                    "Algorithm", "Selected target", "f(n)", "g(n)", "h(n)",
                    "condition", "h distance", "Current tile", "Next node",
                    "Frontier size", "Frontier max", "Frontier rule",
                    "Target rule", "Neighbor rule", "Path source",
                    "Path length", "Queued farms",
                )
            ordered_stats = {}
            merged_stats = {**base_stats, **stats}
            for key in priority_keys:
                if key in merged_stats:
                    ordered_stats[key] = merged_stats[key]
            for key, val in merged_stats.items():
                ordered_stats.setdefault(key, val)
            self.stats = ordered_stats
        else:
            self.stats = base_stats
            self.stats.update(stats)
        return target


    def _mode4_belief_tile_possible(self, tile):
        if tile in self.discovered_blocked:
            return False
        if tile not in self.belief_unknown_tiles:
            return True
        if not self.belief_worlds:
            return False
        return any(tile not in world for world in self.belief_worlds)


    def _mode4_belief_add_four_neighbors(self, tile):
        if not self.mode4_belief_bfs_started:
            self.mode4_belief_bfs_started = True
            self.mode4_belief_bfs_visited.add(tile)
            self.mode4_belief_bfs_parent.setdefault(tile, None)

        x, y = tile
        added = 0
        for next_tile in (
                (x - 1, y), (x + 1, y),
                (x, y - 1), (x, y + 1)):
            if not (0 <= next_tile[0] < self.cols
                    and 0 <= next_tile[1] < self.rows):
                continue
            if (self.walkable_tiles is not None
                    and next_tile not in self.walkable_tiles):
                continue
            if next_tile in self.mode4_belief_bfs_visited:
                continue
            if not self._mode4_belief_tile_possible(next_tile):
                continue
            self.mode4_belief_bfs_visited.add(next_tile)
            self.mode4_belief_bfs_parent[next_tile] = tile
            self.mode4_belief_bfs_queue.append(next_tile)
            added += 1
        self.mode4_belief_bfs_frontier_max = max(
            self.mode4_belief_bfs_frontier_max,
            len(self.mode4_belief_bfs_queue))
        return added


    def _mode4_belief_tree_path(self, start, target):
        if start == target:
            return []

        def ancestors(tile):
            result = []
            current = tile
            seen = set()
            while current is not None and current not in seen:
                result.append(current)
                seen.add(current)
                current = self.mode4_belief_bfs_parent.get(current)
            return result

        start_chain = ancestors(start)
        target_chain = ancestors(target)
        start_index = {tile: index for index, tile in enumerate(start_chain)}
        lca = None
        target_lca_index = 0
        for index, tile in enumerate(target_chain):
            if tile in start_index:
                lca = tile
                target_lca_index = index
                break
        if lca is None:
            return [target] if self._heuristic(start, target) == 1 else []
        up_path = start_chain[1:start_index[lca] + 1]
        down_path = list(reversed(target_chain[:target_lca_index]))
        return up_path + down_path


    def _mode4_belief_state_target(self, start, remaining):
        added = self._mode4_belief_add_four_neighbors(start)
        unknown_tiles = self._mode4_unknown_tiles()
        while self.mode4_belief_bfs_queue:
            target = self.mode4_belief_bfs_queue.popleft()
            if target in self.discovered_blocked:
                continue
            if not self._mode4_belief_tile_possible(target):
                continue
            path = self._mode4_belief_tree_path(start, target)
            if not path and start != target:
                continue
            if not self._path_is_contiguous(start, path):
                continue
            self.mode4_belief_bfs_explored.add(target)
            stats = {
                "Algorithm": "Belief-State BFS",
                "Search style": "single BFS queue for whole Mode 4",
                "Queue rule": "add 4 directions, then FIFO pop",
                "Neighbor order": "left, right, up, down",
                "Selected target": f"{target}",
                "Current tile": f"{start}",
                "Next node": f"{path[0] if path else start}",
                "Path length": len(path),
                "Nodes explored": len(self.mode4_belief_bfs_explored),
                "Queue size": len(self.mode4_belief_bfs_queue),
                "Queue max": self.mode4_belief_bfs_frontier_max,
                "Adjacent added": added,
                "Possible worlds": len(self.belief_worlds),
                "Belief unknowns": len(self.belief_unknown_tiles),
                "Unknown tiles": len(unknown_tiles),
                "Known free tiles": len(self.known_free),
                "Discovered blocked": len(self.discovered_blocked),
                "Policy": "global BFS traversal; no per-target BFS call",
            }
            return target, path, set(self.mode4_belief_bfs_explored), (
                0, len(path), 0), stats

        return None, [], set(), (0, 0, 0), {
            "Algorithm": "Belief-State BFS",
            "Map size": "5x5",
            "Real hidden rocks": len(self.hidden_blocked),
            "Robot prior": f"at most {self.belief_max_hidden} hidden rocks",
            "Spawn outside": "yes" if self.spawn_tile not in self.farm_tiles else "no",
            "Entry tile": f"{self.mode4_entry_tile}",
            "Selected target": "None",
            "Frontier tiles": len(self.frontier_tiles),
            "Unknown tiles": len(unknown_tiles),
            "Known free tiles": len(self.known_free),
            "Discovered blocked": len(self.discovered_blocked),
            "Worlds before obs": self.belief_worlds_before_observation,
            "Worlds after obs": self.belief_worlds_after_observation,
            "Current belief size": len(self.belief_worlds),
            "Possible worlds": len(self.belief_worlds),
            "Queue size": len(self.mode4_belief_bfs_queue),
            "Result": (
                "Inconsistent belief: no possible worlds remain"
                if self.belief_inconsistent
                else "Global BFS queue empty"),
            "Policy": "global BFS traversal; no per-target BFS call",
        }


    def _mode4_online_astar_reset(self, start=None, remaining=None):
        self.mode4_online_astar_open = []
        self.mode4_online_astar_queued = set()
        self.mode4_online_astar_parent = {}
        self.mode4_online_astar_g_score = {}
        self.mode4_online_astar_f_score = {}
        self.mode4_online_astar_explored = set()
        self.mode4_online_astar_root = start
        self.mode4_online_astar_remaining_key = (
            tuple(sorted(remaining)) if remaining is not None else None)
        self.mode4_online_astar_frontier_max = 0
        self.mode4_online_astar_started = False


    def _mode4_online_bfs_reset(self):
        self.mode4_online_bfs_queue = deque()
        self.mode4_online_bfs_queued = set()
        self.mode4_online_bfs_parent = {}
        self.mode4_online_bfs_explored = set()
        self.mode4_online_bfs_frontier_max = 0


    def _mode4_belief_bfs_reset(self):
        self.mode4_belief_bfs_queue = deque()
        self.mode4_belief_bfs_visited = set()
        self.mode4_belief_bfs_parent = {}
        self.mode4_belief_bfs_explored = set()
        self.mode4_belief_bfs_frontier_max = 0
        self.mode4_belief_bfs_started = False


    def _mode4_online_bfs_add_adjacent_farms(self, current, remaining_set,
                                             blocked):
        added = 0
        for candidate in self._neighbors(current, blocked):
            if candidate not in remaining_set:
                continue
            if candidate in self.mode4_online_bfs_queued:
                continue
            if candidate in blocked or candidate in self.discovered_blocked:
                continue
            if candidate in self.done_tiles or candidate in self.enemy_done_tiles:
                continue
            self.mode4_online_bfs_queue.append(candidate)
            self.mode4_online_bfs_queued.add(candidate)
            self.mode4_online_bfs_parent[candidate] = current
            added += 1
        self.mode4_online_bfs_frontier_max = max(
            self.mode4_online_bfs_frontier_max,
            len(self.mode4_online_bfs_queue))
        return added


    def _mode4_online_bfs_frontier_target(self, start, remaining):
        remaining_set = {
            tile for tile in remaining
            if tile in self.farm_tiles
            and tile not in self.done_tiles
            and tile not in self.discovered_blocked
            and tile not in self.enemy_done_tiles
        }
        if not remaining_set:
            return None, [], set(), (0, 0, 0), {
                "Algorithm": self.algorithm_name,
                "Path length": 0,
                "Result": "No target candidates",
            }

        blocked = self._blocked_tiles()
        blocked.discard(start)
        added = self._mode4_online_bfs_add_adjacent_farms(
            start, remaining_set, blocked)

        while self.mode4_online_bfs_queue:
            target = self.mode4_online_bfs_queue.popleft()
            self.mode4_online_bfs_queued.discard(target)
            if target in blocked or target in self.discovered_blocked:
                self.mode4_online_bfs_parent.pop(target, None)
                continue
            if target not in remaining_set or target in self.done_tiles:
                continue
            target_blocked = set(blocked)
            target_blocked.discard(target)
            path, explored, _, core_stats = algorithms.find_path_by_algorithm(
                "BFS", start, target, target_blocked, self._neighbors,
                self._search_heuristic, self.counter, self._step_cost)
            if start != target and not path:
                continue
            if not self._path_is_contiguous(start, path):
                continue
            self.mode4_online_bfs_explored.update(explored)
            stats = {
                "Algorithm": self.algorithm_name,
                "Selected target": f"{target}",
                "Current tile": f"{start}",
                "Next node": f"{path[0] if path else start}",
                "Frontier size": len(self.mode4_online_bfs_queue),
                "Frontier max": self.mode4_online_bfs_frontier_max,
                "Frontier rule": "FIFO queue stores discovered farm tiles",
                "Target rule": "popleft() first discovered farm",
                "Neighbor rule": "add farm tiles in 4 adjacent directions",
                "Path source": (
                    "Belief-state BFS in _online_plan"
                    if self.algorithm_name in (
                        "Belief-State BFS", "Belief BFS", "Belief A*")
                    else "BFS core from Mode 1"),
                "Path length": len(path),
                "Nodes explored": len(explored),
                "Queue max": core_stats.get("Queue max", "-"),
                "Queued farms": len(self.mode4_online_bfs_queued),
                "Adjacent farms added": added,
            }
            return target, path, set(self.mode4_online_bfs_explored), (0, 0, 0), stats

        return None, [], set(self.mode4_online_bfs_explored), (0, 0, 0), {
            "Algorithm": self.algorithm_name,
            "Path length": 0,
            "Frontier size": 0,
            "Frontier max": self.mode4_online_bfs_frontier_max,
            "Queued farms": len(self.mode4_online_bfs_queued),
            "Adjacent farms added": added,
            "Result": "FIFO frontier empty",
            "Frontier rule": "FIFO queue stores discovered farm tiles",
            "Target rule": "popleft() first discovered farm",
        }


    def _mode4_online_astar_push_farm_candidate(self, current, candidate,
                                                remaining_set, blocked):
        if candidate not in remaining_set:
            return False
        if candidate in blocked or candidate in self.discovered_blocked:
            return False
        if candidate in self.done_tiles or candidate in self.enemy_done_tiles:
            return False
        parent_g = self.mode4_online_astar_g_score.get(current)
        if parent_g is None:
            parent_g = (
                0 if current == self.mode4_online_astar_root
                else self.mode4_completed_g)
            self.mode4_online_astar_g_score[current] = parent_g
        g_val = parent_g + 1
        old_g = self.mode4_online_astar_g_score.get(candidate, math.inf)
        if candidate in self.mode4_online_astar_queued and g_val >= old_g:
            return False
        condition = self._mode2_condition_h(candidate)
        distance = self._heuristic(current, candidate)
        h_val = distance + 2 * condition
        f_val = g_val + h_val
        self.mode4_online_astar_queued.add(candidate)
        self.mode4_online_astar_parent[candidate] = current
        self.mode4_online_astar_g_score[candidate] = g_val
        self.mode4_online_astar_f_score[candidate] = f_val
        heapq.heappush(
            self.mode4_online_astar_open,
            (
                f_val, h_val, g_val, condition,
                next(self.counter), candidate,
            ))
        self.mode4_online_astar_frontier_max = max(
            self.mode4_online_astar_frontier_max,
            len(self.mode4_online_astar_open))
        return True


    def _mode4_online_astar_add_adjacent_farms(self, current, remaining_set,
                                               blocked):
        added = 0
        for candidate in self._neighbors(current, blocked):
            if self._mode4_online_astar_push_farm_candidate(
                    current, candidate, remaining_set, blocked):
                added += 1
        return added


    def _mode4_online_astar_frontier_target(self, start, remaining):
        """Mode 4 Online A*: persistent PQ containing only adjacent farms."""
        remaining_set = {
            tile for tile in remaining
            if tile in self.farm_tiles
            and tile not in self.done_tiles
            and tile not in self.discovered_blocked
            and tile not in self.enemy_done_tiles
        }
        if not remaining_set:
            return None, [], set(), (0, 0, 0), {
                "Algorithm": self.algorithm_name,
                "Path length": 0,
                "Result": "No target candidates",
            }

        blocked = self._blocked_tiles()
        blocked.discard(start)
        if not self.mode4_online_astar_started:
            self.mode4_online_astar_started = True
            self.mode4_online_astar_root = start
            self.mode4_online_astar_g_score[start] = 0
        added = self._mode4_online_astar_add_adjacent_farms(
            start, remaining_set, blocked)

        while self.mode4_online_astar_open:
            (
                current_f, current_h, current_g, condition,
                _counter, target,
            ) = heapq.heappop(self.mode4_online_astar_open)
            if target in blocked or target in self.discovered_blocked:
                self.mode4_online_astar_queued.discard(target)
                self.mode4_online_astar_parent.pop(target, None)
                self.mode4_online_astar_g_score.pop(target, None)
                self.mode4_online_astar_f_score.pop(target, None)
                continue
            if target not in remaining_set or target in self.done_tiles:
                self.mode4_online_astar_queued.discard(target)
                continue
            if current_f != self.mode4_online_astar_f_score.get(target):
                continue
            if current_g != self.mode4_online_astar_g_score.get(target):
                continue
            target_blocked = set(blocked)
            target_blocked.discard(target)
            if self.algorithm_name == "Online BFS":
                path, explored, _, core_stats = algorithms.find_path_by_algorithm(
                    "BFS", start, target, target_blocked, self._neighbors,
                    self._search_heuristic, self.counter, self._step_cost)
                path_source = "BFS core from Mode 1"
            else:
                path, explored, _, core_stats = algorithms.astar_4dir_f_update(
                    start, target, target_blocked, self._neighbors,
                    self._heuristic, self.counter)
                path_source = "A* 4-dir from current start to popped farm"
            if start != target and not path:
                continue
            if not self._path_is_contiguous(start, path):
                continue
            self.mode4_online_astar_queued.discard(target)
            self.mode4_online_astar_explored.update(explored)
            next_node = path[0] if path else start
            stats = {
                "Algorithm": self.algorithm_name,
                "Selected target": f"{target}",
                "f(n)": f"{current_f}",
                "g(n)": f"{current_g}",
                "h(n)": f"{current_h}",
                "condition": f"{condition}",
                "h distance": f"{self._heuristic(start, target)}",
                "Current tile": f"{start}",
                "Next node": f"{next_node}",
                "Frontier size": len(self.mode4_online_astar_open),
                "Frontier max": self.mode4_online_astar_frontier_max,
                "Frontier rule": "persistent PQ stores only farm tiles",
                "Target rule": "heapq.heappop(farm frontier)",
                "Neighbor rule": "add farm tiles in 4 adjacent directions",
                "Path source": path_source,
                "Path length": len(path),
                "Nodes explored": len(explored),
                "Queue max": core_stats.get("Queue max", "-"),
                "g completed": f"{self.mode4_completed_g}",
                "Queued farms": len(self.mode4_online_astar_queued),
                "Adjacent farms added": added,
            }
            return (
                target, path,
                set(self.mode4_online_astar_explored),
                (current_f, current_g, current_h), stats)

        return None, [], set(self.mode4_online_astar_explored), (0, 0, 0), {
            "Algorithm": self.algorithm_name,
            "Path length": 0,
            "Frontier size": 0,
            "Frontier max": self.mode4_online_astar_frontier_max,
            "Queued farms": len(self.mode4_online_astar_queued),
            "Adjacent farms added": added,
            "Result": "No adjacent farm in persistent PQ",
            "Frontier rule": "persistent PQ stores only farm tiles",
            "Neighbor rule": "add farm tiles in 4 adjacent directions",
        }


    def _update_mode4_online_astar_hud(self, current, next_tile, target):
        if (self.mode != 4
                or self.algorithm_name != "Online A*"
                or target is None):
            return
        display_node = current
        if self.mode4_active_step is not None:
            display_node = self.mode4_active_step[0]
        condition = self._mode2_condition_h(target)
        display_f, display_g, display_h = self.astar_current_fgh
        self.stats.update({
            "Selected target": f"{target}",
            "Target": f"{target}",
            "f(n)": f"{display_f}",
            "g(n)": f"{display_g}",
            "h(n)": f"{display_h}",
            "condition": f"{condition}",
            "Current tile": f"{display_node}",
            "Next node": f"{next_tile}",
            "Display f from robot": f"{display_f}",
            "Frontier size": len(self.mode4_online_astar_open),
            "Frontier max": self.mode4_online_astar_frontier_max,
            "Frontier rule": "persistent PQ stores only farm tiles",
            "Target rule": "heapq.heappop(farm frontier)",
            "Neighbor rule": "add farm tiles in 4 adjacent directions",
        })


    def _mode4_priority_queue_astar_to_any_goal(self, start, remaining,
                                                step_cost=None):
        """Select a Mode-4 target only when it is popped from one A* heap."""
        candidates = [
            tile for tile in remaining
            if tile not in self.done_tiles
            and tile not in self.discovered_blocked
            and tile not in self.enemy_done_tiles
        ]
        if not candidates:
            return None, [], set(), (0, 0, 0), {
                "Algorithm": self.algorithm_name,
                "Path length": 0,
                "Result": "No target candidates",
            }

        blocked = self._blocked_tiles()
        blocked.discard(start)
        for target in candidates:
            blocked.discard(target)

        open_set = []
        came_from = {}
        f_score = {}
        explored = set()
        max_frontier = 0

        def condition_priority(tile):
            return self._mode2_condition_h(tile)

        def state_priority(node, target):
            g_to_target = self._heuristic(node, target)
            condition = condition_priority(target)
            f_val = g_to_target + 2 * condition
            return f_val, g_to_target, condition

        def push_state(node, target, parent=None):
            f_val, g_to_target, condition = state_priority(node, target)
            state = (node, target)
            if f_val >= f_score.get(state, math.inf):
                return False
            f_score[state] = f_val
            if parent is not None:
                came_from[state] = parent
            heapq.heappush(
                open_set,
                # Heap tie-break: f, condition, distance, target coords.
                (f_val, condition, g_to_target, target,
                 next(self.counter), node, target))
            return True

        def reconstruct_state_path(node, target):
            state = (node, target)
            path = [node]
            while state in came_from:
                state = came_from[state]
                path.append(state[0])
            path.reverse()
            return path[1:]

        for target in candidates:
            if push_state(start, target):
                max_frontier = max(max_frontier, len(open_set))

        while open_set:
            (
                current_f, current_condition, current_g_to_target, _,
                _, current, target
            ) = heapq.heappop(open_set)
            state = (current, target)
            if current_f > f_score.get(state, math.inf):
                continue

            explored.add(current)
            if current == target:
                path = reconstruct_state_path(current, target)
                if not self._path_is_contiguous(start, path):
                    return None, [], explored, (0, 0, 0), {
                        "Algorithm": self.algorithm_name,
                        "Path length": 0,
                        "Nodes explored": len(explored),
                        "Frontier max": max_frontier,
                        "Result": "Non-contiguous path rejected",
                        "Update rule": "only update when f_new < f_old",
                        "Neighbor rule": "push only 4 adjacent neighbors",
                    }
                condition_cost = 2 * current_condition
                display_g = len(path)
                display_h = condition_cost
                display_f = display_g + display_h
                return target, path, explored, (
                    display_f, display_g, display_h), {
                    "Top PQ candidate": (
                        f"node={start}, target={target}, f={display_f}"),
                    "Selected target": f"{target}",
                    "f(n)": f"{display_f}",
                    "g(n)": f"{display_g}",
                    "condition": f"{current_condition}",
                    "h(n)": f"{display_h}",
                    "g leg": f"{display_g}",
                    "g completed": f"{self.mode4_completed_g}",
                    "Current node": f"{start}",
                    "Target carried in PQ": f"{target}",
                    "Next node": f"{path[0] if path else start}",
                    "Path length": len(path),
                    "Nodes explored": len(explored),
                    "Frontier max": max_frontier,
                    "Popped node": f"{current}",
                    "Popped target": f"{target}",
                    "Popped f": f"{current_f}",
                    "Display f from robot": f"{display_f}",
                    "h condition": (
                        f"2 * {current_condition} = {display_h} "
                        f"({self._condition_for_tile(target)})"),
                    "Candidate targets": len(candidates),
                    "Search style": "single priority queue",
                    "Replan every step": "yes",
                    "Reason": (
                        "Online A* replans after each observed tile"),
                    "Target rule": (
                        "target selected only when (node == target) "
                        "is popped from PQ"),
                    "Update rule": "only update when f_new < f_old",
                    "Neighbor rule": "push only 4 adjacent neighbors",
                }

            for neighbor in self._neighbors(current, blocked):
                if self._heuristic(current, neighbor) != 1:
                    continue
                if push_state(neighbor, target, state):
                    max_frontier = max(max_frontier, len(open_set))

        return None, [], explored, (0, 0, 0), {
            "Algorithm": self.algorithm_name,
            "Path length": 0,
            "Nodes explored": len(explored),
            "Frontier max": max_frontier,
            "Result": "No reachable target",
            "Candidate targets": len(candidates),
            "Search style": "single priority queue",
            "Target rule": (
                "target selected only when (node == target) is popped from PQ"),
            "Update rule": "only update when f_new < f_old",
            "Neighbor rule": "push only 4 adjacent neighbors",
        }



    def _expand_vision(self, center):
        """Má»Ÿ rá»™ng vĂ¹ng Ä‘Ă£ khĂ¡m phĂ¡ quanh center."""
        before_explored = set(self.explored_tiles)
        before_blocked = set(self.discovered_blocked)
        self.belief_worlds_before_observation = len(self.belief_worlds)
        algorithms.expand_vision(
            center, self.vision_radius, self.rows, self.cols,
            self.hidden_blocked, self.explored_tiles,
            self.discovered_blocked)
        observed_tiles = self.explored_tiles - before_explored
        observed_blocked = self.discovered_blocked - before_blocked
        observed_free = observed_tiles - self.discovered_blocked
        observed_free = {
            tile for tile in observed_free
            if self.walkable_tiles is None or tile in self.walkable_tiles
        }
        self.known_free.update(observed_free)
        self.known_free.add(center)
        if self.mode4_entry_tile is not None:
            self.known_free.add(self.mode4_entry_tile)
        tracked_unknowns = set(self.belief_unknown_tiles)
        if self.belief_worlds:
            self.belief_worlds = algorithms.update_belief_worlds(
                self.belief_worlds,
                observed_free & tracked_unknowns,
                observed_blocked & tracked_unknowns)
        self.belief_unknown_tiles -= observed_free
        self.belief_unknown_tiles -= observed_blocked
        self.belief_worlds_after_observation = len(self.belief_worlds)
        self.belief_inconsistent = not self.belief_worlds
        if self.belief_inconsistent:
            self.message = "Inconsistent belief: no possible worlds remain"
            self._rebuild_mode4_belief_from_observations()
        else:
            self._last_risk_map = algorithms.belief_risk_map(
                self.belief_worlds, self.belief_unknown_tiles)
        self._update_belief_state()


    def _and_or_seen_targets_now(self):
        return frozenset(
            tile for tile in self.farm_tiles
            if tile in self.explored_tiles
            and tile not in self.done_tiles
            and tile not in self.discovered_blocked
            and tile not in self.enemy_done_tiles
        )


    def _and_or_planning_targets(self, start, seen_targets):
        ordered = sorted(
            seen_targets,
            key=lambda tile: (
                self._mode2_h_parts(start, tile)[3],
                self._mode2_condition_h(tile),
                self._heuristic(start, tile),
                tile,
            )
        )
        return frozenset(ordered)


    def _and_or_target_valid(self, tile):
        return (
            tile in self.farm_tiles
            and tile not in self.done_tiles
            and tile not in self.discovered_blocked
            and tile not in self.enemy_done_tiles
        )


    def _online_plan(self, start, goal):
        """Online Search: chá»‰ biáº¿t váº­t cáº£n Ä‘Ă£ khĂ¡m phĂ¡, phĂ¡t hiá»‡n thĂªm khi Ä‘i."""
        self.mode4_phase = "REPLAN" if self.replan_count else "EXPLORE"
        if goal in self.discovered_blocked:
            self._update_belief_state()
            return []
        if start == goal:
            explored_in_area = self.explored_tiles & (
                set(self.farm_tiles) | {self.spawn_tile})
            self.astar_current_fgh = (
                self.mode4_completed_g, self.mode4_completed_g, 0)
            self.stats = {
                "Algorithm": self.algorithm_name,
                "Current target": f"{goal}",
                "Path length": 0,
                "Nodes explored": 1,
                "f(n)": f"{self.mode4_completed_g}",
                "g(n)": f"{self.mode4_completed_g}",
                "h(n)": "0",
                "g leg": 0,
                "g completed": self.mode4_completed_g,
                "Replanned": f"{self.replan_count} lan",
                "Explored tiles": f"{len(explored_in_area)}",
                "Blocks found": (
                    f"{len(self.discovered_blocked)}/{len(self.hidden_blocked)}"),
                "Online policy": "Already at selected target",
            }
            if self.algorithm_name in (
                    "Online BFS", "Belief-State BFS",
                    "Belief BFS", "Belief A*"):
                self.stats["Queue max"] = 1
            return []

        self._update_belief_state()
        if self.algorithm_name in (
                "Belief-State BFS", "Belief BFS",
                "Belief A*", "AND-OR Search"):
            blocked = self._blocked_tiles()
            blocked.discard(start)
            blocked.discard(goal)
            if self.algorithm_name == "AND-OR Search":
                path, explored, fgh, belief_stats = (
                    algorithms.astar_4dir_f_update(
                        start, goal, blocked, self._neighbors,
                        self._search_heuristic, self.counter,
                        self._step_cost))
                self.astar_current_fgh = fgh
                _, g_leg, h_val = fgh
                g_total = self.mode4_completed_g + g_leg
                f_val = g_total + h_val
                belief_stats.update({
                    "Algorithm": "AND-OR Search",
                    "f(n)": f"{f_val}",
                    "g(n)": f"{g_total}",
                    "h(n)": f"{h_val}",
                    "g leg": g_leg,
                    "g completed": self.mode4_completed_g,
                    "Search style": "OR picks A* node; AND samples outcome",
                    "Target rule": "same h(n) priority as Mode 2 A*",
                    "AND outcomes": "70 correct / 10 / 10 / 10 drift",
                    "Policy": "replan after actual outcome",
                })
            else:
                self.belief_unknown_tiles = self._mode4_unknown_tiles()
                risk_map = algorithms.belief_risk_map(
                    self.belief_worlds, self.belief_unknown_tiles)
                self._last_risk_map = dict(risk_map)
                path, explored, fgh, belief_stats = self._belief_bfs_plan(
                    start, goal, blocked)
                self.astar_current_fgh = fgh
                if path:
                    path = path[:1]
                    belief_stats["Next node"] = f"{path[0]}"
                    belief_stats["Robot execution"] = (
                        "one real robot; execute first BFS tile only")
                else:
                    belief_stats["Robot execution"] = "no action"
            self.astar_explored = explored
        else:
            blocked = self._blocked_tiles()
            blocked.discard(start)
            blocked.discard(goal)
            path, explored, fgh, belief_stats = algorithms.astar_4dir_f_update(
                start, goal, blocked, self._neighbors,
                self._search_heuristic, self.counter, self._step_cost)
            self.astar_explored = set(explored)
            self.astar_current_fgh = fgh
            belief_stats["Algorithm"] = self.algorithm_name
            if self.algorithm_name == "Online A*":
                _, g_leg, h_val = fgh
                g_total = self.mode4_completed_g + g_leg
                f_val = g_total + h_val
                belief_stats.update({
                    "f(n)": f"{f_val}",
                    "g(n)": f"{g_total}",
                    "h(n)": f"{h_val}",
                    "g leg": g_leg,
                    "g completed": self.mode4_completed_g,
                })

        if "g leg" not in belief_stats:
            belief_stats["g leg"] = len(path)
        if "g completed" not in belief_stats:
            belief_stats["g completed"] = self.mode4_completed_g
        belief_stats["Path length"] = len(path)
        belief_stats["Current target"] = f"{goal}"
        belief_stats.setdefault("Current node", f"{start}")
        belief_stats.setdefault("Next node", f"{path[0] if path else start}")
        if self.algorithm_name == "Online A*":
            belief_stats.setdefault(
                "Update rule", "only update when f_new < f_old")
            belief_stats.setdefault(
                "Neighbor rule", "push only 4 adjacent neighbors")

        explored_in_area = self.explored_tiles & (
            set(self.farm_tiles) | {self.spawn_tile})
        total_explored = len(explored_in_area)
        total_map = len(self.farm_tiles) + 1
        base_stats = {
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
        priority_keys = (
            "Algorithm", "Current target", "f(n)", "g(n)", "h(n)",
            "Current node", "Next node", "Path length", "Queue max",
            "Frontier max", "Belief states", "Possible worlds",
            "g leg", "g completed", "Search style", "Target rule",
            "Update rule", "Neighbor rule",
        )
        ordered_stats = {}
        for key in priority_keys:
            if key in belief_stats:
                ordered_stats[key] = belief_stats[key]
            elif key in base_stats:
                ordered_stats[key] = base_stats[key]
        for key, val in base_stats.items():
            ordered_stats.setdefault(key, val)
        for key, val in belief_stats.items():
            ordered_stats.setdefault(key, val)
        self.stats = ordered_stats
        if self.algorithm_name == "Online A*":
            self.stats["Online policy"] = "Plan known map; patch on discovery"
        elif self.algorithm_name == "Online BFS":
            self.stats["Online policy"] = "BFS on known map; replan on discovery"
        elif self.algorithm_name in ("Belief-State BFS", "Belief BFS", "Belief A*"):
            self.stats["Online policy"] = "Belief-state BFS over hidden blocks"
        else:
            self.stats["Online policy"] = "OR uses A*; AND samples 70/10/10/10"
            self.stats["Backtracking"] = "replan from actual outcome"
        self._update_belief_state()
        return path


    def _sample_and_or_outcome(self, current, intended):
        dx = intended[0] - current[0]
        dy = intended[1] - current[1]
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
        intended_direction = (dx, dy)
        if intended_direction not in directions:
            return current, 100, "Invalid intended step; stay"
        outcomes = [intended_direction]
        outcomes.extend(
            direction for direction in directions
            if direction != intended_direction)
        probabilities = (70, 10, 10, 10)
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
        return actual, probability, "Wrong direction; replan from actual tile"


    def _and_or_reset_navigation(self):
        """Clear transient navigation state for the current stochastic step."""
        self.and_or_visited.clear()


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
        latest_seen_targets = self._and_or_planning_targets(
            actual, self._and_or_seen_targets_now())
        seen_targets_changed = latest_seen_targets != self.and_or_seen_targets
        if seen_targets_changed:
            self.and_or_seen_targets = latest_seen_targets

        # Correct move: keep aiming at the selected target. Do not care for
        # untreated crops passed along the route.
        if actual == intended:
            self.and_or_visited.add(actual)
            if self.path:
                self.path.pop(0)
            return

        # Drift move: only then evaluate the tile the robot actually reached.
        if (actual in self.farm_tiles
                and actual not in self.done_tiles
                and actual not in self.discovered_blocked
                and actual not in self.enemy_done_tiles):
            previous_target = self.current_target
            if (previous_target is not None
                    and previous_target != actual
                    and self._and_or_target_valid(previous_target)):
                self.and_or_resume_target = previous_target
            self.current_target = actual
            self.path = []
            self._and_or_reset_navigation()
            self.state = self._state_after_arrival(actual)
            self.stats.update({
                "Observed tile": f"{actual}",
                "Observed action": "drift found untreated tile; handle now",
                "Previous target": f"{previous_target}",
                "Resume target": f"{self.and_or_resume_target}",
            })
            self.message = f"AND-OR xet o {actual}: xu ly target da thay"
            self.wait_time = 0.18
            return

        # ---- BLOCKED: stayed in place ----
        if label.startswith("Blocked"):
            self.path = []
            self.current_target = None
            self.state = "CHOOSE_TARGET"
            self.replan_count += 1
            self.mode4_phase = "REPLAN"
            self._clear_full_garden_plan()
            self._and_or_reset_navigation()
            self.stats.update({
                "Contingency": f"{label}; choose new target",
            })
            self.message = (
                f"AND-OR: {label}, "
                f"chon lai muc tieu (lan {self.replan_count})")
            self._update_belief_state()
            self.wait_time = 0.25
            return

        # ---- DRIFT: ended up on a different tile ----

        self.and_or_visited.add(actual)
        self.path = []
        self.replan_count += 1
        self.mode4_phase = "REPLAN"
        self._clear_full_garden_plan()
        self._and_or_reset_navigation()

        if actual in self.done_tiles:
            self.and_or_resume_target = None
            self.current_target = None
            self.state = "CHOOSE_TARGET"
            self.stats.update({
                "Contingency": "drifted to already handled tile",
                "Next OR step": "choose a new target",
            })
            self.message = (
                f"AND-OR drift {probability}% vao o da chua {actual}; "
                f"chon lai muc tieu (lan {self.replan_count})")
            self.wait_time = 0.18
            return

        if self._and_or_target_valid(self.current_target):
            self.and_or_resume_target = self.current_target
            self.current_target = None
            self.state = "CHOOSE_TARGET"
            self.stats.update({
                "Contingency": "drifted to free tile",
                "Resume target": f"{self.and_or_resume_target}",
                "Next OR step": "A* replan from actual tile to target",
            })
            self.message = (
                f"AND-OR drift {probability}%: {label}; "
                f"di lai toi target cu (lan {self.replan_count})")
            self.wait_time = 0.15
            return

        self.and_or_resume_target = None
        self.current_target = None
        self.state = "CHOOSE_TARGET"
        self.stats.update({
            "Contingency": "drifted and old target is no longer valid",
            "Next OR step": "choose a new target",
        })
        self.message = (
            f"AND-OR drift {probability}%: {label}; "
            f"chon lai muc tieu (lan {self.replan_count})")
        self.wait_time = 0.18

    # -------------------------------------------------------------- MODE 5


    # ------------------------------------------------------------------
    # mode 5 CSP
    # ------------------------------------------------------------------

    def _solve_csp_crop_plan(self):
        """Solve Mode 5 CSP with shared adjacency constraints for 4 crops."""
        assignment, self.csp_steps, self.stats = algorithms.solve_csp_crop_plan(
            self.farm_tiles, algorithm=self.algorithm_name
        )
        return assignment


    # --- CSP animation phases ---


    def _begin_csp_analyze(self):
        """Phase 1: Robot dung yen, highlight cac cap rang buoc ke nhau."""
        self.csp_phase = "analyze"
        self.csp_analyze_timer = 0.0
        self.csp_display = {}
        self.csp_domains = {}
        self.csp_flash_timers = {}
        pairs = []
        farm_set = set(self.farm_tiles)
        seen = set()
        for tile in self.farm_tiles:
            x, y = tile
            for nx, ny in ((x + 1, y), (x, y + 1)):
                nb = (nx, ny)
                if nb in farm_set and (tile, nb) not in seen:
                    pairs.append((tile, nb))
                    seen.add((tile, nb))
        self.csp_analyze_pairs = pairs
        self.csp_analyze_pair_index = 0
        self.csp_analyze_pair_timer = 0.0
        self.player.direction.update(0, 0)
        self.message = (
            f"AGRI-1 dang phan tich rang buoc ({len(pairs)} cap o ke)..."
        )
        self.stats["CSP phase"] = "Phan tich rang buoc"
        self.stats["Rang buoc"] = f"{len(pairs)} cap ke nhau"
        self.stats["Constraint"] = (
            "no same; no corn-tomato; no wheat-carrot")


    def _update_csp_analyze(self, dt):
        """Phase 1 update: highlight cap rang buoc lan luot roi sang replay."""
        self.player.direction.update(0, 0)
        self.csp_analyze_timer += dt
        self.csp_analyze_pair_timer += dt
        if self.csp_analyze_pair_timer >= 0.09:
            self.csp_analyze_pair_timer = 0.0
            self.csp_analyze_pair_index += 1
        progress = min(1.0, self.csp_analyze_timer / self.csp_analyze_duration)
        self.stats["Tien do phan tich"] = f"{int(progress * 100)}%"
        if self.csp_analyze_timer >= self.csp_analyze_duration:
            self._begin_csp_replay()


    def _begin_csp_replay(self):
        """Phase 2: Tao chuoi quyet dinh CSP va thuc hien tung buoc."""
        # Solve at run time, after the constraint-analysis phase, instead
        # of calculating a hidden finished plan as soon as mode 5 opens.
        self.seed_plan = self._solve_csp_crop_plan()
        self.csp_phase = "replay"
        self.csp_replay_index = 0
        self.csp_replay_timer = 0.0
        self.csp_display = {}
        if self.algorithm_name in ("Fwd Check", "AC-3"):
            self.csp_domains = {
                tile: list(algorithms.CSP_CROPS)
                for tile in self.farm_tiles
            }
        else:
            self.csp_domains = {}
        self.csp_flash_timers = {}
        total = len(self.csp_steps)
        bt = self.stats.get("Backtracks", 0)
        if self.algorithm_name == "Min Conflict":
            self.message = (
                f"Min-Conflicts: Gieo ngau nhien toan bo ruong, "
                f"sau do sua {bt} xung dot..."
            )
        elif self.algorithm_name == "Backtrack":
            self.message = (
                f"Backtrack: di tuan tu tung o, thu crop ngau nhien "
                f"({total} buoc, {bt} backtrack)..."
            )
        elif self.algorithm_name in ("Fwd Check", "AC-3"):
            self.message = (
                f"{self.algorithm_name}: thu gan va cat domain "
                f"({total} buoc, {bt} backtrack)..."
            )
        else:
            self.message = (
                f"{self.algorithm_name}: bat dau thu rang buoc "
                f"({total} buoc, {bt} backtrack)..."
            )
        self.stats["CSP phase"] = "Tim va thu truc tiep"
        self.stats["Tong buoc"] = total


    def _csp_set_speech(self, text, duration=1.2, tile=None):
        self.csp_speech = text
        self.csp_speech_timer = duration  # giu lai de tuong thich, nhung khong dung de xoa
        self.csp_speech_tile = tile if tile is not None else self.csp_walk_target


    def _csp_start_moving(self, path, act_duration=0.25):
        """Bat dau robot di den tile moi: xoa speech bubble cu, set walk state."""
        self.csp_speech = ""           # xoa bubble ngay khi bat dau di
        self.csp_speech_tile = None
        self.csp_walk_path = path
        self.csp_walk_state = "moving"
        self.csp_walk_act_duration = act_duration


    def _csp_queue_speech(self, text, duration, will_move):
        """Fix 2: chi hien speech bubble ("suy nghi") khi robot DA DEN tile.

        Neu robot can di (will_move=True), hoan lai text/duration vao
        csp_speech_pending - se duoc _update_csp_replay bat len ngay luc
        csp_walk_path rong (vua toi dich). Neu robot da dung san tai tile
        (will_move=False) thi khong can doi, hien luon.
        """
        if will_move:
            self.csp_speech_pending = (text, duration)
        else:
            self.csp_speech_pending = None
            self._csp_set_speech(text, duration)


    def _csp_find_conflict_neighbors(self, tile, crop):
        """Tra ve cac o ke dang vi pham rang buoc CSP voi crop nay."""
        x, y = tile
        conflicts = set()
        for nx, ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
            nb = (nx, ny)
            if (
                nb in self.csp_assigned
                and not algorithms.csp_crop_pair_valid(crop, self.csp_assigned[nb])
            ):
                conflicts.add(nb)
        return conflicts


    def _csp_all_conflicting_tiles(self):
        """Quet toan bo csp_assigned hien tai, tra ve tap TAT CA tile dang
        vi pham rang buoc voi mot hang xom. Dung de
        rebuild csp_conflict_tiles chinh xac sau moi reassign, tranh truong
        hop mot tile bi to do vi gia tri CU cua hang xom nhung khong duoc
        xoa khi hang xom doi sang gia tri khac."""
        bad = set()
        for tile, crop in self.csp_assigned.items():
            if self._csp_find_conflict_neighbors(tile, crop):
                bad.add(tile)
        return bad


    def _update_csp_flash_timers(self, dt):
        """Keep temporary red backtrack flashes moving even while the robot walks."""
        for tile in list(self.csp_flash_timers):
            self.csp_flash_timers[tile] -= dt
            if self.csp_flash_timers[tile] <= 0:
                del self.csp_flash_timers[tile]
                if tile in self.csp_display and self.csp_display[tile][1] == "backtrack":
                    self.csp_display.pop(tile, None)


    def _update_csp_replay(self, dt):
        """Phase 2: Robot di den tung o, thu crop, highlight xung dot, backtrack."""
        # Speech bubble KHONG dung timer de xoa - chi xoa khi robot bat dau move sang tile moi
        # (xem cho csp_walk_state chuyen sang "moving" ben duoi)
        self._update_csp_flash_timers(dt)

        # Cap nhat conflict highlight timer
        if self.csp_conflict_timer > 0:
            self.csp_conflict_timer = max(0, self.csp_conflict_timer - dt)
            if self.csp_conflict_timer <= 0:
                self.csp_conflict_tiles.clear()

        # Cap nhat backtrack flash
        if self.csp_backtrack_flash > 0:
            self.csp_backtrack_flash = max(0, self.csp_backtrack_flash - dt)

        # --- Robot dang di den tile ---
        if self.csp_walk_state == "moving":
            if not self.csp_walk_path:
                self.csp_walk_state = "acting"
                self.csp_walk_act_timer = self.csp_walk_act_duration
                # Robot da den noi: gio moi hien "suy nghi" (speech bubble)
                # da duoc hen tu luc doc step, tranh hien som khi con dang di.
                if self.csp_speech_pending:
                    text, duration = self.csp_speech_pending
                    self._csp_set_speech(text, duration, tile=self.csp_walk_target)
                    self.csp_speech_pending = None
                return
            next_tile = self.csp_walk_path[0]
            reached = self._set_player_direction_to(self._tile_center(next_tile))
            if reached:
                self.csp_walk_path.pop(0)
            return

        # --- Robot dang "lam viec" tai tile (dung lai ngan) ---
        if self.csp_walk_state == "acting":
            self.player.direction.update(0, 0)
            self.csp_walk_act_timer -= dt
            if self.csp_walk_act_timer > 0:
                return
            self.csp_walk_state = "idle"

        # --- Robot idle: doc buoc tiep theo ---
        if self.csp_walk_state != "idle":
            return

        # Kiem tra xem het buoc chua
        if self.csp_replay_index >= len(self.csp_steps):
            # Force-clear timers khi da het buoc - tranh bi ket mai mai
            if self.csp_conflict_timer <= 0:
                self.csp_conflict_tiles.clear()
            if self.csp_backtrack_flash <= 0 and not self.csp_conflict_tiles:
                self._begin_csp_ready()
            return

        # Doc buoc ke tiep - gom nhom cac buoc "try" lien tiep cung tile
        event, var, value = self.csp_steps[self.csp_replay_index]
        self.csp_replay_index += 1

        if var is None:
            # Min-Conflicts "init": hien toan bo map ngau nhien NGAY LAP TUC,
            # sau do highlight DO tat ca o xung dot truoc khi robot di chua.
            if event == "init" and isinstance(value, dict):
                # 1. Dien day du tat ca o voi crop ngau nhien
                for init_tile, init_crop in value.items():
                    self.csp_assigned[init_tile] = init_crop
                    self.csp_display[init_tile] = (init_crop, "assign")

                # 2. Tim tat ca cac o dang xung dot (vi pham rang buoc ke nhau)
                conflict_set = set()
                farm_set = set(self.farm_tiles)
                for tile, crop in self.csp_assigned.items():
                    tx, ty = tile
                    for nx, ny in ((tx-1,ty),(tx+1,ty),(tx,ty-1),(tx,ty+1)):
                        nb = (nx, ny)
                        if nb in farm_set and nb in self.csp_assigned:
                            if not algorithms.csp_crop_pair_valid(crop, self.csp_assigned[nb]):
                                conflict_set.add(tile)
                                conflict_set.add(nb)

                # 3. Highlight do tat ca o xung dot, doi robot chua
                if conflict_set:
                    self.csp_conflict_tiles = conflict_set
                    self.csp_conflict_timer = 999.0  # giu mai den khi het xung dot
                    n_conflicts = len(conflict_set)
                    self.message = (
                        f"Min-Conflicts: Ruong da gieo ngau nhien! "
                        f"Co {n_conflicts} o xung dot (do) -> Robot bat dau chua..."
                    )
                    self._csp_set_speech(f"{n_conflicts} xung dot!", 1.8)
                else:
                    self.message = "Min-Conflicts: May man! Khong co xung dot ngay tu dau."
                    self._csp_set_speech("Xong luon!", 1.5)

                # 4. Dung lai 1.5 giay de nguoi dung thay full map + highlight do
                self.csp_walk_state = "acting"
                self.csp_walk_act_duration = 1.5
                self.csp_walk_act_timer = 1.5
                self.stats["CSP phase"] = "Khoi tao ngau nhien"
                self.stats["O xung dot ban dau"] = len(conflict_set)
                self.stats["Tong o"] = len(self.csp_assigned)
            return

        if event == "domain":
            self.csp_domains[var] = list(value)
            self.stats["CSP phase"] = "Cap nhat domain"
            self.stats["Buoc"] = f"{self.csp_replay_index}/{len(self.csp_steps)}"
            self.stats["Domain update"] = f"{var}: {','.join(value)}"
            return

        self.csp_current_var = var
        self.csp_current_value = value
        self.csp_current_event = event

        crop_names = {"corn": "Ngo", "tomato": "Ca chua", "wheat": "Lua mi", "carrot": "Ca rot"}
        crop_vn = crop_names.get(value, value)

        current_tile = self._world_to_tile(self.player.rect.center)

        if event == "think":
            self.csp_display[var] = (value, "try")
            self.csp_walk_target = var
            if var != current_tile:
                blocked = self._blocked_tiles()
                blocked.discard(current_tile)
                blocked.discard(var)
                path, _, _, _ = algorithms.find_path_by_algorithm(
                    "A*", current_tile, var, blocked,
                    self._neighbors, self._search_heuristic, self.counter, self._step_cost)
                if path:
                    self._csp_start_moving(path, act_duration=0.18)
            self._csp_queue_speech(
                f"Suy nghi... {crop_vn}?", 0.25,
                will_move=(self.csp_walk_state == "moving"))
            self.message = f"Dang can nhac cay cho o {var}..."
            if self.csp_walk_state != "moving":
                self.csp_walk_state = "acting"
                self.csp_walk_act_timer = 0.18

        elif event == "try":
            # Trong thu ung vien ngau nhien TRUOC, sau do moi kiem tra
            # rang buoc. Neu trung hang xom, step backtrack ke tiep se nhổ.
            self.csp_display[var] = (value, "try")
            self.csp_assigned[var] = value
            conflicts = self._csp_find_conflict_neighbors(var, value)
            if conflicts:
                self.csp_conflict_tiles = set(conflicts)
                self.csp_conflict_tiles.add(var)
                self.csp_conflict_timer = 999.0
            else:
                self.csp_conflict_tiles.clear()
                self.csp_conflict_timer = 0.0
            # Neu robot chua o tile nay, di den do
            self.csp_walk_target = var
            if var != current_tile:
                blocked = self._blocked_tiles()
                blocked.discard(current_tile)
                blocked.discard(var)
                path, _, _, _ = algorithms.find_path_by_algorithm(
                    "A*", current_tile, var, blocked,
                    self._neighbors, self._search_heuristic, self.counter, self._step_cost)
                if path:
                    self._csp_start_moving(path, act_duration=0.25)
            self.csp_walk_act_duration = 0.4
            self.csp_walk_act_timer = self.csp_walk_act_duration
            if self.csp_walk_state != "moving":
                self.csp_walk_state = "acting"
            self._csp_queue_speech(f"Trong thu {crop_vn}!", 0.8,
                                    will_move=(self.csp_walk_state == "moving"))
            self.message = f"Trong ngau nhien {crop_vn} tai o {var}, dang kiem tra..."

        elif event == "assign":
            self.csp_display[var] = (value, "assign")
            self.csp_assigned[var] = value
            self.csp_conflict_tiles.clear()
            self.csp_conflict_timer = 0.0
            self.message = f"Gan {crop_vn} tai o {var} ({len(self.csp_assigned)}/{len(self.farm_tiles)})"
            # Di den tile neu chua o do
            if var != current_tile:
                blocked = self._blocked_tiles()
                blocked.discard(current_tile)
                blocked.discard(var)
                path, _, _, _ = algorithms.find_path_by_algorithm(
                    "A*", current_tile, var, blocked,
                    self._neighbors, self._search_heuristic, self.counter, self._step_cost)
                if path:
                    self._csp_start_moving(path, act_duration=0.35)
            self.csp_walk_act_duration = 0.35
            self.csp_walk_act_timer = self.csp_walk_act_duration
            if self.csp_walk_state != "moving":
                self.csp_walk_state = "acting"
            self._csp_queue_speech(f"Trong {crop_vn}! OK", 1.0,
                                    will_move=(self.csp_walk_state == "moving"))

        elif event == "backtrack":
            self.csp_backtracks_live += 1
            self.csp_backtrack_flash = 0.6
            # Xoa assignment cua tile nay
            self.csp_assigned.pop(var, None)
            self.csp_display[var] = (value, "backtrack")
            self.csp_flash_timers[var] = 0.6
            # Tim cac o ke de highlight do
            conflicts = self._csp_find_conflict_neighbors(var, value)
            if conflicts:
                self.csp_conflict_tiles = set(conflicts)
                self.csp_conflict_tiles.add(var)
                self.csp_conflict_timer = 0.7
            self._csp_queue_speech(f"Sai! Quay lui...", 1.2, will_move=False)
            self.message = (
                f"BACKTRACK #{self.csp_backtracks_live}: "
                f"{crop_vn} tai {var} vi pham rang buoc!"
            )
            self.csp_walk_act_duration = 0.55
            self.csp_walk_act_timer = self.csp_walk_act_duration
            self.csp_walk_state = "acting"
            # Player status "confused"
            self.player.status = "down_idle"

        elif event == "conflict":
            # Min-Conflicts: o nay dang xung dot (gia tri value la crop CU
            # dang vi pham rang buoc). Robot se di toi day de sua, nhung
            # o nay PHAI giu mau do cho toi luc robot thuc su sua xong
            # (event "reassign" ke tiep) - khong discard som o day.
            self.csp_backtracks_live += 1
            self.csp_backtrack_flash = 0.4
            self.csp_display[var] = (value, "backtrack")
            self.csp_flash_timers[var] = 0.5
            # Dung ham rebuild toan bo (giong reassign) de dam bao nhat
            # quan: csp_assigned[var] van la gia tri CU tai day, nen tap
            # tra ve chinh la {var} + hang xom cung crop CU cua no.
            self.csp_conflict_tiles = self._csp_all_conflicting_tiles()
            self.csp_conflict_timer = 999.0  # giu highlight do cho den khi het xung dot
            # Di den tile neu chua o do
            if var != current_tile:
                blocked = self._blocked_tiles()
                blocked.discard(current_tile)
                blocked.discard(var)
                path, _, _, _ = algorithms.find_path_by_algorithm(
                    "A*", current_tile, var, blocked,
                    self._neighbors, self._search_heuristic, self.counter, self._step_cost)
                if path:
                    self._csp_start_moving(path, act_duration=0.20)
            self._csp_queue_speech(f"Xung dot! {crop_vn}...", 0.7,
                                    will_move=(self.csp_walk_state == "moving"))
            self.message = (
                f"Min-Conflict #{self.csp_backtracks_live}: "
                f"o {var} xung dot voi hang xom!"
            )

        elif event == "reassign":
            # Min-Conflicts: gán lại giá trị mới để giải quyết xung đột.
            # Day la luc o nay THUC SU duoc sua -> chuyen tu do sang xanh.
            self.csp_display[var] = (value, "assign")
            self.csp_assigned[var] = value
            # Tinh lai TOAN BO tap xung dot tu csp_assigned hien tai, thay vi
            # chi xoa/them tung tile rieng le. Ly do: khi var doi gia tri,
            # mot hang xom A da bi to do truoc do (vi tung cung mau VOI GIA
            # TRI CU cua var) co the het xung dot - nhung A khong nam trong
            # "remaining_conflicts" cua GIA TRI MOI nen se bi ket do mai
            # mai neu chi xoa/them theo kieu increment. Rebuild toan bo moi
            # dam bao chinh xac 100%.
            self.csp_conflict_tiles = self._csp_all_conflicting_tiles()
            self.csp_conflict_timer = 999.0 if self.csp_conflict_tiles else 0.0
            self.message = (
                f"Min-Conflict: doi o {var} -> {crop_vn} "
                f"({len(self.csp_assigned)}/{len(self.farm_tiles)} da xu ly)"
            )
            # Di den tile neu chua o do
            if var != current_tile:
                blocked = self._blocked_tiles()
                blocked.discard(current_tile)
                blocked.discard(var)
                path, _, _, _ = algorithms.find_path_by_algorithm(
                    "A*", current_tile, var, blocked,
                    self._neighbors, self._search_heuristic, self.counter, self._step_cost)
                if path:
                    self._csp_start_moving(path, act_duration=0.30)
            self.csp_walk_act_duration = 0.30
            self.csp_walk_act_timer = self.csp_walk_act_duration
            if self.csp_walk_state != "moving":
                self.csp_walk_state = "acting"
            self._csp_queue_speech(f"Doi sang {crop_vn}!", 0.9,
                                    will_move=(self.csp_walk_state == "moving"))

        # Cap nhat stats HUD realtime
        idx = self.csp_replay_index
        total = len(self.csp_steps)
        self.stats["CSP phase"] = "Thuc hien CSP"
        self.stats["Buoc"] = f"{idx}/{total}"
        self.stats["Dang thu"] = f"{crop_names.get(value, value)} tai {var}"
        self.stats["Da gan"] = f"{len(self.csp_assigned)}/{len(self.farm_tiles)}"
        if self.algorithm_name == "Min Conflict":
            self.stats["Conflicts fixed"] = self.csp_backtracks_live
        else:
            self.stats["Backtracks"] = self.csp_backtracks_live


    def _begin_csp_ready(self):
        """Phase 3: CSP replay xong la hoan tat Mode 5, khong di lai."""
        self.csp_phase = "ready"
        self.csp_display = {}
        self.csp_domains = {}
        self.csp_flash_timers = {}
        self.csp_current_var = None
        self.csp_current_event = None
        self.csp_speech = ""
        self.csp_speech_pending = None
        # Tat het highlight do - khong con xung dot
        self.csp_conflict_tiles.clear()
        self.csp_conflict_timer = 0.0
        # Dong bo csp_assigned tu seed_plan (ket qua chinh xac tu CSP solver)
        # Tranh truong hop Min-Conflicts replay khong log day du tat ca bien
        if self.seed_plan:
            self.csp_assigned = dict(self.seed_plan)
        resolved_tiles = set(self.csp_assigned)
        self.done_tiles.update(resolved_tiles)
        self.csp_seeded_tiles.update(resolved_tiles)
        self.current_target = None
        self.path = []
        self._clear_full_garden_plan()
        self.state = "DONE"
        self.player.direction.update(0, 0)
        assigned = len(resolved_tiles)
        bt = self.csp_backtracks_live
        if self.algorithm_name == "Min Conflict":
            self.message = (
                f"Min-Conflicts hoan tat: {assigned} o, sua {bt} xung dot."
            )
        else:
            self.message = (
                f"Mode 5 hoan tat: {assigned} o, {bt} backtrack."
            )
        self.stats["CSP phase"] = "Hoan tat"
        self.stats["Da giai quyet"] = f"{assigned}/{len(self.farm_tiles)}"
        self.wait_time = 0.0

    # -------------------------------------------------------------- MODE 6

    # ------------------------------------------------------------------
    # mode 6 adversarial search
    # ------------------------------------------------------------------

    def _enemy_retreat_tile(self, damaged_tile):
        if self.enemy_spawn_tile is not None:
            return self.enemy_spawn_tile
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


    def _update_mode6_fix(self, tile):
        """Fix crow damage in exactly the same time the crow needs to destroy."""
        now = pygame.time.get_ticks()
        if self.mode6_fix_target != tile or not self.mode6_fix_until:
            self.mode6_fix_target = tile
            self.mode6_fix_until = (
                now + int(self.MODE6_ACTION_TIME * 1000))
            self.player.selected_tool = "water"
            self.player.status = "down_water"
            self.mode6_phase = "MODE6_RESOLVE"
            self.message = (
                f"AGRI-1 dang sua cay bi qua pha {tile}: "
                f"{self.MODE6_ACTION_TIME:.1f}s")
            return
        if now < self.mode6_fix_until:
            remaining = (self.mode6_fix_until - now) / 1000
            self.message = f"AGRI-1 dang sua {tile}: {remaining:.1f}s"
            return

        self.done_tiles.add(tile)
        self._advance_full_garden_plan(tile)
        self.current_target = None
        self.path = []
        self.mode6_fix_until = 0
        self.mode6_fix_target = None
        self.state = "DONE" if self.stop_after_current_target else "CHOOSE_TARGET"
        self.stop_after_current_target = False
        self.mode6_phase = "MODE6_RESOLVE"
        self.message = f"AGRI-1 bao ve cay {tile}"
        self._update_mode6_step_hud(
            phase=self.mode6_phase, current=tile, next_tile=None)

    def _update_mode6_step_hud(self, phase=None, current=None, next_tile=None):
        """Refresh algorithm-specific Mode-6 information every live step."""
        if self.mode != 6:
            return
        if phase:
            self.mode6_phase = phase

        details = self.mode6_tree_details
        for tile, profile in details.get("crop_profiles", {}).items():
            profile["state"] = (
                "protected" if tile in self.done_tiles
                else "destroyed" if tile in self.enemy_done_tiles
                else "remaining")
        robot_action = details.get("robot_action", self.mode6_robot_action)
        enemy_action = details.get("enemy_action", self.mode6_enemy_action)
        robot_target = details.get("robot_target", self.mode6_robot_target)
        enemy_target = details.get(
            "enemy_priority_target",
            details.get("enemy_target", self.mode6_enemy_target))
        robot_path_length = details.get(
            "robot_path_length", len(self.mode6_robot_path))
        enemy_path_length = details.get(
            "enemy_path_length", len(self.mode6_enemy_path))
        robot_frontier = details.get("robot_frontier", [])
        enemy_frontier = details.get("enemy_frontier", [])
        depth = details.get("depth", algorithms.MODE6_SEARCH_DEPTH)
        protected_score, destroyed_score = algorithms.mode6_state_scores(
            self.done_tiles, self.enemy_done_tiles,
            self.mode6_crop_profiles)
        step_text = (
            f"{self.mode6_move_done}/{self.mode6_move_total}"
            if self.mode6_move_total else "-"
        )
        enemy_step_text = (
            f"{self.mode6_enemy_move_done}/{self.mode6_enemy_move_total}"
            if self.mode6_enemy_move_total else "-"
        )
        robot_current = self._world_to_tile(self.player.rect.center)
        robot_next = self.mode6_robot_step_target
        enemy_current = (
            (int(round(self.enemy_tile[0])), int(round(self.enemy_tile[1])))
            if self.enemy_tile is not None else None)
        enemy_next = self.mode6_enemy_step_target
        remaining_trees = len([
            tile for tile in self.farm_tiles
            if tile not in self.done_tiles
            and tile not in self.enemy_done_tiles
            and tile not in self.discovered_blocked])
        live = {
            "Algorithm": self.algorithm_name,
            "Robot Algorithm": self.algorithm_name,
            "Enemy Policy": (
                "Random Chance" if self.algorithm_name == "Expectimax"
                else "Full MIN legal actions"),
            "Phase": self.mode6_phase,
            "Turn": self.mode6_turn,
            "Search depth": depth,
            "Lookahead Rounds": depth // 2,
            "Game tree": "Step-by-step adversarial search",
            "Robot movement": "one action: UP/DOWN/LEFT/RIGHT/STAY",
            "Crow movement": (
                "disabled; random chance damages trees"
                if self.algorithm_name == "Expectimax"
                else "one action: UP/DOWN/LEFT/RIGHT/STAY"),
            "MAX action": f"{robot_action}",
            "MIN action": f"{enemy_action}",
            "Robot Action": f"{robot_action}",
            "Enemy Action": f"{enemy_action}",
            "Robot target": f"{robot_target}",
            "Enemy target": f"{enemy_target}",
            "Robot path length": robot_path_length,
            "Enemy path length": enemy_path_length,
            "MAX frontier": ", ".join(robot_frontier),
            "MIN frontier": ", ".join(enemy_frontier),
            "Evaluation score": f"{self.minimax_value:.1f}",
            "Protected score": protected_score,
            "Destroyed score": destroyed_score,
            "Remaining trees": remaining_trees,
            "AGRI step": step_text,
            "Crow step": enemy_step_text,
            "Resolve timer": (
                f"{self.mode6_resolve_elapsed:.1f}/"
                f"{self.MODE6_RESOLVE_TIME:.1f}s"
                if (self.mode6_phase == "RESOLVE_STEP"
                    and self.MODE6_RESOLVE_TIME > 0) else "-"),
            "Robot current tile": f"{robot_current}",
            "Robot next tile": f"{robot_next}",
            "Enemy current tile": f"{enemy_current}",
            "Enemy next tile": f"{enemy_next}",
            "Threatened tile": (
                f"{self.mode6_enemy_threat_tile} "
                f"({self.mode6_enemy_threat_turns}/2)"
                if self.mode6_enemy_threat_tile is not None else "-"),
            "Threat turns": self.mode6_enemy_threat_turns,
            "Repairing tile": (
                f"{self.mode6_robot_repair_tile} "
                f"({self.mode6_robot_repair_turns}/2)"
                if self.mode6_robot_repair_tile is not None else "-"),
            "Robot commitment": (
                f"locked {self.mode6_robot_committed_tile}"
                if self.mode6_robot_committed_tile is not None else "-"),
            "Golden status": self._mode6_golden_status(),
            "Depth": depth,
        }
        if self.algorithm_name in ("Expectimax", "Expectiminimax"):
            live["Expected value"] = f"{self.minimax_value:.1f}"

        if self.algorithm_name == "Minimax":
            live.update({
                "Node flow": "MAX -> GREEDY",
            })
        elif self.algorithm_name == "Alpha-Beta":
            live.update({
                "Node flow": "MAX -> GREEDY",
                "Alpha": self.alpha_beta_info["alpha"],
                "Beta": self.alpha_beta_info["beta"],
                "Pruned Nodes": self.alpha_beta_info["pruned"],
            })
        elif self.algorithm_name == "Expectimax":
            live.update({
                "Node flow": "MAX -> CHANCE",
                "Chance outcomes": "An toan 50% | Set 20% | Cot thu loi 30%",
                "Chance event": self.mode6_chance_event or "-",
                "Chance probability": (
                    f"{self.mode6_chance_probability * 100:.0f}%"),
            })
        else:
            live.update({
                "Node flow": "MAX -> MIN -> CHANCE",
                "Chance event": self.mode6_chance_event or "-",
                "Chance probability": (
                    f"{self.mode6_chance_probability * 100:.0f}%"),
            })

        tree_stats = dict(self.stats)
        tree_stats.update(live)
        self.stats = tree_stats


    def _enemy_destroy_remaining(self):
        if not self.enemy_destroy_until:
            return 0.0
        remaining_ms = self.enemy_destroy_until - pygame.time.get_ticks()
        return max(0.0, remaining_ms / 1000)


    def _enemy_is_destroying(self):
        return self._enemy_destroy_remaining() > 0


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
        # The selected game tree supplies every MIN/chance response.
        return


    def _resolve_mode6_outcome(self, target):
        """Resolve MIN directly or sample the CHANCE node at the target."""
        damaged_tile = target
        outcome_key = "weather_damage"
        outcome_label = "Crow destroyed tree"
        sampled_probability = 1.0

        if self.algorithm_name == "Expectimax":
            outcomes = algorithms.mode6_chance_outcomes(
                target, self.mode6_crop_profiles)
            roll = random.random()
            cumulative = 0.0
            for event_key, probability in outcomes:
                cumulative += probability
                if roll <= cumulative:
                    outcome_key = event_key
                    sampled_probability = probability
                    break
            outcome_label = algorithms.MODE6_CHANCE_LABELS[outcome_key]
            if outcome_key in ("no_damage", "lightning_rod"):
                damaged_tile = None
            self.mode6_chance_event = outcome_label
            self.mode6_chance_probability = sampled_probability

        if damaged_tile is not None:
            self.enemy_done_tiles.add(damaged_tile)
            if self.algorithm_name == "Expectimax":
                self._trigger_lightning_effect(damaged_tile)
            self.message = f"{outcome_label}: {damaged_tile}"
        else:
            if outcome_key == "lightning_rod" and target is not None:
                self._trigger_lightning_effect(
                    self.mode6_lightning_rod_tile or target)
            self.message = f"Chance node: {outcome_label}"
        self.mode6_phase = (
            "MODE6_CHANCE"
            if self.algorithm_name == "Expectimax"
            else "MODE6_RESOLVE")
        self._update_mode6_step_hud(
            phase=self.mode6_phase,
            current=self._world_to_tile(self.player.rect.center),
            next_tile=self.current_target,
        )
        self.stats["Sampled Outcome"] = outcome_label
        self.stats["Outcome target"] = (
            f"{damaged_tile}" if damaged_tile is not None else "None")
        self.stats["Outcome probability"] = (
            f"{sampled_probability * 100:.1f}%")
        return damaged_tile


    def _plan_enemy_path(self, target):
        """Plan a legal four-direction path without changing player HUD stats."""
        if not self.enemy_tile or not target:
            return []
        start = (
            int(round(self.enemy_tile[0])),
            int(round(self.enemy_tile[1])),
        )
        blocked = self._blocked_tiles()
        blocked.discard(start)
        blocked.discard(target)
        path, _, _, _ = algorithms.find_path_by_algorithm(
            "A*", start, target, blocked, self._neighbors,
            self._search_heuristic, self.counter, self._step_cost)
        self.mode6_enemy_move_total = len(path)
        self.mode6_enemy_move_done = 0
        return path


    def _mode6_processed_tiles(self):
        """All Mode 6 trees already claimed by either actor."""
        return set(self.done_tiles) | set(self.enemy_done_tiles)


    def _mode6_clear_resolved_targets(self):
        processed = self._mode6_processed_tiles()
        if not processed:
            return

        for attr in (
                "current_target",
                "enemy_target",
                "mode6_robot_target",
                "mode6_enemy_target",
                "mode6_robot_step_target",
                "mode6_enemy_step_target",
                "mode6_pending_target"):
            if getattr(self, attr, None) in processed:
                setattr(self, attr, None)

        if self.mode6_robot_repair_tile in processed:
            self.mode6_robot_repair_tile = None
            self.mode6_robot_repair_turns = 0
        if self.mode6_enemy_threat_tile in processed:
            self.mode6_enemy_threat_tile = None
            self.mode6_enemy_threat_turns = 0
        if self.mode6_enemy_forbidden_tile in processed:
            self.mode6_enemy_forbidden_tile = None
        if self.mode6_robot_committed_tile in processed:
            self.mode6_robot_committed_tile = None

        if self.mode6_robot_target is None:
            self.mode6_robot_path = []
        if self.mode6_enemy_target is None:
            self.mode6_enemy_path = []

        for key in (
                "robot_target",
                "enemy_target",
                "enemy_priority_target",
                "enemy_threat_tile",
                "robot_repair_tile"):
            if self.mode6_tree_details.get(key) in processed:
                self.mode6_tree_details[key] = None


    def _mode6_remaining_tiles(self):
        processed = self._mode6_processed_tiles()
        return [
            tile for tile in self.farm_tiles
            if tile not in processed
        ]


    def _mode6_select_active_chance_tile(self):
        remaining = self._mode6_remaining_tiles()
        if not remaining:
            return None
        tiles, weights = algorithms.mode6_chance_tile_weights(
            remaining, self.mode6_crop_profiles)
        return random.choices(tiles, weights=weights, k=1)[0]


    def _mode6_should_interrupt_repair(self, robot_tile, remaining_set):
        current = self.mode6_robot_repair_tile
        if current != robot_tile:
            return False
        if self.mode6_robot_repair_turns <= 0:
            return False

        current_value = self.mode6_crop_profiles.get(
            current, {}).get("value", 0)
        enemy_tile = (
            self._mode6_actor_tile(self.enemy_tile)
            if self.enemy_tile is not None else None
        )

        for tile in remaining_set:
            if tile == current:
                continue

            value = self.mode6_crop_profiles.get(tile, {}).get("value", 0)
            if value < current_value + self.MODE6_INTERRUPT_VALUE_MARGIN:
                continue

            robot_dist = self._heuristic(robot_tile, tile)
            enemy_dist = (
                self._heuristic(enemy_tile, tile)
                if enemy_tile is not None else 999
            )

            threatened = tile == self.mode6_enemy_threat_tile
            if threatened:
                return True
            if enemy_dist <= robot_dist + 1:
                return True

        return False


    def _mode6_apply_enemy_contested_retarget(self, enemy_tile, remaining_set):
        contested = self.mode6_enemy_forbidden_tile
        if contested is None:
            return
        if contested not in remaining_set:
            self.mode6_enemy_forbidden_tile = None
            return
        if enemy_tile != contested:
            self.mode6_enemy_forbidden_tile = None
            return

        alternatives = set(remaining_set)
        alternatives.discard(contested)
        fallback = self._mode6_progress_action(
            enemy_tile, alternatives, forbidden_tile=contested)
        if fallback == "STAY":
            return

        self.mode6_enemy_action = fallback
        self.mode6_tree_details["enemy_action"] = fallback
        self.mode6_tree_details["enemy_target"] = None
        self.mode6_tree_details["enemy_priority_target"] = None
        self.mode6_tree_details["enemy_retarget"] = (
            f"robot contested {contested}; crow picks another tree")
        if isinstance(self.stats, dict):
            self.stats["Enemy retarget"] = (
                f"robot contested {contested}; choose another tree")


    def _mode6_apply_stay_claims(self, robot_tile, enemy_tile, remaining_set):
        robot_claim = self.mode6_robot_repair_tile
        enemy_claim = self.mode6_enemy_threat_tile

        if robot_claim in remaining_set:
            enemy_next = self._mode6_apply_action(
                enemy_tile, self.mode6_enemy_action)
            if enemy_next == robot_claim and enemy_tile != robot_claim:
                alternatives = set(remaining_set)
                alternatives.discard(robot_claim)
                fallback = self._mode6_progress_action(
                    enemy_tile, alternatives, forbidden_tile=robot_claim)
                if fallback != "STAY":
                    self.mode6_enemy_action = fallback
                    self.mode6_tree_details["enemy_action"] = fallback
                    self.mode6_tree_details["enemy_claim_rule"] = (
                        f"robot already repairing {robot_claim}")
                    if isinstance(self.stats, dict):
                        self.stats["Enemy claim rule"] = (
                            f"avoid robot repairing {robot_claim}")

        if enemy_claim in remaining_set:
            robot_next = self._mode6_apply_action(
                robot_tile, self.mode6_robot_action)
            if robot_next == enemy_claim and robot_tile != enemy_claim:
                alternatives = set(remaining_set)
                alternatives.discard(enemy_claim)
                fallback = self._mode6_progress_action(
                    robot_tile, alternatives, forbidden_tile=enemy_claim)
                if fallback != "STAY":
                    self.mode6_robot_action = fallback
                    self.mode6_tree_details["robot_action"] = fallback
                    self.mode6_tree_details["robot_claim_rule"] = (
                        f"crow already threatening {enemy_claim}")
                    if isinstance(self.stats, dict):
                        self.stats["Robot claim rule"] = (
                            f"avoid crow threatening {enemy_claim}")

    def _mode6_golden_tile(self):
        for tile in self.farm_tiles:
            if self._condition_for_tile(tile) == "golden":
                return tile
        return None


    def _mode6_golden_status(self):
        tile = self._mode6_golden_tile()
        if tile is None:
            return "none"
        if tile in self.done_tiles:
            return "protected"
        if tile in self.enemy_done_tiles:
            return "destroyed"
        return "remaining"


    def _mode6_finish_game(self):
        self.player.speed = self.mode6_player_base_speed
        protected_score, destroyed_score = algorithms.mode6_state_scores(
            self.done_tiles, self.enemy_done_tiles, self.mode6_crop_profiles)
        final_score = algorithms.evaluate_farm_state(
            self.farm_tiles, self.done_tiles, self.enemy_done_tiles,
            self.mode6_crop_profiles)
        if protected_score > destroyed_score:
            result = "AGRI-1 thang"
        elif protected_score < destroyed_score:
            result = "Qua thang"
        else:
            result = "Hoa"
        self.state = "DONE"
        self.mode6_phase = "DONE"
        self.player.direction.update(0, 0)
        self.message = (
            f"{result}! Score {final_score:.1f} | "
            f"Bao ve {protected_score} - Bi pha {destroyed_score}")
        self._update_mode6_step_hud(phase="DONE")


    def _mode6_run_chance(self):
        self.mode6_enemy_threat_tile = None
        self.mode6_enemy_threat_turns = 0
        tile = self.mode6_pending_target
        self.mode6_pending_actor = None
        self.mode6_pending_target = None
        if tile is None or tile in self.done_tiles or tile in self.enemy_done_tiles:
            self.mode6_chance_event = "No damage"
            self.mode6_chance_probability = 0.0
            self._update_mode6_step_hud(phase="CHANCE")
            self._mode6_next_state_or_done()
            return

        outcomes = algorithms.mode6_chance_outcomes(
            tile, self.mode6_crop_profiles)
        roll = random.random()
        cumulative = 0.0
        event_key = "no_damage"
        probability = 0.0
        for candidate, candidate_probability in outcomes:
            cumulative += candidate_probability
            if roll <= cumulative:
                event_key = candidate
                probability = candidate_probability
                break
        label = algorithms.MODE6_CHANCE_LABELS[event_key]
        self.mode6_chance_event = label
        self.mode6_chance_probability = probability
        if event_key == "no_damage":
            self.message = f"CHANCE: {label}, cay {tile} van con remaining"
        elif event_key == "lightning_rod":
            self._trigger_lightning_effect(
                self.mode6_lightning_rod_tile or tile)
            self.message = f"CHANCE: {label}, cay {tile} duoc bao ve"
        elif event_key == "weather_damage":
            self.enemy_done_tiles.add(tile)
            self._trigger_lightning_effect(tile)
            if self.mode6_robot_repair_tile == tile:
                self.mode6_robot_repair_tile = None
                self.mode6_robot_repair_turns = 0
            value = self.mode6_crop_profiles.get(tile, {}).get("value", 0)
            self.message = f"CHANCE: {label}, cay {tile} bi pha (-{value})"
        self._mode6_clear_resolved_targets()
        self._update_mode6_step_hud(phase="CHANCE")
        self._mode6_next_state_or_done()


    def _mode6_actor_tile(self, pos):
        return int(round(pos[0])), int(round(pos[1]))


    def _mode6_walkable_tiles(self):
        walkable = set(self.walkable_tiles or [])
        walkable.update(self.farm_tiles)
        walkable.add(self.spawn_tile)
        if self.enemy_spawn_tile is not None:
            walkable.add(self.enemy_spawn_tile)
        if self.enemy_tile is not None:
            walkable.add(self._mode6_actor_tile(self.enemy_tile))
        return walkable


    def _mode6_apply_action(self, pos, action):
        dx, dy = algorithms.MODE6_ACTION_DELTAS.get(action, (0, 0))
        next_tile = (pos[0] + dx, pos[1] + dy)
        blocked = self._blocked_tiles(include_hidden=True)
        walkable = self._mode6_walkable_tiles()
        if next_tile in walkable and next_tile not in blocked:
            return next_tile
        return pos


    def _mode6_path_to_target(self, start, target):
        if target is None:
            return []
        if start == target:
            return []
        blocked = self._blocked_tiles()
        blocked.discard(start)
        blocked.discard(target)

        # Mode 6 should not route through trees already claimed by either side.
        # Keep the actor's current tile open so it can leave a resolved tree.
        resolved = self._mode6_processed_tiles()
        resolved.discard(start)
        resolved.discard(target)
        preferred_blocked = blocked | resolved
        path, _, _, _ = algorithms.astar(
            start, target, preferred_blocked, self._neighbors, self._heuristic,
            self.counter, self._step_cost)
        if path:
            return path

        # A resolved row can split the board. Allow crossing it only when no
        # unresolved-only route exists, while the target itself stays remaining.
        path, _, _, _ = algorithms.astar(
            start, target, blocked, self._neighbors, self._heuristic,
            self.counter, self._step_cost)
        return path


    def _mode6_retarget_from_ranked(self, start, ranked_targets,
                                    locked_target, locked_route):
        remaining = set(self._mode6_remaining_tiles())
        locked_route = set(locked_route or [])
        for tile in ranked_targets:
            if tile == locked_target or tile in locked_route:
                continue
            if tile not in remaining:
                continue
            path = self._mode6_path_to_target(start, tile)
            if path or start == tile:
                return tile, path
        for tile in ranked_targets:
            if tile == locked_target or tile not in remaining:
                continue
            path = self._mode6_path_to_target(start, tile)
            if path or start == tile:
                return tile, path
        return None, []


    def _mode6_apply_target_lock(self, robot_tile, enemy_tile):
        if self.mode6_robot_target is None or self.mode6_enemy_target is None:
            return
        remaining = set(self._mode6_remaining_tiles())
        if (self.mode6_robot_target not in remaining
                or self.mode6_enemy_target not in remaining):
            return

        robot_route = [robot_tile] + list(self.mode6_robot_path)
        enemy_route = [enemy_tile] + list(self.mode6_enemy_path)
        robot_route_set = set(robot_route)
        enemy_route_set = set(enemy_route)
        same_target = self.mode6_robot_target == self.mode6_enemy_target
        enemy_target_on_robot_route = self.mode6_enemy_target in robot_route_set
        robot_target_on_enemy_route = self.mode6_robot_target in enemy_route_set
        if not (same_target or enemy_target_on_robot_route
                or robot_target_on_enemy_route):
            return

        robot_distance = (
            0 if robot_tile == self.mode6_robot_target
            else len(self.mode6_robot_path) if self.mode6_robot_path
            else 10 ** 6)
        enemy_distance = (
            0 if enemy_tile == self.mode6_enemy_target
            else len(self.mode6_enemy_path) if self.mode6_enemy_path
            else 10 ** 6)
        if robot_distance >= 10 ** 6 and enemy_distance >= 10 ** 6:
            return

        robot_keeps = False
        if same_target:
            robot_keeps = robot_distance <= enemy_distance
        elif enemy_target_on_robot_route and not robot_target_on_enemy_route:
            robot_keeps = True
        elif robot_target_on_enemy_route and not enemy_target_on_robot_route:
            robot_keeps = False
        else:
            robot_keeps = robot_distance <= enemy_distance

        if robot_keeps:
            new_target, new_path = self._mode6_retarget_from_ranked(
                enemy_tile,
                self.mode6_tree_details.get("enemy_ranked_targets", []),
                self.mode6_robot_target,
                robot_route)
            if new_target is not None:
                self.mode6_enemy_target = new_target
                self.mode6_enemy_path = new_path
                self.mode6_tree_details["target_lock"] = (
                    f"robot keeps {self.mode6_robot_target}")
            return

        new_target, new_path = self._mode6_retarget_from_ranked(
            robot_tile,
            self.mode6_tree_details.get("robot_ranked_targets", []),
            self.mode6_enemy_target,
            enemy_route)
        if new_target is not None:
            self.mode6_robot_target = new_target
            self.mode6_robot_path = new_path
            self.mode6_tree_details["target_lock"] = (
                f"enemy keeps {self.mode6_enemy_target}")


    def _mode6_should_force_target_lock(self, robot_tile, enemy_tile):
        shared_target = None
        if (self.mode6_robot_target is not None
                and self.mode6_robot_target == self.mode6_enemy_target):
            shared_target = self.mode6_robot_target

        if shared_target is None:
            self.mode6_shared_target = None
            self.mode6_shared_target_count = 0
        elif shared_target == self.mode6_shared_target:
            self.mode6_shared_target_count += 1
        else:
            self.mode6_shared_target = shared_target
            self.mode6_shared_target_count = 1

        current = (robot_tile, enemy_tile)
        history = list(getattr(self, "mode6_loop_positions", []))
        history.append(current)
        self.mode6_loop_positions = history[-4:]
        self.mode6_loop_count = self.mode6_shared_target_count
        return self.mode6_shared_target_count >= 3


    def _mode6_apply_enemy_aim_commit(self, enemy_tile, remaining_set):
        target = self.mode6_tree_details.get("enemy_priority_target")
        if target not in remaining_set:
            self.mode6_enemy_aim_target = None
            self.mode6_enemy_aim_count = 0
            return

        if target == self.mode6_enemy_aim_target:
            self.mode6_enemy_aim_count += 1
        else:
            self.mode6_enemy_aim_target = target
            self.mode6_enemy_aim_count = 1

        self.mode6_tree_details["enemy_aim_count"] = (
            f"{self.mode6_enemy_aim_target}: "
            f"{self.mode6_enemy_aim_count}/2")
        if self.mode6_enemy_aim_count < 2:
            return

        path = self._mode6_path_to_target(enemy_tile, target)
        if not path and enemy_tile != target:
            return

        self.mode6_enemy_target = target
        self.mode6_enemy_path = path
        enemy_step = path[0] if path else enemy_tile
        self.mode6_enemy_action = self._mode6_action_from_step(
            enemy_tile, enemy_step)
        self.mode6_tree_details["enemy_aim_commit"] = (
            f"enemy commits {target} after {self.mode6_enemy_aim_count} aims")
        if isinstance(self.stats, dict):
            self.stats["Enemy aim commit"] = (
                f"{target} ({self.mode6_enemy_aim_count}/2)")


    def _mode6_action_from_step(self, start, step):
        dx = step[0] - start[0]
        dy = step[1] - start[1]
        for action, delta in algorithms.MODE6_ACTION_DELTAS.items():
            if delta == (dx, dy):
                return action
        return "STAY"


    def _mode6_progress_action(self, pos, remaining, forbidden_tile=None):
        walkable = self._mode6_walkable_tiles()
        candidates = []
        for action in ("UP", "DOWN", "LEFT", "RIGHT"):
            next_tile = self._mode6_apply_action(pos, action)
            if next_tile == pos or next_tile not in walkable:
                continue
            if forbidden_tile is not None and next_tile == forbidden_tile:
                continue
            if remaining:
                nearest = min(
                    abs(next_tile[0] - tile[0]) + abs(next_tile[1] - tile[1])
                    for tile in remaining)
            else:
                nearest = 0
            candidates.append((nearest, algorithms.MODE6_ACTIONS.index(action),
                               action))
        if not candidates:
            return "STAY"
        candidates.sort()
        return candidates[0][2]


    def _mode6_snap_player_to_tile(self, tile):
        target_world = self._tile_center(tile)
        self.player.pos.update(target_world)
        self.player.rect.center = (round(target_world.x), round(target_world.y))
        self.player.hitbox.center = self.player.rect.center
        self.player.direction.update(0, 0)


    def _mode6_set_player_step_direction(self, delta):
        if abs(delta.x) > abs(delta.y):
            direction_x = 1 if delta.x > 0 else -1
            self.player.direction.update(direction_x, 0)
            self.player.status = "right" if direction_x > 0 else "left"
        else:
            direction_y = 1 if delta.y > 0 else -1
            self.player.direction.update(0, direction_y)
            self.player.status = "down" if direction_y > 0 else "up"


    def _mode6_step_is_done(self, actor_tile, target):
        return target is None or target == actor_tile


    def _mode6_player_at_target(self, target):
        if target is None:
            return True
        target_world = self._tile_center(target)
        return (target_world - self.player.pos).length() < 0.75


    def _mode6_enemy_at_target(self, target):
        if self.enemy_tile is None or target is None:
            return True
        return (
            abs(target[0] - self.enemy_tile[0])
            + abs(target[1] - self.enemy_tile[1])
        ) < 0.01


    def _mode6_move_enemy_to(self, dt, target):
        if self.enemy_tile is None or target is None:
            return True
        ex, ey = self.enemy_tile
        dx = target[0] - ex
        dy = target[1] - ey
        distance = abs(dx) + abs(dy)
        if distance < 0.01:
            self.enemy_tile = target
            return True
        step = (
            self.mode6_player_base_speed * self.MODE6_ENEMY_SPEED_SCALE
            / TILE_SIZE * dt)
        if distance <= step:
            self.enemy_tile = target
            return True
        if abs(dx) > 0:
            self.enemy_tile = (ex + (step if dx > 0 else -step), ey)
        elif abs(dy) > 0:
            self.enemy_tile = (ex, ey + (step if dy > 0 else -step))
        return False


    def _mode6_move_player_to(self, dt, target):
        if target is None:
            self.player.direction.update(0, 0)
            return True
        target_world = self._tile_center(target)
        delta = target_world - self.player.pos
        distance = delta.length()
        if distance < 0.75:
            self._mode6_snap_player_to_tile(target)
            return True

        self._mode6_set_player_step_direction(delta)
        step = self.mode6_player_base_speed * self.MODE6_ROBOT_SPEED_SCALE * dt
        if distance <= step:
            self._mode6_snap_player_to_tile(target)
            return True

        if abs(delta.x) > abs(delta.y):
            self.player.pos.x += step if delta.x > 0 else -step
        else:
            self.player.pos.y += step if delta.y > 0 else -step
        self.player.rect.center = (
            round(self.player.pos.x), round(self.player.pos.y))
        self.player.hitbox.center = self.player.rect.center
        self.player.skip_auto_move_once = True
        return False


    def _mode6_next_state_or_done(self):
        self._mode6_clear_resolved_targets()
        if len(self.done_tiles) + len(self.enemy_done_tiles) >= len(self.farm_tiles):
            self._mode6_finish_game()
        else:
            self.mode6_resolve_elapsed = 0.0
            self.mode6_robot_path = []
            self.mode6_enemy_path = []
            self.state = "PLAN_STEP"
            self.mode6_phase = "PLAN_STEP"


    def _update_mode6(self, dt):
        if not self.is_running:
            self.player.speed = self.mode6_player_base_speed
            self.player.direction.update(0, 0)
            return
        if self.state == "DONE":
            self.player.speed = self.mode6_player_base_speed
            self.player.direction.update(0, 0)
            self._update_mode6_step_hud(phase="DONE")
            return

        if (self.algorithm_name in ("Expectimax", "Expectiminimax")
                and self.state != "CHANCE"):
            self.mode6_random_chance_elapsed += dt

        if self.state in ("PLAN_STEP", "CHOOSE_TARGET"):
            self._mode6_clear_resolved_targets()
        remaining = self._mode6_remaining_tiles()
        if not remaining and self.state not in ("CHANCE", "RESOLVE_STEP"):
            self._mode6_finish_game()
            return

        if self.state in ("PLAN_STEP", "CHOOSE_TARGET"):
            self.state = "CHOOSE_TARGET"
            self.mode6_phase = "CHOOSE_TARGET"
            self.player.direction.update(0, 0)
            if self.mode6_plan_cooldown > 0:
                self.mode6_plan_cooldown = max(
                    0.0, self.mode6_plan_cooldown - dt)
                self.message = (
                    f"Dang doi lap ke hoach: "
                    f"{self.mode6_plan_cooldown:.1f}s")
                self._update_mode6_step_hud(phase="CHOOSE_TARGET")
                return
            try:
                self.mode6_turn += 1
                self.mode6_chance_event = ""
                self.mode6_chance_probability = 0.0
                robot_tile = self._world_to_tile(self.player.rect.center)
                enemy_tile = self._mode6_actor_tile(self.enemy_tile)
                self.mode6_robot_step_start = robot_tile
                self.mode6_enemy_step_start = enemy_tile
                mode6_walkable = set(self._mode6_walkable_tiles())
                mode6_walkable -= self._blocked_tiles(include_hidden=True)
                plan_started = pygame.time.get_ticks()
                result = algorithms.adversarial_choose(
                    self.algorithm_name,
                    robot_tile,
                    enemy_tile,
                    tuple(self.farm_tiles),
                    frozenset(self.done_tiles),
                    frozenset(self.enemy_done_tiles),
                    depth=algorithms.MODE6_SEARCH_DEPTH,
                    crop_profiles={
                        tile: {
                            **dict(profile),
                            "condition": self._condition_for_tile(tile),
                        }
                        for tile, profile in self.mode6_crop_profiles.items()
                    },
                    walkable_tiles=mode6_walkable,
                    robot_prev=self.mode6_robot_prev_tile,
                    enemy_prev=self.mode6_enemy_prev_tile,
                    enemy_threat_tile=self.mode6_enemy_threat_tile,
                    enemy_threat_turns=self.mode6_enemy_threat_turns,
                    robot_repair_tile=self.mode6_robot_repair_tile,
                    robot_repair_turns=self.mode6_robot_repair_turns,
                )
                self.mode6_last_plan_ms = (
                    pygame.time.get_ticks() - plan_started)
            except Exception as error:
                self.is_running = False
                self.message = f"Mode 6 search loi: {error}"
                return
            (self.mode6_robot_action, self.mode6_enemy_action,
             self.minimax_value, self.alpha_beta_info, self.stats,
             self.mode6_tree_details) = result
            if isinstance(self.stats, dict):
                self.stats["Plan time"] = f"{self.mode6_last_plan_ms}ms"
                self.stats["Search depth"] = algorithms.MODE6_SEARCH_DEPTH
            self.mode6_tree_details["plan_time_ms"] = self.mode6_last_plan_ms
            self.mode6_plan_cooldown = self.MODE6_PLAN_COOLDOWN
            robot_tile = self.mode6_robot_step_start
            enemy_tile = self.mode6_enemy_step_start
            remaining_set = set(remaining)
            if self.algorithm_name == "Expectimax":
                self.mode6_enemy_action = "STAY"
                self.mode6_tree_details["enemy_action"] = "CHANCE"
                self.mode6_tree_details["enemy_target"] = None
                self.mode6_tree_details["enemy_priority_target"] = None
                self.mode6_tree_details["enemy_policy"] = "Random Chance"
                if isinstance(self.stats, dict):
                    self.stats["Enemy Policy"] = "Random Chance"
                    self.stats["Enemy Action"] = "CHANCE"
                    self.stats["MIN action"] = "N/A"
            partial_repair = (
                robot_tile in remaining_set
                and self.mode6_robot_repair_tile == robot_tile
                and self.mode6_robot_repair_turns > 0
            )
            if partial_repair:
                if self._mode6_should_interrupt_repair(
                        robot_tile, remaining_set):
                    self.mode6_tree_details["robot_commitment"] = (
                        "interrupt repair: enemy target value +35")
                    if isinstance(self.stats, dict):
                        self.stats["Robot commitment"] = (
                            "interrupt repair: enemy target value +35")
                else:
                    self.mode6_robot_action = "STAY"
                    self.mode6_tree_details["robot_action"] = "STAY"
                    self.mode6_tree_details["robot_commitment"] = (
                        "finish repair: no enemy target +35")
                    if isinstance(self.stats, dict):
                        self.stats["Robot commitment"] = (
                            "finish repair: no enemy target +35")
            elif self.mode6_robot_committed_tile is not None:
                self.mode6_robot_committed_tile = None
            if self.algorithm_name != "Expectimax":
                self._mode6_apply_enemy_contested_retarget(
                    enemy_tile, remaining_set)
                self._mode6_apply_stay_claims(
                    robot_tile, enemy_tile, remaining_set)
                self.mode6_robot_target = self._mode6_apply_action(
                    robot_tile, self.mode6_robot_action)
                self.mode6_enemy_target = self._mode6_apply_action(
                    enemy_tile, self.mode6_enemy_action)
                force_target_lock = self._mode6_should_force_target_lock(
                    robot_tile, enemy_tile)
                if force_target_lock:
                    target_result = algorithms.adversarial_choose_target(
                        self.algorithm_name,
                        robot_tile,
                        enemy_tile,
                        tuple(self.farm_tiles),
                        frozenset(self.done_tiles),
                        frozenset(self.enemy_done_tiles),
                        crop_profiles={
                            tile: {
                                **dict(profile),
                                "condition": self._condition_for_tile(tile),
                            }
                            for tile, profile in self.mode6_crop_profiles.items()
                        },
                        walkable_tiles=mode6_walkable,
                        robot_prev=self.mode6_robot_prev_tile,
                        enemy_prev=self.mode6_enemy_prev_tile,
                        enemy_threat_tile=self.mode6_enemy_threat_tile,
                        enemy_threat_turns=self.mode6_enemy_threat_turns,
                        robot_repair_tile=self.mode6_robot_repair_tile,
                        robot_repair_turns=self.mode6_robot_repair_turns,
                    )
                    target_robot, target_enemy, _, _, target_details = target_result
                    self.mode6_tree_details.update({
                        "robot_final_target": target_robot,
                        "enemy_final_target": target_enemy,
                        "robot_ranked_targets": target_details.get(
                            "robot_ranked_targets", []),
                        "enemy_ranked_targets": target_details.get(
                            "enemy_ranked_targets", []),
                    })
                else:
                    target_robot = None
                    target_enemy = None
                if (force_target_lock and target_robot in remaining_set):
                    self.mode6_robot_target = target_robot
                    self.mode6_robot_path = self._mode6_path_to_target(
                        robot_tile, target_robot)
                if (force_target_lock and target_enemy in remaining_set):
                    self.mode6_enemy_target = target_enemy
                    self.mode6_enemy_path = self._mode6_path_to_target(
                        enemy_tile, target_enemy)
                if force_target_lock:
                    self._mode6_apply_target_lock(robot_tile, enemy_tile)
                    self.mode6_tree_details["target_lock_mode"] = (
                        f"shared target lock ({self.mode6_shared_target_count}/3)")
                    if self.mode6_robot_target in remaining_set:
                        self.mode6_robot_path = self._mode6_path_to_target(
                            robot_tile, self.mode6_robot_target)
                        robot_step = (
                            self.mode6_robot_path[0]
                            if self.mode6_robot_path else robot_tile)
                        self.mode6_robot_action = self._mode6_action_from_step(
                            robot_tile, robot_step)
                    if self.mode6_enemy_target in remaining_set:
                        self.mode6_enemy_path = self._mode6_path_to_target(
                            enemy_tile, self.mode6_enemy_target)
                        enemy_step = (
                            self.mode6_enemy_path[0]
                            if self.mode6_enemy_path else enemy_tile)
                        self.mode6_enemy_action = self._mode6_action_from_step(
                            enemy_tile, enemy_step)
                self._mode6_apply_enemy_aim_commit(
                    enemy_tile, remaining_set)
            self.mode6_robot_step_target = self._mode6_apply_action(
                robot_tile, self.mode6_robot_action)
            self.mode6_enemy_step_target = self._mode6_apply_action(
                enemy_tile, self.mode6_enemy_action)
            if self.algorithm_name == "Expectimax":
                self.mode6_robot_target = self.mode6_robot_step_target
                self.mode6_enemy_target = self.mode6_enemy_step_target
                self.mode6_robot_path = (
                    [] if self.mode6_robot_step_target == robot_tile
                    else [self.mode6_robot_step_target])
                self.mode6_enemy_path = (
                    [] if self.mode6_enemy_step_target == enemy_tile
                    else [self.mode6_enemy_step_target])
            else:
                if self.mode6_robot_target in remaining_set:
                    self.mode6_robot_path = self._mode6_path_to_target(
                        robot_tile, self.mode6_robot_target)
                else:
                    self.mode6_robot_target = self.mode6_robot_step_target
                    self.mode6_robot_path = (
                        [] if self.mode6_robot_step_target == robot_tile
                        else [self.mode6_robot_step_target])
                if self.mode6_enemy_target in remaining_set:
                    self.mode6_enemy_path = self._mode6_path_to_target(
                        enemy_tile, self.mode6_enemy_target)
                else:
                    self.mode6_enemy_target = self.mode6_enemy_step_target
                    self.mode6_enemy_path = (
                        [] if self.mode6_enemy_step_target == enemy_tile
                        else [self.mode6_enemy_step_target])
            self.current_target = self.mode6_robot_step_target
            self.enemy_target = (
                None if self.algorithm_name == "Expectimax"
                else self.mode6_enemy_step_target)
            display_enemy_action = (
                "CHANCE" if self.algorithm_name == "Expectimax"
                else self.mode6_enemy_action)
            self.mode6_tree_details["robot_action"] = self.mode6_robot_action
            self.mode6_tree_details["enemy_action"] = display_enemy_action
            self.mode6_tree_details["robot_target"] = self.mode6_robot_target
            self.mode6_tree_details["enemy_target"] = (
                None if self.algorithm_name == "Expectimax"
                else self.mode6_enemy_target)
            self.mode6_tree_details["robot_path_length"] = len(
                self.mode6_robot_path)
            self.mode6_tree_details["enemy_path_length"] = len(
                self.mode6_enemy_path)
            if isinstance(self.stats, dict):
                self.stats["Robot next tile"] = self.mode6_robot_target
                self.stats["Enemy next tile"] = (
                    None if self.algorithm_name == "Expectimax"
                    else self.mode6_enemy_target)
                if self.algorithm_name == "Expectimax":
                    self.stats["Enemy Action"] = "CHANCE"
                    self.stats["MIN action"] = "N/A"
            self.mode6_tree_details["crop_profiles"] = {
                tile: {
                    "condition": self._condition_for_tile(tile),
                    "value": profile["value"],
                    "risk": profile["risk"],
                    "state": (
                        "protected" if tile in self.done_tiles
                        else "destroyed" if tile in self.enemy_done_tiles
                        else "remaining"),
                }
                for tile, profile in self.mode6_crop_profiles.items()
            }
            robot_done = self._mode6_step_is_done(
                robot_tile, self.mode6_robot_step_target)
            enemy_done = self._mode6_step_is_done(
                enemy_tile, self.mode6_enemy_step_target)
            self.mode6_robot_hold_required = (
                robot_done
                and self.mode6_robot_action == "STAY"
                and robot_tile in remaining_set)
            self.mode6_enemy_hold_required = (
                self.algorithm_name != "Expectimax"
                and enemy_done
                and self.mode6_enemy_action == "STAY"
                and enemy_tile in remaining_set
                and enemy_tile != robot_tile)
            self.mode6_robot_hold_elapsed = 0.0
            self.mode6_enemy_hold_elapsed = 0.0
            self.mode6_move_total = (
                1 if self.mode6_robot_hold_required
                else 0 if robot_done else 1)
            self.mode6_move_done = 0
            self.mode6_enemy_move_total = (
                1 if self.mode6_enemy_hold_required
                else 0 if enemy_done else 1)
            self.mode6_enemy_move_done = 0
            self.mode6_resolve_elapsed = 0.0
            if robot_done:
                self._mode6_snap_player_to_tile(robot_tile)
            if enemy_done:
                self.enemy_tile = enemy_tile
            if (robot_done and enemy_done
                    and not self.mode6_robot_hold_required
                    and not self.mode6_enemy_hold_required):
                self.state = "RESOLVE_STEP"
                self.mode6_phase = "RESOLVE_STEP"
            else:
                self.state = "MOVE_STEP"
                self.mode6_phase = "MOVE_STEP"
            if self.algorithm_name == "Expectimax":
                self.message = (
                    f"Target AGRI {self.mode6_robot_target}; "
                    "CHANCE random")
            else:
                self.message = (
                    f"Target AGRI {self.mode6_robot_target}; "
                    f"Qua {self.mode6_enemy_target}")
            self._update_mode6_step_hud(phase=self.mode6_phase)
            return

        if self.state == "MOVE_STEP":
            self.mode6_phase = "MOVE_STEP"
            self.player.speed = (
                self.mode6_player_base_speed * self.MODE6_ROBOT_SPEED_SCALE)
            robot_tile = self._world_to_tile(self.player.rect.center)
            enemy_tile = self._mode6_actor_tile(self.enemy_tile)
            robot_arrived = self._mode6_player_at_target(
                self.mode6_robot_step_target)
            enemy_arrived = self._mode6_enemy_at_target(
                self.mode6_enemy_step_target)
            if self.mode6_robot_hold_required:
                self.mode6_robot_hold_elapsed = min(
                    self.MODE6_ACTION_TIME,
                    self.mode6_robot_hold_elapsed + dt)
                if (self.mode6_robot_hold_elapsed
                        >= self.MODE6_ACTION_TIME * 0.5):
                    self.mode6_robot_committed_tile = robot_tile
                robot_arrived = (
                    self.mode6_robot_hold_elapsed >= self.MODE6_ACTION_TIME)
            if self.mode6_enemy_hold_required:
                self.mode6_enemy_hold_elapsed = min(
                    self.MODE6_ACTION_TIME,
                    self.mode6_enemy_hold_elapsed + dt)
                enemy_arrived = (
                    self.mode6_enemy_hold_elapsed >= self.MODE6_ACTION_TIME)
            if robot_arrived and self.mode6_robot_step_target is not None:
                self._mode6_snap_player_to_tile(
                    self.mode6_robot_step_target)
            if enemy_arrived and self.mode6_enemy_step_target is not None:
                self.enemy_tile = self.mode6_enemy_step_target
            if self.mode6_robot_step_target is not None:
                if not robot_arrived and not self.mode6_robot_hold_required:
                    robot_arrived = self._mode6_move_player_to(
                        dt, self.mode6_robot_step_target)
            if self.mode6_enemy_step_target is not None:
                if not enemy_arrived and not self.mode6_enemy_hold_required:
                    enemy_arrived = self._mode6_move_enemy_to(
                        dt, self.mode6_enemy_step_target)
            if robot_arrived:
                self.mode6_move_done = self.mode6_move_total
            if enemy_arrived:
                self.mode6_enemy_move_done = self.mode6_enemy_move_total
            if robot_arrived and enemy_arrived:
                self.player.speed = self.mode6_player_base_speed
                self.player.direction.update(0, 0)
                self.mode6_robot_prev_tile = self.mode6_robot_step_start
                self.mode6_enemy_prev_tile = self.mode6_enemy_step_start
                self.mode6_pending_actor = None
                self.mode6_pending_target = None
                self.mode6_resolve_elapsed = 0.0
                self.state = "RESOLVE_STEP"
                self.mode6_phase = "RESOLVE_STEP"
            self._update_mode6_step_hud(phase=self.mode6_phase)
            return

        if self.state == "RESOLVE_STEP":
            self.mode6_phase = "RESOLVE_STEP"
            robot_tile = self._world_to_tile(self.player.rect.center)
            enemy_tile = self._mode6_actor_tile(self.enemy_tile)
            remaining = set(self._mode6_remaining_tiles())
            resolve_pending = (
                robot_tile in remaining
                or (self.algorithm_name != "Expectimax"
                    and enemy_tile in remaining))

            if resolve_pending and self.mode6_resolve_elapsed < self.MODE6_RESOLVE_TIME:
                self.mode6_resolve_elapsed = min(
                    self.MODE6_RESOLVE_TIME,
                    self.mode6_resolve_elapsed + dt)
                self.message = (
                    f"Dang xu ly {self.mode6_resolve_elapsed:.1f}/"
                    f"{self.MODE6_RESOLVE_TIME:.1f}s")
                self._update_mode6_step_hud(phase="RESOLVE_STEP")
                return

            robot_completed = False
            if robot_tile in remaining:
                if self.mode6_robot_repair_tile == robot_tile:
                    self.mode6_robot_repair_turns += 1
                else:
                    self.mode6_robot_repair_tile = robot_tile
                    self.mode6_robot_repair_turns = 1
                if self.mode6_enemy_threat_tile == robot_tile:
                    self.mode6_enemy_threat_tile = None
                    self.mode6_enemy_threat_turns = 0
                if self.mode6_robot_repair_turns >= 2:
                    self.done_tiles.add(robot_tile)
                    remaining.remove(robot_tile)
                    self.mode6_robot_repair_tile = None
                    self.mode6_robot_repair_turns = 0
                    self.mode6_robot_committed_tile = None
                    robot_completed = True
                    value = self.mode6_crop_profiles.get(robot_tile, {}).get(
                        "value", 0)
                    if self._condition_for_tile(robot_tile) == "golden":
                        self.message = (
                            f"AGRI-1 bao ve GOLDEN {robot_tile} (+{value})")
                    else:
                        self.message = f"AGRI-1 sua {robot_tile} (+{value})"
                else:
                    self.mode6_robot_committed_tile = robot_tile
                    self.message = f"AGRI-1 dang sua {robot_tile} (1/2)"
            else:
                self.mode6_robot_repair_tile = None
                self.mode6_robot_repair_turns = 0
                if self.mode6_robot_committed_tile == robot_tile:
                    self.mode6_robot_committed_tile = None

            if self.algorithm_name == "Expectimax":
                self.mode6_enemy_threat_tile = None
                self.mode6_enemy_threat_turns = 0
            elif enemy_tile in remaining:
                if enemy_tile == robot_tile:
                    self.mode6_enemy_forbidden_tile = enemy_tile
                    self.enemy_target = None
                    self.mode6_enemy_target = None
                    self.mode6_enemy_step_target = None
                    self.mode6_enemy_path = []
                    self.mode6_enemy_threat_tile = None
                    self.mode6_enemy_threat_turns = 0
                    if not robot_completed:
                        self.message = (
                            f"AGRI-1 tranh chap {enemy_tile}; "
                            "qua doi muc tieu khac")
                else:
                    if self.mode6_enemy_threat_tile == enemy_tile:
                        self.mode6_enemy_threat_turns += 1
                    else:
                        self.mode6_enemy_threat_tile = enemy_tile
                        self.mode6_enemy_threat_turns = 1
                    if self.mode6_enemy_threat_turns >= 2:
                        self.enemy_done_tiles.add(enemy_tile)
                        if self.mode6_robot_repair_tile == enemy_tile:
                            self.mode6_robot_repair_tile = None
                            self.mode6_robot_repair_turns = 0
                        value = self.mode6_crop_profiles.get(enemy_tile, {}).get(
                            "value", 0)
                        if self._condition_for_tile(enemy_tile) == "golden":
                            self.message = (
                                f"Qua pha GOLDEN {enemy_tile} (-{value})")
                        else:
                            self.message = f"Qua pha {enemy_tile} (-{value})"
                        self.mode6_enemy_threat_tile = None
                        self.mode6_enemy_threat_turns = 0
                    elif not robot_completed:
                        self.message = f"Qua dang pha {enemy_tile} (1/2)"
            else:
                self.mode6_enemy_threat_tile = None
                self.mode6_enemy_threat_turns = 0

            self._mode6_clear_resolved_targets()
            chance_ready = (
                self.algorithm_name in ("Expectimax", "Expectiminimax")
                and self.mode6_random_chance_elapsed
                >= self.MODE6_EXPECTIMAX_CHANCE_INTERVAL
            )
            if chance_ready:
                chance_tile = self._mode6_select_active_chance_tile()
                if chance_tile is not None:
                    self.mode6_random_chance_elapsed = 0.0
                    self.mode6_pending_actor = "chance"
                    self.mode6_pending_target = chance_tile
                    self.state = "CHANCE"
                    self.mode6_phase = "CHANCE"
                    self.message += f"; CHANCE tai {chance_tile}"
                    self._update_mode6_step_hud(phase="CHANCE")
                    return
            self._update_mode6_step_hud(phase="RESOLVE_STEP")
            self._mode6_next_state_or_done()
            return

        if self.state == "CHANCE":
            self._mode6_run_chance()
            return


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
        enemy_phase = (
            "MODE6_CHANCE" if self.algorithm_name in (
                "Expectimax", "Expectiminimax") else "MODE6_MOVE")

        if self.enemy_destroy_until:
            if not self._enemy_is_destroying() and self.enemy_target:
                reached_target = self.enemy_target
                self._resolve_mode6_outcome(reached_target)
                # Weather may damage another crop, but it must not teleport
                # the crow away from the tile it physically reached.
                self.enemy_tile = reached_target
                self.enemy_target = None
                self.enemy_path = []
                self.enemy_destroy_until = 0
                self.enemy_retarget_timer = self.ENEMY_RETARGET_DELAY
            return

        rounded_enemy = (
            int(round(self.enemy_tile[0])),
            int(round(self.enemy_tile[1])),
        )
        if rounded_enemy == target and not self.enemy_path:
            if self.enemy_retreat_target and not self.enemy_target:
                self.enemy_tile = self.enemy_retreat_target
                self.enemy_retreat_target = None
                return
            self.enemy_destroy_until = (
                pygame.time.get_ticks() + int(self.ENEMY_DESTROY_TIME * 1000))
            self._update_mode6_step_hud(
                phase=(
                    "MODE6_CHANCE"
                    if self.algorithm_name in (
                        "Expectimax", "Expectiminimax")
                    else "MODE6_RESOLVE"),
                current=self._world_to_tile(self.player.rect.center),
                next_tile=self.current_target,
            )
            return

        if not self.enemy_path or self.enemy_path[-1] != target:
            self.enemy_path = self._plan_enemy_path(target)
            if not self.enemy_path:
                self.enemy_target = None
                self.mode6_phase = "PLAN_STEP"
                self.message = f"Qua khong co duong toi {target}"
                self._update_mode6_step_hud(
                    phase=self.mode6_phase,
                    current=self._world_to_tile(self.player.rect.center),
                    next_tile=self.current_target)
                return

        next_tile = self.enemy_path[0]
        ex, ey = self.enemy_tile
        dx = next_tile[0] - ex
        dy = next_tile[1] - ey
        distance = abs(dx) + abs(dy)
        step = (
            self.player.speed
            * self.MODE6_ENEMY_SPEED_SCALE
            / TILE_SIZE
            * dt
        )
        if distance <= step:
            self.enemy_tile = next_tile
            self.enemy_path.pop(0)
            self.mode6_enemy_move_done += 1
            current = self._world_to_tile(self.player.rect.center)
            player_next = self.path[0] if self.path else self.current_target
            self._update_mode6_step_hud(
                phase="MODE6_MOVE",
                current=current,
                next_tile=player_next,
            )
            return

        # Each path segment changes only one coordinate, matching the player.
        if abs(dx) > 0:
            self.enemy_tile = (
                ex + (step if dx > 0 else -step),
                ey,
            )
        elif abs(dy) > 0:
            self.enemy_tile = (
                ex,
                ey + (step if dy > 0 else -step),
            )

    # -------------------------------------------------------- path dispatch

    # ------------------------------------------------------------------
    # target selection / update / drawing
    # ------------------------------------------------------------------

    def _path_for_mode(self, start, goal):
        if self.mode in (1, 2):
            return self._search_path(start, goal)
        if self.mode == 4:
            return self._online_plan(start, goal)
        return self._astar(start, goal)


    def _full_plan_enabled(self):
        # Mode 3 must run step-by-step through _local_search_choose(),
        # not through build_local_search_full_plan(), otherwise it uses
        # the old full-plan score and can stop/route incorrectly.
        return (
            self.mode in (1, 2, 5)
            and not (self.mode == 1 and self.algorithm_name == "IDS"))


    def _apply_full_plan_stats(self):
        if not self.full_garden_plan:
            return
        stats = dict(self.stats or {})
        stats.update(self.full_garden_stats)
        stats["Algorithm"] = self.algorithm_name
        stats["Planning mode"] = stats.get(
            "Planning mode", "Full garden search")
        stats["Plan target"] = (
            f"{min(self.full_garden_index + 1, len(self.full_garden_plan))}/"
            f"{len(self.full_garden_plan)}")
        stats["Completed"] = f"{len(self.done_tiles)}/{len(self.farm_tiles)}"
        index = min(self.full_garden_index, len(self.full_garden_plan) - 1)
        current_plan_target = self.full_garden_plan[index]
        stats["Target"] = f"{current_plan_target}"
        if self.algorithm_name == "IDS":
            depth_by_target = stats.get("Depth limit by target", {})
            if isinstance(depth_by_target, dict):
                current_limit = depth_by_target.get(
                    current_plan_target, stats.get("Depth limit"))
                stats["Current depth limit"] = current_limit
        if self.algorithm_name == "UCS" and "Target g(n)" in stats:
            precomputed_g = stats["Target g(n)"].get(
                current_plan_target, 0)
            cumulative_path = (
                self.mode1_ucs_path_length_total
                + self.mode1_ucs_current_leg_length)
            cumulative_extra = (
                self.mode1_ucs_extra_cost_total
                + self.mode1_ucs_current_leg_extra_cost)
            cumulative_g = cumulative_path + cumulative_extra
            completed_g = (
                self.mode1_ucs_path_length_total
                + self.mode1_ucs_extra_cost_total)
            leg_g = (
                self.mode1_ucs_current_leg_length
                + self.mode1_ucs_current_leg_extra_cost)
            stats["PQ target g(n)"] = precomputed_g
            stats["Route g(n)"] = cumulative_g
            stats["Completed g(n)"] = completed_g
            stats["Leg g(n)"] = leg_g
            stats["g(n)"] = cumulative_g
            stats["Cost"] = cumulative_g
            stats["Cumulative path length"] = cumulative_path
            stats["Terrain + condition cost"] = cumulative_extra
        self.stats = stats


    def _build_full_garden_plan(self, remaining, start):
        self._clear_full_garden_plan()
        if not remaining:
            return []

        if self.mode == 3:
            plan, scores, stats = algorithms.build_local_search_full_plan(
                self.algorithm_name, remaining, start, self.tile_conditions,
                self._heuristic)
            self.hc_scores = scores
            self.hc_current_score = scores.get(plan[0], 0) if plan else 0
            self.hc_best_neighbor_score = self.hc_current_score
            self.full_garden_plan = list(plan)
            self.full_garden_stats = stats
        elif self.mode == 1:
            if self.algorithm_name == "DFS":
                remaining_set = set(remaining)
                self._build_dfs_walk(start, remaining_set)
                planned = set()
                current_index = 0
                current_parent = start

                for index, tile in enumerate(self.dfs_walk):
                    if tile not in remaining_set or tile in planned:
                        continue

                    planned_path = list(
                        self.dfs_walk[current_index + 1:index + 1])
                    if not self._path_is_contiguous(
                            current_parent, planned_path):
                        continue

                    self.full_garden_plan.append(tile)
                    self.full_garden_leg_paths[tile] = (
                        current_parent, planned_path)
                    planned.add(tile)
                    current_index = index
                    current_parent = tile

                self.full_garden_stats = {
                    "Algorithm": "DFS",
                    "Planning mode": "Full physical DFS traversal",
                    "Path source": "precomputed DFS walk",
                    "Plan targets": len(self.full_garden_plan),
                    "Unreachable": len(remaining_set - planned),
                    "Nodes explored": len(self.dfs_explored_order),
                    "Path length": sum(
                        len(planned_leg[1])
                        for planned_leg
                        in self.full_garden_leg_paths.values()),
                    "Stack max": self.dfs_stack_max,
                }
                self.bfs_explored = set(self.dfs_explored_order)
            elif self.algorithm_name == "UCS":
                self._prepare_mode1_ucs_costs()
            if self.algorithm_name != "DFS":
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
                self.bfs_explored = set(explored)
                if self.algorithm_name == "UCS":
                    self.full_garden_stats["Priority"] = "g(n)"
                    self.full_garden_stats["UCS rule"] = (
                        "UCS priority = g(n), expand 4-neighbor like A*, no h(n)")
                    self.full_garden_stats["g(n) formula"] = (
                        "path_length + terrain_cost + condition_cost")
        elif self.mode == 2:
            return self._build_mode2_frontier_full_plan(remaining, start)
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
                if self.mode == 1 and self.algorithm_name == "DFS":
                    planned_leg = self.full_garden_leg_paths.get(target)
                    if planned_leg is None:
                        return None
                    planned_parent, planned_path = planned_leg
                    if planned_parent != start:
                        return None
                    self.chosen_target_path = list(planned_path)
                elif self.mode == 2:
                    planned_leg = self.full_garden_leg_paths.get(target)
                    if planned_leg is None or planned_leg[0] != start:
                        self._build_mode2_frontier_full_plan(remaining, start)
                        if not self.full_garden_plan:
                            return None
                        return self._next_full_garden_target(remaining, start)

                    (
                        planned_parent, planned_path, planned_fgh,
                        planned_h_before, planned_g_leg
                    ) = planned_leg
                    self.chosen_target_path = list(planned_path)
                    planned_stats = self.full_garden_leg_stats.get(target)
                    if planned_stats is not None:
                        stats_step_cache = None
                        if len(planned_stats) == 6:
                            (
                                stats_parent, stats_path,
                                stats_explored, stats_fgh, stats_key,
                                stats_step_cache
                            ) = planned_stats
                        else:
                            (
                                stats_parent, stats_path,
                                stats_explored, stats_fgh, stats_key
                            ) = planned_stats
                        if stats_parent == start:
                            self._mode2_set_leg_runtime_stats(
                                stats_parent, target, stats_path,
                                stats_explored, stats_fgh, stats_key,
                                stats_step_cache)
                    else:
                        display_fgh = (
                            planned_g_leg + planned_h_before,
                            planned_g_leg,
                            planned_h_before)
                        fallback_key = (
                            planned_h_before, 0, planned_h_before, target)
                        self._mode2_set_leg_runtime_stats(
                            planned_parent, target, planned_path, set(),
                            display_fgh,
                            fallback_key)
                if not (self.mode == 1 and self.algorithm_name == "UCS"):
                    self._apply_full_plan_stats()
                return target
            self.full_garden_index += 1

        if remaining_set:
            if (self.mode == 3
                    and self.algorithm_name in (
                        "Hill Climbing", "Annealing", "Local Beam")
                    and str(self.full_garden_stats.get(
                        "Stop rule", "")).startswith("No ")):
                return None
            # The old plan was exhausted or invalidated by online discovery.
            self._build_full_garden_plan(remaining, start)
            if not self.full_garden_plan:
                return None
            return self._next_full_garden_target(remaining, start)
        return None


    def _choose_target(self, remaining, start):
        if self.mode == 3:
            return self._local_search_choose(remaining, start)
        if self.mode == 4:
            return self._choose_mode4_target(remaining, start)
        if self.mode == 1:
            return self._search_first_target(start, remaining)
        if self.mode == 2:
            return None
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
        self._tick_algorithm_timer(dt)

        if self.mode == 6:
            self._update_mode6(dt)
            return

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

        # --- Mode 5: CSP animation phases ---
        if self.mode == 5:
            if self.csp_phase == "analyze":
                self._update_csp_analyze(dt)
                return
            if self.csp_phase == "replay":
                self._update_csp_replay(dt)
                return
            # csp_phase == "idle": cho START
            if self.csp_phase == "idle":
                self.player.direction.update(0, 0)
                return

        # --- Stuck detection: náº¿u Ä‘ang MOVE mĂ  position khĂ´ng Ä‘á»•i > 1.5s ---
        if (self.state == "MOVE"
                and self.mode != 2
                and not (self.mode == 4
                         and self.algorithm_name == "AND-OR Search")):
            cur_pos = (round(self.player.pos.x), round(self.player.pos.y))
            if not hasattr(self, '_stuck_pos'):
                self._stuck_pos = cur_pos
                self._stuck_timer = 0.0
            elif cur_pos == self._stuck_pos:
                self._stuck_timer += dt
                if self._stuck_timer > 1.5:
                    if (self.mode == 4
                            and self.algorithm_name in (
                                "Online A*", "Online BFS",
                                "Belief-State BFS", "Belief BFS", "Belief A*")):
                        self.path = []
                        self.current_target = None
                        self.state = "CHOOSE_TARGET"
                        self.replan_count += 1
                        self.mode4_phase = "REPLAN"
                        self._update_belief_state()
                        self._stuck_pos = None
                        self._stuck_timer = 0.0
                        self.wait_time = 0.1
                        self.message = (
                            f"{self.algorithm_name}: bi ket, "
                            "clear path va re-plan "
                            f"(lan {self.replan_count})")
                        return
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
            if self.mode == 4:
                self.mode4_active_step = None
            if (self.mode == 6
                    and (self.enemy_target is not None
                         or self._enemy_is_destroying())):
                self.player.direction.update(0, 0)
                self.message = "Cho doi thu hoan tat luot hien tai"
                return
            remaining = [t for t in self.farm_tiles
                         if t not in self.done_tiles
                         and t not in self.discovered_blocked
                         and t not in self.enemy_done_tiles
                         and (self.mode != 6 or t != self.enemy_target)]
            if not remaining:
                self.state = "DONE"
                if self.mode == 4:
                    self.mode4_phase = "DONE"
                    self._update_belief_state()
                done_count = len(self.done_tiles)
                enemy_count = len(self.enemy_done_tiles)
                if self.mode == 6:
                    protected_score, destroyed_score = (
                        algorithms.mode6_state_scores(
                            self.done_tiles, self.enemy_done_tiles,
                            self.mode6_crop_profiles))
                    if protected_score > destroyed_score:
                        result = "AGRI-1 thang"
                    elif protected_score < destroyed_score:
                        result = "Qua thang"
                    else:
                        result = "Hoa"
                    self.mode6_phase = "DONE"
                    if self.enemy_tile:
                        enemy_tile = (
                            int(round(self.enemy_tile[0])),
                            int(round(self.enemy_tile[1])))
                        self.enemy_retreat_target = self._enemy_retreat_tile(enemy_tile)
                    self.message = (
                        f"{result}! Bao ve {protected_score} - "
                        f"Bi pha {destroyed_score}")
                    self._update_mode6_step_hud(phase="DONE")
                else:
                    self.message = (
                        f"Hoan thanh khu {self.mode}: "
                        f"{self.algorithm_name}.")
                return

            start = self._world_to_tile(self.player.rect.center)
            if self.mode == 4:
                center = self._tile_center(start)
                self.player.pos.update(center)
                self.player.rect.center = (round(center.x), round(center.y))
                self.player.hitbox.center = self.player.rect.center
                self.player.direction.update(0, 0)

            # Mode 4: má»Ÿ rá»™ng táº§m nhĂ¬n
            if self.mode == 4:
                self._expand_vision(start)

            if self.mode == 1 and self.algorithm_name == "IDS":
                if self.ids_search_start is None:
                    self._begin_ids_search(start, remaining)
                    return
                self.ids_search_goals.update(remaining)
                target = self._next_ids_found_target(remaining)
                if target is not None:
                    self._move_to_ids_target(target)
                    return
                self.state = "IDS_SEARCH"
                self.player.direction.update(0, 0)
                self.message = (
                    f"IDS tiep tuc depth {self.ids_depth_limit} "
                    f"tu {self.ids_search_start}")
                return

            target = self._next_full_garden_target(remaining, start)
            if target is None:
                if (self.mode == 4
                        and self.algorithm_name == "Online A*"
                        and remaining
                        and self.mode4_online_astar_open):
                    self.state = "CHOOSE_TARGET"
                    self.mode4_phase = "EXPLORE"
                    self._update_belief_state()
                    self.message = (
                        f"{self.algorithm_name}: expand 1 node trong PQ")
                    return
                if (self.mode == 4
                        and self.algorithm_name in (
                            "Online BFS", "Belief-State BFS",
                            "Belief BFS", "Belief A*")
                        and remaining
                        and self.mode4_online_bfs_queue):
                    self.state = "CHOOSE_TARGET"
                    self.mode4_phase = "EXPLORE"
                    self._update_belief_state()
                    self.message = f"{self.algorithm_name}: lay target tu FIFO queue"
                    return
                self.state = "DONE"
                if self.mode == 4:
                    self.mode4_phase = "DONE"
                    self._update_belief_state()
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
            if self.mode == 6:
                self.mode6_phase = "PLAN_STEP"
                if (self.path
                        and not self._path_is_contiguous(start, self.path)):
                    self.path = []
                if self.enemy_target is not None:
                    enemy_start = (
                        int(round(self.enemy_tile[0])),
                        int(round(self.enemy_tile[1])))
                    self.enemy_path = self._plan_enemy_path(
                        self.enemy_target)
                    enemy_at_target = enemy_start == self.enemy_target
                    enemy_path_valid = (
                        not self.enemy_path
                        or self._path_is_contiguous(
                            enemy_start, self.enemy_path))
                    if (not enemy_path_valid
                            or (not self.enemy_path
                                and not enemy_at_target)):
                        self.enemy_target = None
                        self.enemy_path = []
            if self.mode == 1 and self.algorithm_name == "UCS":
                self.mode1_ucs_current_leg_length = len(self.path)
                self.mode1_ucs_current_leg_extra_cost = (
                    self._mode1_ucs_path_extra_cost(self.path))
            if (self.mode == 2 and self.path
                    and not self._path_is_contiguous(start, self.path)):
                self.chosen_target_path = None
                self.path = self._path_for_mode(start, target)
                if (self.path
                        and not self._path_is_contiguous(start, self.path)):
                    self.path = []
                    self.mode2_current_leg_g = 0
                    self.mode2_current_leg_weighted_g = 0.0
                    self.mode2_current_leg_start = None
                    self.mode2_current_leg_target = None
                    self.mode2_current_step_stats = []
                    self.mode2_current_step_index = 0
            if (self.mode == 4 and self.path
                    and not self._path_is_contiguous(start, self.path)):
                self.path = []
                self.current_target = None
                self.state = "CHOOSE_TARGET"
                self.replan_count += 1
                self.mode4_phase = "REPLAN"
                self._update_belief_state()
                self.message = (
                    f"Online: path khong lien ke, "
                    f"chon lai (lan {self.replan_count})")
                self.wait_time = 0.2
                return

            if not self.path and start != target:
                if self.mode == 4:
                    self.path = []
                    self.current_target = None
                    self.state = "CHOOSE_TARGET"
                    self.replan_count += 1
                    self.mode4_phase = "REPLAN"
                    self._update_belief_state()
                    self._clear_full_garden_plan()
                    self.message = (
                        f"Online: o {target} bi chan, "
                        f"re-plan (lan {self.replan_count})")
                    self.wait_time = 0.1
                    return
                if self.mode == 6:
                    self.discovered_blocked.add(target)
                    self.current_target = None
                    self.path = []
                    self.state = "CHOOSE_TARGET"
                    self.mode6_phase = "PLAN_STEP"
                    self.message = (
                        f"AGRI-1 khong co duong toi {target}; "
                        "khong bao ve ao")
                    self._update_mode6_step_hud(
                        phase=self.mode6_phase, current=start,
                        next_tile=None)
                    return
                self.done_tiles.add(target)
                if self.mode == 2 and target == self.mode2_current_leg_target:
                    self.mode2_current_leg_g = 0
                    self.mode2_current_leg_weighted_g = 0.0
                    self.mode2_current_leg_start = None
                    self.mode2_current_leg_target = None
                    self.mode2_current_step_stats = []
                    self.mode2_current_step_index = 0
                self._advance_full_garden_plan(target)
                self.message = f"Bo qua {target}: khong co duong."
                return

            self.state = "MOVE"
            if self.mode == 6:
                self.mode6_move_total = len(self.path)
                self.mode6_move_done = 0
                self._update_mode6_step_hud(
                    phase="MODE6_MOVE",
                    current=start,
                    next_tile=self.path[0] if self.path else target,
                )
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
                if self.mode == 4:
                    current_tile = self._world_to_tile(
                        self.player.rect.center)
                    if (self.current_target is not None
                            and current_tile == self.current_target):
                        center = self._tile_center(current_tile)
                        self.player.pos.update(center)
                        self.player.rect.center = (
                            round(center.x), round(center.y))
                        self.player.hitbox.center = self.player.rect.center
                        self.player.direction.update(0, 0)
                        self.state = self._state_after_arrival(
                            self.current_target)
                        return
                    self.current_target = None
                    self.path = []
                    self.state = "CHOOSE_TARGET"
                    self.replan_count += 1
                    self.mode4_phase = "REPLAN"
                    self._update_belief_state()
                    self.message = (
                        f"{self.algorithm_name}: path rong, "
                        f"re-plan lan {self.replan_count}")
                    self.wait_time = 0.1
                    return
                self.state = self._state_after_arrival(self.current_target)
                return

            next_tile = self.path[0]

            # Mode 4: phĂ¡t hiá»‡n váº­t cáº£n áº©n khi Ä‘i gáº§n
            if self.mode == 4:
                current_tile = self._world_to_tile(self.player.rect.center)
                self._update_mode4_online_astar_hud(
                    current_tile, next_tile, self.current_target)
                step_is_invalid = False
                if self.mode4_active_step is None:
                    step_is_invalid = (
                        self._heuristic(current_tile, next_tile) != 1)
                    if not step_is_invalid:
                        self.mode4_active_step = (current_tile, next_tile)
                else:
                    _, active_next = self.mode4_active_step
                    step_is_invalid = active_next != next_tile
                if step_is_invalid:
                    self.path = []
                    self.current_target = None
                    self.state = "CHOOSE_TARGET"
                    self.mode4_active_step = None
                    self.replan_count += 1
                    self.mode4_phase = "REPLAN"
                    self._update_belief_state()
                    self.message = (
                        f"Online: buoc khong hop le toi {next_tile}, "
                        f"chon lai (lan {self.replan_count})")
                    self.wait_time = 0.2
                    return
                self._expand_vision(current_tile)
                if next_tile in self.discovered_blocked:
                    self.path = []
                    self.current_target = None
                    self.state = "CHOOSE_TARGET"
                    self.replan_count += 1
                    self.mode4_phase = "REPLAN"
                    self._update_belief_state()
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
                    self.mode4_active_step = None
                    self.mode4_completed_g += 1
                    self._finish_and_or_outcome()
                return

            reached = self._set_player_direction_to(
                self._tile_center(next_tile))
            if reached:
                if self.mode == 4:
                    center = self._tile_center(next_tile)
                    self.player.pos.update(center)
                    self.player.rect.center = (
                        round(center.x), round(center.y))
                    self.player.hitbox.center = self.player.rect.center
                    self.player.direction.update(0, 0)
                    self.mode4_active_step = None
                    self.mode4_completed_g += 1
                self.path.pop(0)
                if self.mode == 6:
                    self.mode6_move_done += 1
                    self._update_mode6_step_hud(
                        phase=(
                            "MODE6_MOVE"),
                        current=next_tile,
                        next_tile=(
                            self.path[0] if self.path
                            else self.current_target),
                    )
                if self.mode == 2:
                    self._mode2_apply_precomputed_step_stats(
                        self.mode2_current_step_index + 1)
                # Mode 4: tiáº¿p tá»¥c má»Ÿ rá»™ng táº§m nhĂ¬n khi di chuyá»ƒn
                if self.mode == 4:
                    self._expand_vision(next_tile)
                    if self.algorithm_name in (
                            "Online A*", "Online BFS",
                            "Belief-State BFS", "Belief BFS", "Belief A*"):
                        reached_mode4_target = (
                            next_tile == self.current_target
                            or self.algorithm_name in (
                                "Belief-State BFS", "Belief BFS", "Belief A*"))
                        if (reached_mode4_target
                                and next_tile in self.farm_tiles
                                and next_tile not in self.done_tiles
                                and next_tile not in self.discovered_blocked
                                and next_tile not in self.enemy_done_tiles):
                            self.path = []
                            self.current_target = next_tile
                            self.state = self._state_after_arrival(next_tile)
                            self.message = (
                                f"{self.algorithm_name}: xu ly cay tai {next_tile}")
                        else:
                            self._update_belief_state()
                            self.message = (
                                f"{self.algorithm_name}: tiep tuc theo path tu PQ")
                            if not self.path:
                                self.current_target = None
                                self.state = "CHOOSE_TARGET"
                                self.mode4_phase = "EXPLORE"
            return

        # --- FIX_CROW / WATER_RESCUE / TREAT_PEST / HOE / PLANT / WATER ---
        current_tile = self._world_to_tile(self.player.rect.center)
        action_tile = self.current_target if self.current_target is not None else current_tile
        target_pos = self._tile_center(action_tile)
        self.player.status = "down_idle"
        self.player.direction.update(0, 0)

        if self.state == "FIX_CROW":
            self._update_mode6_step_hud(
                phase="MODE6_RESOLVE",
                current=action_tile,
                next_tile=action_tile,
            )
            self._update_mode6_fix(action_tile)
            return

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

    def draw_bg(self, surface, offset):
        """Khong con dung de ve map overlays nua (xem ghi chu o draw_fg).
        Giu lai ham nay (no-op) de tuong thich nguoc voi noi da goi
        draw_bg() truoc khi ve sprites, tranh phai sua lai level.py."""
        pass


    def draw_fg(self, surface, offset):
        """Ve TOAN BO overlay AI (task markers, fog, CSP highlight, duong
        di...) CONG VOI speech bubble/"BACKTRACK!"/panel. Tat ca deu phai
        ve SAU sprites/player nhu code goc, vi cac asset minh hoa o dat
        (weeds, pest, storm_debris, CSP tile...) duoc blit truc tiep len
        surface (khong phai sprite cua level) nen se bi ground/soil sprite
        cua level VE SAU de len che mat neu goi truoc customize_draw."""
        self._draw_map_overlays(surface, offset)
        self._draw_mode6_lightning_rod(surface, offset)
        self._draw_lightning_effects(surface, offset)
        self._draw_map_foreground(surface, offset)
        self._draw_panel(surface)
        if (self.mode == 4
                and self.algorithm_name in (
                    "Belief-State BFS", "Belief BFS", "Belief A*")):
            self._draw_belief_minimap(surface)


    def draw(self, surface, offset):
        """Tuong thich nguoc: ve toan bo (bg no-op + fg). Cac noi goi moi
        nen goi draw_fg() mot lan SAU khi sprites/player da ve xong."""
        self.draw_bg(surface, offset)
        self.draw_fg(surface, offset)

    # ------------------------------------------------------ map overlays

    def _draw_resolved_asset(self, surface, offset, tile):
        condition = self._condition_for_tile(tile)
        if (self.mode == 6
                or condition in ("dry", "critical", "pest", "crow")):
            self._blit_tile_asset(surface, offset, tile, "healthy_plant", y_offset=-6)


    def _draw_task_tile_marker(self, surface, offset, tile):
        world = self._tile_center(tile)
        sx = int(world.x - offset.x)
        sy = int(world.y - offset.y)
        marker = pygame.Surface((TILE_SIZE - 10, TILE_SIZE - 10),
                                pygame.SRCALPHA)
        is_mode6_protected = self.mode == 6 and tile in self.done_tiles
        fill_color = (
            (55, 190, 90, 105)
            if is_mode6_protected else Colors.TASK_MARKER_FILL)
        border_color = (
            (75, 235, 115)
            if is_mode6_protected else Colors.TASK_MARKER_BORDER)
        corner_color = (
            (120, 255, 145)
            if is_mode6_protected else Colors.TASK_MARKER_CORNER)
        pygame.draw.rect(marker, fill_color, marker.get_rect(),
                         border_radius=6)
        surface.blit(marker, (sx - TILE_SIZE // 2 + 5,
                              sy - TILE_SIZE // 2 + 5))
        rect = pygame.Rect(0, 0, TILE_SIZE - 10, TILE_SIZE - 10)
        rect.center = (sx, sy)
        pygame.draw.rect(surface, border_color, rect, 2 if is_mode6_protected else 1,
                         border_radius=5)
        corner = 10
        color = corner_color
        pygame.draw.line(surface, color, rect.topleft, (rect.left + corner, rect.top), 2)
        pygame.draw.line(surface, color, rect.topleft, (rect.left, rect.top + corner), 2)
        pygame.draw.line(surface, color, rect.topright, (rect.right - corner, rect.top), 2)
        pygame.draw.line(surface, color, rect.topright, (rect.right, rect.top + corner), 2)
        pygame.draw.line(surface, color, rect.bottomleft, (rect.left + corner, rect.bottom), 2)
        pygame.draw.line(surface, color, rect.bottomleft, (rect.left, rect.bottom - corner), 2)
        pygame.draw.line(surface, color, rect.bottomright, (rect.right - corner, rect.bottom), 2)
        pygame.draw.line(surface, color, rect.bottomright, (rect.right, rect.bottom - corner), 2)


    def _draw_map_overlays(self, surface, offset):
        """Váº½ thĂ´ng tin thuáº­t toĂ¡n lĂªn map (explored nodes, scores, fog, enemy...)."""
        for tile in self.farm_tiles:
            if self.mode == 4 and tile not in self.explored_tiles:
                continue
            self._draw_task_tile_marker(surface, offset, tile)

        if self.mode in (1, 2, 3, 4, 6):
            for tile in self.farm_tiles:
                if tile in self.enemy_done_tiles:
                    if self.mode == 6:
                        self._blit_tile_asset(
                            surface, offset, tile,
                            "crow_wilted_plant", y_offset=-6)
                    continue
                if self.mode == 4 and tile in self.hidden_blocked:
                    continue
                if self.mode == 4 and tile not in self.explored_tiles:
                    continue
                if tile in self.done_tiles:
                    self._draw_resolved_asset(surface, offset, tile)
                    continue
                condition = self._condition_for_tile(tile)
                asset_key = (
                    self._mode6_condition_asset(condition)
                    if self.mode == 6 else self._condition_asset(condition))
                if asset_key is not None:
                    self._blit_tile_asset(
                        surface, offset, tile, asset_key, y_offset=-6)
                if condition == "pest":
                    worm_offset = -10 if self.mode == 6 else 2
                    self._blit_tile_asset(
                        surface, offset, tile, "worm", y_offset=worm_offset)
                if False and self.mode == 6:
                    profile = self.mode6_crop_profiles.get(
                        tile, algorithms.MODE6_TREE_STATUS["dry"])
                    status_code = {
                        "dry": "D",
                        "dead": "X",
                        "pest": "P",
                        "critical": "C",
                    }.get(condition, "?")
                    label = pygame.font.Font(None, 1).render(
                        f"{status_code} V{profile['value']} "
                        f"R{profile['risk'] * 100:.0f}",
                        True, Colors.TEXT_PRIMARY)
                    world = self._tile_center(tile)
                    label_rect = label.get_rect(
                        center=(int(world.x - offset.x),
                                int(world.y - offset.y + 22)))
                    surface.blit(label, label_rect)

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
                # To mau theo score moi cua Mode 3: thap = uu tien cao, cao = it uu tien hon
                visible_scores = [
                    v for t, v in self.hc_scores.items()
                    if t not in self.done_tiles
                ]
                min_score = min(visible_scores) if visible_scores else 0
                max_score = max(visible_scores) if visible_scores else 1
                ratio = (score - min_score) / max(1, max_score - min_score)
                ratio = max(0, min(1, ratio))
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
                self._blit_tile_asset(surface, offset, tile, "rocks")
                world = self._tile_center(tile)
                rect = pygame.Rect(0, 0, TILE_SIZE - 8, TILE_SIZE - 8)
                rect.center = (int(world.x - offset.x),
                                int(world.y - offset.y))
                font_sm = pygame.font.Font(None, 18)
                txt = font_sm.render("BLOCK", True, Colors.BLOCK_LABEL)
                surface.blit(txt, (rect.x + 4, rect.y + 20))

            # Váº½ Ă´ block áº©n chÆ°a phĂ¡t hiá»‡n (nháº¹, debug)
        # --- Mode 5: CSP visualization ---
        if self.mode == 5:
            font_sm = pygame.font.Font(None, 18)

            # Phase analyze: highlight cac cap rang buoc lan luot
            if self.csp_phase == "analyze":
                for tile in self.farm_tiles:
                    world = self._tile_center(tile)
                    sx = int(world.x - offset.x)
                    sy = int(world.y - offset.y)
                    s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4), pygame.SRCALPHA)
                    s.fill((60, 80, 120, 60))
                    surface.blit(s, (sx - TILE_SIZE // 2 + 2, sy - TILE_SIZE // 2 + 2))
                idx = self.csp_analyze_pair_index
                if idx < len(self.csp_analyze_pairs):
                    for tile in self.csp_analyze_pairs[idx]:
                        world = self._tile_center(tile)
                        sx = int(world.x - offset.x)
                        sy = int(world.y - offset.y)
                        s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4), pygame.SRCALPHA)
                        s.fill((255, 220, 50, 130))
                        pygame.draw.rect(s, (255, 220, 50, 220), s.get_rect(), 2, border_radius=6)
                        surface.blit(s, (sx - TILE_SIZE // 2 + 2, sy - TILE_SIZE // 2 + 2))
                        txt = font_sm.render("?", True, (255, 255, 200))
                        surface.blit(txt, (sx - 4, sy - 7))
                for prev_idx in range(max(0, idx - 3), idx):
                    if prev_idx < len(self.csp_analyze_pairs):
                        for tile in self.csp_analyze_pairs[prev_idx]:
                            world = self._tile_center(tile)
                            sx = int(world.x - offset.x)
                            sy = int(world.y - offset.y)
                            s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4), pygame.SRCALPHA)
                            s.fill((255, 200, 50, 50))
                            surface.blit(s, (sx - TILE_SIZE // 2 + 2, sy - TILE_SIZE // 2 + 2))

            # Phase replay: hien tung buoc assign/try/backtrack
            elif self.csp_phase == "replay":
                for tile in self.farm_tiles:
                    world = self._tile_center(tile)
                    sx = int(world.x - offset.x)
                    sy = int(world.y - offset.y)
                    s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4), pygame.SRCALPHA)
                    s.fill((40, 40, 60, 80))
                    surface.blit(s, (sx - TILE_SIZE // 2 + 2, sy - TILE_SIZE // 2 + 2))

                # Highlight o da assign chinh thuc (xanh la)
                for tile, crop in self.csp_assigned.items():
                    if tile in self.csp_flash_timers:
                        continue
                    world = self._tile_center(tile)
                    sx = int(world.x - offset.x)
                    sy = int(world.y - offset.y)
                    s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4), pygame.SRCALPHA)
                    s.fill((40, 160, 70, 110))
                    pygame.draw.rect(s, (80, 220, 110, 200), s.get_rect(), 2, border_radius=6)
                    surface.blit(s, (sx - TILE_SIZE // 2 + 2, sy - TILE_SIZE // 2 + 2))
                    self._draw_crop_icon(surface, sx, sy, crop)
                    label = {"corn": "C", "tomato": "T", "wheat": "W", "carrot": "R"}
                    txt = font_sm.render(label.get(crop, "?"), True, (200, 255, 200))
                    surface.blit(txt, (sx - 4, sy + 16))

                # Highlight o xung dot (do dam) - de len tren assigned tiles
                for tile in self.csp_conflict_tiles:
                    world = self._tile_center(tile)
                    sx = int(world.x - offset.x)
                    sy = int(world.y - offset.y)
                    s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4), pygame.SRCALPHA)
                    s.fill((255, 30, 30, 180))
                    pygame.draw.rect(s, (255, 80, 80, 255), s.get_rect(), 3, border_radius=6)
                    surface.blit(s, (sx - TILE_SIZE // 2 + 2, sy - TILE_SIZE // 2 + 2))
                    # Hien thi crop icon va nhan X de biet la xung dot
                    crop = self.csp_assigned.get(tile)
                    if crop:
                        self._draw_crop_icon(surface, sx, sy, crop)
                        label = {"corn": "C", "tomato": "T", "wheat": "W", "carrot": "R"}
                        lbl = font_sm.render(label.get(crop, "?"), True, (255, 180, 180))
                        surface.blit(lbl, (sx - 4, sy + 16))
                    # Dau "!" bao hieu xung dot
                    txt = font_sm.render("!", True, (255, 240, 60))
                    surface.blit(txt, (sx - 3, sy - 8))

                # Highlight o dang xu ly hien tai (vang)
                if self.csp_current_var and self.csp_current_var not in self.csp_flash_timers:
                    tile = self.csp_current_var
                    world = self._tile_center(tile)
                    sx = int(world.x - offset.x)
                    sy = int(world.y - offset.y)
                    s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4), pygame.SRCALPHA)
                    if self.csp_current_event in ("think", "try"):
                        s.fill((220, 190, 30, 120))
                        pygame.draw.rect(s, (255, 220, 60, 200), s.get_rect(), 2, border_radius=6)
                        surface.blit(s, (sx - TILE_SIZE // 2 + 2, sy - TILE_SIZE // 2 + 2))
                        if self.csp_current_value:
                            tmp = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                            self._draw_crop_icon(tmp, TILE_SIZE//2, TILE_SIZE//2, self.csp_current_value)
                            tmp.set_alpha(130)
                            surface.blit(tmp, (sx - TILE_SIZE//2, sy - TILE_SIZE//2))
                        txt = font_sm.render("?", True, (255, 255, 180))
                        surface.blit(txt, (sx - 4, sy - 8))

                # Highlight o dang backtrack (do nhan)
                for tile in list(self.csp_flash_timers):
                    world = self._tile_center(tile)
                    sx = int(world.x - offset.x)
                    sy = int(world.y - offset.y)
                    s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4), pygame.SRCALPHA)
                    s.fill((220, 30, 30, 190))
                    pygame.draw.rect(s, (255, 80, 80, 255), s.get_rect(), 3, border_radius=6)
                    # 1) Ve o mau truoc
                    surface.blit(s, (sx - TILE_SIZE // 2 + 2, sy - TILE_SIZE // 2 + 2))
                    # 2) Ve icon cay de len tren o mau (lay gia tri tu csp_display)
                    display_entry = self.csp_display.get(tile)
                    if display_entry:
                        flash_crop = display_entry[0] if isinstance(display_entry, tuple) else display_entry
                        if flash_crop:
                            self._draw_crop_icon(surface, sx, sy, flash_crop)
                    # 3) Ve chu "X" sau cung de luon hien tren icon
                    txt = font_sm.render("X", True, (255, 200, 200))
                    surface.blit(txt, (sx - 4, sy - 8))

                # Domain hien tai cua cac o chua gan. FC/AC-3 cat domain
                # am tham; lop text nay cho thay tap gia tri con lai.
                domain_label = {
                    "corn": "C", "tomato": "T", "wheat": "W", "carrot": "R"
                }
                font_domain = pygame.font.Font(None, 16)
                for tile, values in self.csp_domains.items():
                    if tile in self.csp_assigned:
                        continue
                    if not values:
                        text = "{}"
                    else:
                        text = "".join(domain_label.get(crop, "?") for crop in values)
                    world = self._tile_center(tile)
                    sx = int(world.x - offset.x)
                    sy = int(world.y - offset.y)
                    txt = font_domain.render(text, True, (235, 240, 255))
                    pad_x = 3
                    bg = pygame.Rect(
                        sx - txt.get_width() // 2 - pad_x,
                        sy + TILE_SIZE // 2 - 17,
                        txt.get_width() + pad_x * 2,
                        txt.get_height() + 2,
                    )
                    bg_surf = pygame.Surface(bg.size, pygame.SRCALPHA)
                    bg_surf.fill((25, 28, 42, 170))
                    surface.blit(bg_surf, bg.topleft)
                    surface.blit(txt, (bg.x + pad_x, bg.y + 1))

                # (Speech bubble + "BACKTRACK!" flash text duoc ve trong
                # _draw_map_foreground, SAU khi sprites/player da ve, de
                # khong bi nhan vat che mat - xem draw_fg().)

            # Phase ready/thuc thi: hien cac o da assign trong qua trinh replay
            elif self.csp_phase in ("ready", "idle") and self.csp_assigned:
                for tile, crop in self.csp_assigned.items():
                    world = self._tile_center(tile)
                    sx = int(world.x - offset.x)
                    sy = int(world.y - offset.y)
                    s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4), pygame.SRCALPHA)
                    if tile in self.csp_seeded_tiles:
                        s.fill((40, 140, 60, 120))
                        pygame.draw.rect(s, (80, 200, 100, 180), s.get_rect(), 2, border_radius=6)
                    else:
                        s.fill((90, 60, 40, 80))
                        pygame.draw.rect(s, (120, 80, 50, 150), s.get_rect(), 2, border_radius=6)
                    surface.blit(s, (sx - TILE_SIZE // 2 + 2, sy - TILE_SIZE // 2 + 2))
                    self._draw_crop_icon(surface, sx, sy, crop)
                    label = {"corn": "C", "tomato": "T", "wheat": "W", "carrot": "R"}
                    txt = font_sm.render(label.get(crop, "?"), True, (255, 255, 200))
                    surface.blit(txt, (sx - 4, sy + 16))
            # Fallback: neu csp_assigned trong ma seed_plan co (truong hop idle chua chay)
            elif self.csp_phase in ("ready", "idle") and self.seed_plan and not self.csp_assigned:
                for tile, crop in self.seed_plan.items():
                    world = self._tile_center(tile)
                    sx = int(world.x - offset.x)
                    sy = int(world.y - offset.y)
                    s = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4), pygame.SRCALPHA)
                    if tile in self.csp_seeded_tiles:
                        s.fill((40, 140, 60, 120))
                        pygame.draw.rect(s, (80, 200, 100, 180), s.get_rect(), 2, border_radius=6)
                    else:
                        s.fill((90, 60, 40, 80))
                        pygame.draw.rect(s, (120, 80, 50, 150), s.get_rect(), 2, border_radius=6)
                    surface.blit(s, (sx - TILE_SIZE // 2 + 2, sy - TILE_SIZE // 2 + 2))
                    self._draw_crop_icon(surface, sx, sy, crop)
                    label = {"corn": "C", "tomato": "T", "wheat": "W", "carrot": "R"}
                    txt = font_sm.render(label.get(crop, "?"), True, (255, 255, 200))
                    surface.blit(txt, (sx - 4, sy + 16))

        # --- Mode 6: Enemy + enemy path ---
        if (self.mode == 6 and self.enemy_tile
                and self.algorithm_name != "Expectimax"):
            world = self._tile_center(self.enemy_tile)
            ex = int(world.x - offset.x)
            ey = int(world.y - offset.y)
            # ThĂ¢n enemy
            enemy_image = self._enemy_image()
            if enemy_image:
                enemy_rect = enemy_image.get_rect(center=(ex, ey - 4))
                surface.blit(enemy_image, enemy_rect)
            else:
                pygame.draw.circle(surface, Colors.ENEMY_FALLBACK, (ex, ey), 20)
                pygame.draw.circle(surface, Colors.ENEMY_FALLBACK_OUTLINE, (ex, ey), 20, 3)
            # Enemy target line
            if self.enemy_target:
                route = [(ex, ey)]
                for route_tile in self.enemy_path:
                    tw = self._tile_center(route_tile)
                    route.append((
                        int(tw.x - offset.x),
                        int(tw.y - offset.y),
                    ))
                if len(route) > 1:
                    pygame.draw.lines(
                        surface, Colors.ENEMY, False, route, 2)
                tw = self._tile_center(self.enemy_target)
                tx = int(tw.x - offset.x)
                ty = int(tw.y - offset.y)
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

        if self.mode == 6 and self.algorithm_name == "Expectimax":
            if self.enemy_target:
                w = self._tile_center(self.enemy_target)
                tx = int(w.x - offset.x)
                ty = int(w.y - offset.y)
                pygame.draw.circle(surface, Colors.TEXT_WARNING,
                                   (tx, ty), 18, 3)
            for tile in self.enemy_done_tiles:
                w = self._tile_center(tile)
                rect = pygame.Rect(0, 0, TILE_SIZE - 6, TILE_SIZE - 6)
                rect.center = (int(w.x - offset.x), int(w.y - offset.y))
                pygame.draw.rect(surface, Colors.ENEMY_FALLBACK, rect, 3,
                                 border_radius=4)

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

    # -------------------------------------------------- map foreground

    def _draw_map_foreground(self, surface, offset):
        """Ve cac overlay PHAI nam TREN sprites/player: speech bubble va
        "BACKTRACK!" flash text phia tren dau robot trong luc CSP replay.
        Goi sau khi all_sprites.customize_draw(player) da ve xong, neu
        khong robot se de len che mat bubble/text."""
        if self.mode != 5 or self.csp_phase != "replay":
            return

        # Speech bubble theo robot - mat khi robot bat dau di sang tile moi
        if self.csp_speech and self.csp_walk_state != "moving":
            px = int(self.player.rect.centerx - offset.x)
            py = int(self.player.rect.top - offset.y) - 8
            font_sp = pygame.font.Font(None, 22)
            txt_surf = font_sp.render(self.csp_speech, True, (30, 30, 30))
            bubble_w = txt_surf.get_width() + 16
            bubble_h = txt_surf.get_height() + 10
            bubble_rect = pygame.Rect(px - bubble_w//2, py - bubble_h, bubble_w, bubble_h)
            # Mau bubble theo event
            if self.csp_current_event == "backtrack":
                bubble_color = (255, 100, 100, 230)
                border_color_b = (220, 50, 50)
            elif self.csp_current_event == "assign":
                bubble_color = (100, 220, 120, 230)
                border_color_b = (60, 180, 80)
            else:
                bubble_color = (240, 230, 160, 230)
                border_color_b = (180, 160, 60)
            bubble_surf = pygame.Surface((bubble_w, bubble_h), pygame.SRCALPHA)
            bubble_surf.fill(bubble_color)
            surface.blit(bubble_surf, bubble_rect.topleft)
            pygame.draw.rect(surface, border_color_b, bubble_rect, 2, border_radius=6)
            # Duoi bubble
            tail_x = px
            pygame.draw.polygon(surface, border_color_b, [
                (tail_x - 5, bubble_rect.bottom),
                (tail_x + 5, bubble_rect.bottom),
                (tail_x, bubble_rect.bottom + 7)
            ])
            surface.blit(txt_surf, (bubble_rect.x + 8, bubble_rect.y + 5))

        # "BACKTRACK!" floating text — chi hien voi CSP Backtracking, khong hien voi Min Conflict
        if self.csp_backtrack_flash > 0 and self.algorithm_name != "Min Conflict":
            px = int(self.player.rect.centerx - offset.x)
            py = int(self.player.rect.top - offset.y) - 60
            font_bt = pygame.font.Font(None, 30)
            alpha = int(255 * min(1.0, self.csp_backtrack_flash / 0.3))
            bt_surf = font_bt.render(f"BACKTRACK #{self.csp_backtracks_live}!", True, (255, 80, 80))
            bt_surf.set_alpha(alpha)
            surface.blit(bt_surf, (px - bt_surf.get_width()//2, py))

    # ------------------------------------------------------------ panel

    def _filtered_hud_stats(self):
        common = {
            1: ("Target", "Current node", "Node depth", "Stack size",
                "Depth limit", "This iteration", "Found this depth",
                "Found depth", "Total expansions", "Unique explored",
                "g(n)", "Route g(n)", "Leg g(n)", "Completed g(n)",
                "PQ target g(n)", "Cost", "Cumulative path length",
                "Terrain + condition cost",
                "Path length", "Nodes explored", "Queue max",
                "Stack max", "Current depth limit"),
            2: ("Target", "f(n)", "g(n)", "h(n)", "h distance",
                "h condition", "h formula", "f limit", "Path length",
                "Nodes explored", "This iteration", "Total expansions"),
            3: ("Target", "Target type", "Current score", "Hill steps",
                "Beam width", "Generated", "Temperature", "Delta",
                "Accepted worse", "Rejected", "Restarts"),
            4: ("Current target", "Current node", "Next node", "Path length",
                "Nodes explored", "Queue max", "Replanned", "Blocks found"),
            5: ("Variables", "Backtracks", "Arc checks", "Conflicts",
                "Assigned"),
        }
        keys = (
            ("Target", "f(n)", "g(n)", "g steps", "Turns",
             "Turn penalty", "h(n)")
            if self.mode == 2 and self.algorithm_name == "A*_v2"
            else common.get(self.mode, ()))
        items = [(key, self.stats[key]) for key in keys if key in self.stats]
        if self.mode in common:
            return items[:7]

        hidden = {
            "Algorithm", "Robot Algorithm", "Planning mode", "Phase",
            "Selection reason", "Online policy", "Value order", "Domain",
        }
        return [
            (key, value) for key, value in self.stats.items()
            if key not in hidden
        ][:5]


    def _draw_panel(self, surface):
        """Draw the left control panel with algorithm selection and run buttons."""
        font = pygame.font.Font(None, 22)
        font_title = pygame.font.Font(None, 26)
        font_small = pygame.font.Font(None, 20)
        border_color = MODE_COLORS.get(self.mode, (200, 200, 200))

        panel_width = 390
        selectable_modes = (1, 2, 3, 4, 5, 6)
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
        y = draw_text(
            f"TIME: {self._format_elapsed_time(self._current_algorithm_elapsed())}",
            x, y, (255, 218, 70), font)
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
            start_rect = pygame.Rect(x, y, 82, 36)
            pause_rect = pygame.Rect(x + 88, y, 82, 36)
            reset_rect = pygame.Rect(x + 176, y, 82, 36)
            manual_rect = pygame.Rect(x + 264, y, 82, 36)
            manual_mode = getattr(self, "manual_control", False)
            self.control_buttons["start"] = draw_button(
                "start", start_rect, "START",
                selected=self.is_running and not manual_mode)
            self.control_buttons["pause"] = draw_button(
                "pause", pause_rect, "PAUSE",
                selected=not self.is_running and not manual_mode)
            self.control_buttons["reset"] = draw_button(
                "reset", reset_rect, "RESET", danger=True)
            self.control_buttons["manual"] = draw_button(
                "manual", manual_rect,
                "MANUAL" if not manual_mode else "AI",
                selected=manual_mode)
            y += 50
        else:
            y = draw_text(f"Thuat toan: {self.algorithm_name}", x, y,
                          border_color, font)
            y += 8

        pygame.draw.line(surface, Colors.PANEL_SEPARATOR,
                         (x, y), (panel.right - 15, y))
        y += 12
        objective = self._objective_snapshot()
        if getattr(self, "manual_control", False):
            status = "Thu cong"
        else:
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
            plan_label = "ROUTE PLAN" if self.mode == 2 else "PLAN"
            y = draw_text(
                f"{plan_label}: {len(self.full_garden_plan)} target | "
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
        if self.mode == 4:
            belief = self.belief_state or {}
            y = draw_text(
                f"Phase: {belief.get('phase', self.mode4_phase)} | "
                f"Frontier: {belief.get('frontier', 0)} | "
                f"Unknown: {belief.get('unknown', 0)}",
                x, y, Colors.TEXT_STAT, font_small)
            if self.algorithm_name in ("Belief-State BFS", "AND-OR Search"):
                y = draw_text(
                    f"Belief: {belief.get('belief_worlds', 0)} worlds | "
                    f"Obs: {belief.get('belief_before', 0)} -> "
                    f"{belief.get('belief_after', 0)}",
                    x, y, Colors.TEXT_STAT, font_small)
                if belief.get("inconsistent", False):
                    y = draw_text(
                        "Belief inconsistent: 0 worlds",
                        x, y, Colors.TEXT_WARNING, font_small)
            else:
                y = draw_text(
                    f"Known: {belief.get('known_free', 0)} | "
                    f"Blocked: {belief.get('discovered_blocked', 0)} | "
                    f"Replan: {belief.get('replans', self.replan_count)}",
                    x, y, Colors.TEXT_STAT, font_small)

        # --- Mode 5: CSP HUD realtime ---
        if self.mode == 5 and self.csp_phase == "replay":
            crop_vn = {"corn": "Ngo", "tomato": "Ca chua",
                       "wheat": "Lua mi", "carrot": "Ca rot"}
            phase_color = Colors.TEXT_WARNING
            if self.csp_current_event == "assign":
                phase_color = Colors.TEXT_SUCCESS
            elif self.csp_current_event == "backtrack":
                phase_color = (255, 100, 100)
            if self.csp_current_var:
                y = draw_text(
                    f"O dang xu ly: {self.csp_current_var}",
                    x, y, Colors.TEXT_STAT, font_small)
            if self.csp_current_value:
                action = {"think": "Dang nghi", "try": "Trong thu",
                          "assign": "Da trong",
                          "backtrack": "Quay lui"}.get(self.csp_current_event, "")
                y = draw_text(
                    f"{action}: {crop_vn.get(self.csp_current_value, '')}",
                    x, y, phase_color, font_small)
            y = draw_text(
                f"Da gan: {len(self.csp_assigned)}/{len(self.farm_tiles)} o",
                x, y, Colors.TEXT_SUCCESS, font_small)
            if self.csp_backtracks_live > 0:
                bt_label = "Conflicts fixed" if self.algorithm_name == "Min Conflict" else "Backtrack"
                y = draw_text(
                    f"{bt_label}: {self.csp_backtracks_live} lan",
                    x, y, (255, 120, 120), font_small)
            # Progress bar CSP
            bar_rect2 = pygame.Rect(x, y + 2, panel_width - 30, 7)
            pygame.draw.rect(surface, (40, 40, 60), bar_rect2, border_radius=3)
            total_steps = max(len(self.csp_steps), 1)
            fill2 = int(bar_rect2.width * self.csp_replay_index / total_steps)
            if fill2 > 0:
                pygame.draw.rect(surface,
                    (255, 200, 80) if self.csp_backtracks_live > 0 else border_color,
                    pygame.Rect(bar_rect2.x, bar_rect2.y, fill2, bar_rect2.height),
                    border_radius=3)
            y += 18

        if self.mode == 6:
            protected_score, destroyed_score = algorithms.mode6_state_scores(
                self.done_tiles, self.enemy_done_tiles,
                self.mode6_crop_profiles)
            y = draw_text(
                f"Da sua: {objective['completed']} | "
                f"Bi pha huy: {objective['lost']} | "
                f"Con lai: {objective['remaining']}",
                x, y, Colors.TEXT_WARNING, font_small)
            node_flow = {
                "Minimax": "MAX -> MIN",
                "Alpha-Beta": "MAX -> MIN + prune",
                "Expectimax": "MAX -> CHANCE",
                "Expectiminimax": "MAX -> MIN -> CHANCE",
            }.get(self.algorithm_name, "MAX -> MIN")
            depth = self.mode6_tree_details.get(
                "depth", algorithms.MODE6_SEARCH_DEPTH)
            y = draw_text(
                f"Luot {self.mode6_turn}: {self.mode6_phase} | Depth: {depth}",
                          x, y, Colors.TEXT_STAT, font_small)
            if (self.mode6_robot_repair_tile is not None
                    or self.mode6_enemy_threat_tile is not None):
                repair_text = (
                    f"Repair: {self.mode6_robot_repair_tile} "
                    f"({self.mode6_robot_repair_turns}/2)"
                    if self.mode6_robot_repair_tile is not None
                    else "Repair: -")
                threat_text = (
                    f"Threat: {self.mode6_enemy_threat_tile} "
                    f"({self.mode6_enemy_threat_turns}/2)"
                    if self.mode6_enemy_threat_tile is not None
                    else "Threat: -")
                y = draw_text(
                    f"{repair_text} | {threat_text}",
                    x, y, Colors.TEXT_WARNING, font_small)
            y = draw_text(f"Mo hinh: {node_flow}",
                          x, y, Colors.TEXT_STAT, font_small)
            robot_action = self.mode6_tree_details.get(
                "robot_action", self.mode6_robot_action)
            enemy_action = self.mode6_tree_details.get(
                "enemy_action", self.mode6_enemy_action)
            robot_target = self.mode6_tree_details.get(
                "robot_target", self.mode6_robot_target)
            enemy_target = self.mode6_tree_details.get(
                "enemy_target", self.mode6_enemy_target)
            target_text = (
                f"Target: {robot_target} | Weather theo risk"
                if self.algorithm_name == "Expectimax"
                else f"Target: {robot_target} | Qua: {enemy_target}")
            y = draw_text(target_text, x, y, Colors.TEXT_PRIMARY, font_small)
            action_text = (f"Action AGRI: {robot_action} | CHANCE"
                           if self.algorithm_name == "Expectimax"
                           else f"Action AGRI: {robot_action} | Qua: {enemy_action}")
            y = draw_text(action_text, x, y, Colors.TEXT_STAT, font_small)
            y = draw_text(
                f"Bao ve: {protected_score} | Bi pha: {destroyed_score} | "
                f"Eval: {self.minimax_value:.1f}",
                x, y, Colors.TEXT_SUCCESS, font_small)
            if self.algorithm_name == "Alpha-Beta":
                y = draw_text(
                    f"Alpha: {self.alpha_beta_info['alpha']} | "
                    f"Beta: {self.alpha_beta_info['beta']} | "
                    f"Pruned: {self.alpha_beta_info['pruned']}",
                    x, y, (255, 120, 120), font_small)
            elif self.algorithm_name in ("Expectimax", "Expectiminimax"):
                y = draw_text(
                    "CHANCE 5s: an toan 50% | set 20% | thu loi 30%",
                    x, y, Colors.TEXT_STAT, font_small)
                if self.mode6_chance_event:
                    y = draw_text(
                        f"Ket qua: {self.mode6_chance_event} "
                        f"({self.mode6_chance_probability * 100:.0f}%)",
                        x, y, Colors.TEXT_WARNING, font_small)
            node_text = (
                f"Nodes: {self.stats.get('Nodes evaluated', 0)} | "
                f"MAX: {self.stats.get('MAX nodes', 0)}")
            if self.algorithm_name != "Expectimax":
                node_text += f" | MIN: {self.stats.get('MIN nodes', 0)}"
            y = draw_text(node_text, x, y, Colors.TEXT_STAT, font_small)

        if self.stats and self.mode != 6:
            y = draw_text("THONG KE", x, y, Colors.TEXT_MUTED, font_small)
            stats_bottom = panel.bottom - 42
            for key, val in self._filtered_hud_stats():
                if y + font_small.get_height() + 6 > stats_bottom:
                    break
                y = draw_text(f"{key}: {val}", x, y, Colors.TEXT_STAT, font_small)

        if self.mode in (1, 2, 3):
            if getattr(self, "manual_control", False):
                hint = "Manual: WASD di chuyen | Space sua cay | M ve AI"
            else:
                hint = "Click nut hoac dung Q/E de doi thuat toan | M choi thu cong"
        else:
            if getattr(self, "manual_control", False):
                hint = "Manual: WASD di chuyen | Space sua cay | M ve AI"
            else:
                hint = "Nhan phim 1-6 de doi khu nhiem vu | M choi thu cong"
        draw_text(hint, x, panel.bottom - 30, Colors.TEXT_MUTED, font_small)
