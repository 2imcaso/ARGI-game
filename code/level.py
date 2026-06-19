from random import randint

import os
import xml.etree.ElementTree as ET
import pygame 
from pytmx.util_pygame import load_pygame

from settings import *
from player import Player
from overlay import Overlay
from sprites import Generic, Water, WildFlower, Tree, Interaction, Particle
from support import *
from transition import Transition
from soil import SoilLayer
from sky import Rain, Sky
from menu import Menu
from ai_controller import FarmAIController
from design_tokens import Colors, MODE_COLORS
import algorithms

class Level:
	HOUSE_SHIFT_TILES = (0, 5)
	HOUSE_ORIGINAL_BOUNDS = (20, 21, 27, 26)  # left, top, right, bottom in TMX tiles
	SOLID_DECORATION_KINDS = {'sign', 'stake', 'stump', 'stump_small', 'bush', 'mushroom', 'storm_debris', 'crow_damage'}
	COMPACT_DECORATION_KINDS = {'path', 'mist', 'puddle', 'mud'}
	OVERVIEW_DURATION = 3.0
	OVERVIEW_PADDING_TILES = 3
	MAP_LINKS = (
		(1, 2), (2, 3),
		(1, 4), (4, 5), (5, 6),
		(2, 5), (3, 6),
	)
	AREA_BOARD_TILES = {
		1: (12, 14),
		2: (24, 14),
		3: (32, 22),
		4: (10, 26),
		5: (21, 27),
		6: (30, 27),
	}

	# ----------------------------------------------------------------
	# 6 khu demo rieng tren cung nong trai.
	# Moi ngay AGRI-1 duoc dieu toi mot khu sau bao khac nhau.
	# Táº¥t cáº£ tile Ä‘Æ°á»£c kiá»ƒm tra: náº±m trong Farmable layer,
	# khĂ´ng trĂ¹ng Collision layer, spawn reachable tá»›i toĂ n bá»™ farm.
	# ----------------------------------------------------------------
	MODE_CONFIGS = {
		# Day 1 - BFS: front-left recovery plot.
		1: {
			'area_name': 'Khu 1 - Vuon truoc trai',
			'spawn_tile': (11, 18),
			'obstacles': [],
			'terrain_costs': {
				(13,18): 12,
				(15,16): 8,
				(16,19): 10,
			},
			'extra_walkable_tiles': [
				(11,17),(12,17),
			],
			'tiles': [
				(13,15),(14,15),(15,15),(16,15),(17,15),
				(13,16),(14,16),(15,16),(16,16),(17,16),
				(13,17),(14,17),(15,17),(16,17),(17,17),
				(13,18),(14,18),(15,18),(16,18),(17,18),
				(13,19),(14,19),(15,19),(16,19),(17,19),
			],
			'decorations': [
				('sign', 12, 16), ('flower', 18, 15),
			],
		},
		# Day 2 - A*: warehouse lane, with storm debris blocking the route.
		2: {
			'area_name': 'Khu 2 - Loi vao nha kho',
			'spawn_tile': (18, 18),
			'obstacles': [
				(19,15),(20,15),(21,15),(22,15),(23,15),(24,15),(25,15),(26,15),
				(19,21),(20,21),(21,21),(22,21),(23,21),(24,21),(25,21),(26,21),
				(21,16),(21,17),(21,18),
				(23,18),(23,19),(23,20),
				(25,16),(25,17),(25,18),
			],
			'obstacle_kinds': {
				(19,15): 'fallen_log', (20,15): 'storm_debris',
				(21,15): 'rock_pile', (22,15): 'bush',
				(23,15): 'storm_debris', (24,15): 'broken_crate',
				(25,15): 'fallen_log', (26,15): 'storm_debris',
				(19,21): 'storm_debris', (20,21): 'stump_small',
				(21,21): 'broken_crate', (22,21): 'rock_pile',
				(23,21): 'fallen_log', (24,21): 'storm_debris',
				(25,21): 'bush', (26,21): 'rock_pile',
				(21,16): 'fallen_log', (21,17): 'bush', (21,18): 'rock_pile',
				(23,18): 'broken_crate', (23,19): 'stump_small', (23,20): 'storm_debris',
				(25,16): 'rock_pile', (25,17): 'fallen_log', (25,18): 'broken_crate',
			},
			'tiles': [
				(27,16),(28,16),(29,16),(30,16),(31,16),
				(27,17),(28,17),(29,17),(30,17),(31,17),
				(27,18),(28,18),(29,18),(30,18),(31,18),
				(27,19),(28,19),(29,19),(30,19),(31,19),
				(27,20),(28,20),(29,20),(30,20),(31,20),
			],
			'decorations': [
				('sign', 18, 16),
				('puddle', 20, 19), ('puddle', 27, 21), ('puddle', 32, 18),
				('bush', 19, 17), ('stump_small', 32, 17),
				('stake', 27, 16), ('stake', 32, 16),
				('flower', 32, 20),
			],
		},
		# Day 3 - Local Search: east orchard 5x5, no obstacles.
		3: {
			'area_name': 'Khu 3 - Vuon cay phia dong',
			'spawn_tile': (34, 24),
			'obstacles': [],
			'tiles': [
				(29,22),(30,22),(31,22),(32,22),(33,22),
				(29,23),(30,23),(31,23),(32,23),(33,23),
				(29,24),(30,24),(31,24),(32,24),(33,24),
				(29,25),(30,25),(31,25),(32,25),(33,25),
				(29,26),(30,26),(31,26),(32,26),(33,26),
			],
			'decorations': [
				('puddle', 28, 24),
				('flower', 35, 23),
				('stake', 28, 21), ('stake', 35, 21),
			],
		},
		# Day 4 - Online Search: hidden walls force discovery and backtracking.
		4: {
			'area_name': 'Khu 4 - Bai dat suong mu',
			'spawn_tile': (18, 31),
			'obstacles': [],
			'tiles': [
				(11,27),(12,27),(13,27),(14,27),(15,27),(16,27),(17,27),
				(11,28),(12,28),(13,28),(14,28),(15,28),(16,28),(17,28),
				(11,29),(12,29),(13,29),(14,29),(15,29),(16,29),(17,29),
				(11,30),(12,30),(13,30),(14,30),(15,30),(16,30),(17,30),
				(11,31),(12,31),(13,31),(14,31),(15,31),(16,31),(17,31),
				(11,32),(12,32),(13,32),(14,32),(15,32),(16,32),(17,32),
			],
			'hidden_blocks': [
				(13,27),(13,28),(13,29),(13,30),
				(15,29),(15,30),(15,31),(15,32),
				(16,27),
			],
			'decorations': [
				('mist', 12, 26), ('mist', 15, 26), ('mist', 18, 30),
				('puddle', 14, 33), ('sign', 10, 27),
			],
		},
		# Day 5 - CSP Backtracking: central replant grid.
		5: {
			'area_name': 'Khu 5 - O quy hoach trung tam',
			'spawn_tile': (20, 30),
			'obstacles': [],
			'tiles': [
				(21,28),(22,28),(23,28),(24,28),(25,28),
				(21,29),(22,29),(23,29),(24,29),(25,29),
				(21,30),(22,30),(23,30),(24,30),(25,30),
				(21,31),(22,31),(23,31),(24,31),(25,31),
				(21,32),(22,32),(23,32),(24,32),(25,32),
			],
			'decorations': [
				('stake', 20, 28), ('stake', 26, 28),
				('stake', 20, 33), ('stake', 26, 33),
				('sign', 20, 29),
			],
		},
		# Day 6 - Minimax: south-east protected rows.
		6: {
			'area_name': 'Khu 6 - Hang cay can bao ve',
			'spawn_tile': (29, 31),
			'obstacles': [],
			'tiles': [
				(30,28),(31,28),(32,28),(33,28),(34,28),
				(30,29),(31,29),(32,29),(33,29),(34,29),
				(30,30),(31,30),(32,30),(33,30),(34,30),
				(30,31),(31,31),(32,31),(33,31),(34,31),
				(30,32),(31,32),(32,32),(33,32),(34,32),
			],
			'enemy_spawn': (38, 31),
			'decorations': [
				('crow', 37, 28), ('crow_damage', 36, 32),
				('sign', 29, 28), ('stump', 38, 32),
			],
		},
	}

	def __init__(self):

		# get the display surface
		self.display_surface = pygame.display.get_surface()

		# sprite groups
		self.all_sprites = CameraGroup()
		self.collision_sprites = pygame.sprite.Group()
		self.tree_sprites = pygame.sprite.Group()
		self.interaction_sprites = pygame.sprite.Group()

		# Dynamic obstacles (per-mode)
		self.dynamic_obstacles = []
		self.dynamic_decorations = []
		self.link_sprites = []
		self.link_tiles = set()
		self.overview_timer = 0.0
		self.overview_active = False
		self.overview_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
		self.has_shown_overview = False

		self.soil_layer = SoilLayer(self.all_sprites, self.collision_sprites)
		self.setup()
		self.overlay = Overlay(self.player)
		self.current_mode = 1
		self.selected_algorithms = {
			1: algorithms.UNINFORMED_ALGORITHMS[0],
			2: algorithms.INFORMED_ALGORITHMS[-1],
			3: algorithms.LOCAL_ALGORITHMS[1],
			4: algorithms.ONLINE_ALGORITHMS[0],
			5: algorithms.CSP_ALGORITHMS[0],
		}

		# Khá»Ÿi táº¡o mode 1
		self._init_mode(1)

		self.transition = Transition(self.reset, self.player)

		# sky
		self.rain = Rain(self.all_sprites)
		self.raining = True
		self.soil_layer.raining = self.raining
		self.sky = Sky()
		self.use_day_night = False

		# shop disabled for one-map AI demo
		self.menu = Menu(self.player, self.toggle_shop)
		self.shop_active = False

		# music
		self.success = pygame.mixer.Sound("../audio/success.wav")
		self.success.set_volume(0.01)
		self.music_bg = pygame.mixer.Sound("../audio/bg.mp3")
		self.music_bg.play(loops = -1)
		self.music_bg.set_volume(0.01)

	
	def setup(self):
		tmx_data = load_pygame("../data/map.tmx")
		
		# House layers are intentionally skipped for the six-area AI demo.
		# The story treats each mode as a different damaged field section, so
		# removing the house gives the farm plots more visual space.
		
		# Fence
		for x, y, surf in tmx_data.get_layer_by_name('Fence').tiles():
			Generic((x * TILE_SIZE, y * TILE_SIZE), surf, [self.all_sprites, self.collision_sprites])

		# water
		water_frames = import_folder("../graphics/water")
		for x, y, surf in tmx_data.get_layer_by_name('Water').tiles():
			Water((x * TILE_SIZE, y * TILE_SIZE), water_frames, self.all_sprites)

		# trees
		for obj in tmx_data.get_layer_by_name("Trees"):
			Tree((obj.x, obj.y), obj.image, [self.all_sprites, self.collision_sprites, self.tree_sprites], obj.name, self.player_add)

		# wildflowers
		for obj in tmx_data.get_layer_by_name('Decoration'):
			WildFlower((obj.x, obj.y), obj.image, [self.all_sprites, self.collision_sprites])

		# collision tiles
		for x, y, surf in tmx_data.get_layer_by_name('Collision').tiles():
			if self._is_house_tile(x, y):
				continue
			pos = (x * TILE_SIZE, y * TILE_SIZE)
			Generic(pos, pygame.Surface((TILE_SIZE, TILE_SIZE)), self.collision_sprites)
		
		# Player â€” spawn máº·c Ä‘á»‹nh, sáº½ di chuyá»ƒn trong _init_mode
		farm_spawn_pos = ((43 * TILE_SIZE) + (TILE_SIZE // 2), (16 * TILE_SIZE) + (TILE_SIZE // 2))
		for obj in tmx_data.get_layer_by_name('Player'):
			if obj.name == 'Start':
				self.player = Player(farm_spawn_pos, self.all_sprites, self.collision_sprites, self.tree_sprites, self.interaction_sprites, self.soil_layer, self.toggle_shop)
			
			if obj.name == 'Bed':
				pos = self._shift_house_object_pos(obj.x, obj.y)
				Interaction(pos, (obj.width, obj.height), self.interaction_sprites, obj.name)

			if obj.name == 'Trader':
				Interaction((obj.x, obj.y), (obj.width, obj.height), self.interaction_sprites, obj.name)

		Generic((0, 0),
	  	 		pygame.image.load("../graphics/world/ground.png").convert_alpha(),
				groups = self.all_sprites,
				z = LAYERS['ground'])
		self._cover_baked_path_tiles()

	def _cover_baked_path_tiles(self):
		"""Hide path tiles that are baked into graphics/world/ground.png."""
		map_path = os.path.join("..", "data", "map.tmx")
		root = ET.parse(map_path).getroot()
		width = int(root.attrib["width"])

		first_gids = []
		path_first_gid = None
		for tileset in root.findall("tileset"):
			first_gid = int(tileset.attrib["firstgid"])
			first_gids.append(first_gid)
			source = tileset.attrib.get("source", "")
			name = tileset.attrib.get("name", "")
			if "Paths" in source or name == "Paths":
				path_first_gid = first_gid

		if path_first_gid is None:
			return

		next_gids = [gid for gid in first_gids if gid > path_first_gid]
		path_last_gid = min(next_gids) if next_gids else 10 ** 9

		grass_sheet = pygame.image.load("../graphics/environment/Grass.png").convert_alpha()
		grass_tile = grass_sheet.subsurface(pygame.Rect(0, 0, TILE_SIZE, TILE_SIZE)).copy()

		for layer in root.findall("layer"):
			data = layer.find("data")
			if data is None or data.text is None:
				continue
			values = [int(value) for value in data.text.replace("\n", "").split(",") if value.strip()]
			for index, raw_gid in enumerate(values):
				gid = raw_gid & 0x1FFFFFFF
				if path_first_gid <= gid < path_last_gid:
					x = index % width
					y = index // width
					Generic(
						(x * TILE_SIZE, y * TILE_SIZE),
						grass_tile.copy(),
						self.all_sprites,
						z=LAYERS['soil'])

	def _is_house_tile(self, x, y):
		left, top, right, bottom = self.HOUSE_ORIGINAL_BOUNDS
		return left <= x <= right and top <= y <= bottom

	def _shift_house_pos(self, x, y):
		dx, dy = self.HOUSE_SHIFT_TILES
		return ((x + dx) * TILE_SIZE, (y + dy) * TILE_SIZE)

	def _shift_house_object_pos(self, x, y):
		tx = int(x // TILE_SIZE)
		ty = int(y // TILE_SIZE)
		if self._is_house_tile(tx, ty):
			dx, dy = self.HOUSE_SHIFT_TILES
			return (x + dx * TILE_SIZE, y + dy * TILE_SIZE)
		return (x, y)

	# ----------------------------------------------------------------
	# Táº¡o farm tá»« danh sĂ¡ch tile tuá»³ Ă½ (thay vĂ¬ lÆ°á»›i vuĂ´ng)
	# ----------------------------------------------------------------
	def clear_demo_soil(self):
		# XĂ³a plant sprites
		for plant in self.soil_layer.plant_sprites.sprites():
			if plant in self.collision_sprites:
				self.collision_sprites.remove(plant)
			plant.kill()

		# XĂ³a soil/water sprites khá»i mĂ n hĂ¬nh
		for group in [self.soil_layer.water_sprites, self.soil_layer.soil_sprites]:
			for sprite in group.sprites():
				sprite.kill()

		# XĂ³a TOĂ€N Bá»˜ flag â€” ká»ƒ cáº£ 'F' Ä‘á»ƒ map cÅ© khĂ´ng cĂ²n hiá»‡n
		for row in self.soil_layer.grid:
			for cell in row:
				cell.clear()

		# KhĂ´i phá»¥c 'F' gá»‘c tá»« TMX (Farmable layer) Ä‘á»ƒ ná»n váº«n Ä‘Ăºng
		from pytmx.util_pygame import load_pygame
		tmx = load_pygame("../data/map.tmx")
		for x, y, surf in tmx.get_layer_by_name('Farmable').tiles():
			if 0 <= y < len(self.soil_layer.grid) and 0 <= x < len(self.soil_layer.grid[y]):
				self.soil_layer.grid[y][x].append('F')

		# Rebuild hit_rects, dá»n soil_sprites group
		self.soil_layer.create_hit_rects()
		self.soil_layer.soil_sprites.empty()

	def create_farm_from_tiles(self, tiles):
		"""Táº¡o farm tá»« danh sĂ¡ch tile (x, y) tuá»³ Ă½."""
		farm_tiles = []
		for (x, y) in tiles:
			if 0 <= y < len(self.soil_layer.grid) and 0 <= x < len(self.soil_layer.grid[y]):
				cell = self.soil_layer.grid[y][x]
				if 'F' not in cell:
					cell.append('F')
				farm_tiles.append((x, y))
		self.soil_layer.create_hit_rects()
		return farm_tiles

	def _task_tiles_for_config(self, cfg):
		blocked_tasks = set(cfg.get('terrain_costs', {}).keys())
		return [tile for tile in cfg['tiles'] if tile not in blocked_tasks]

	def _area_anchor(self, mode):
		tiles = self.MODE_CONFIGS[mode]['tiles']
		x = round(sum(tile[0] for tile in tiles) / len(tiles))
		y = round(sum(tile[1] for tile in tiles) / len(tiles))
		return (x, y)

	def _line_tiles(self, start, end):
		x, y = start
		ex, ey = end
		step_x = 1 if ex >= x else -1
		while x != ex:
			yield (x, y)
			x += step_x
		step_y = 1 if ey >= y else -1
		while y != ey:
			yield (x, y)
			y += step_y
		yield (ex, ey)

	def _build_link_tiles(self):
		link_tiles = set()
		for start_mode, end_mode in self.MAP_LINKS:
			start = self._area_anchor(start_mode)
			end = self._area_anchor(end_mode)
			link_tiles.update(self._line_tiles(start, end))
		for cfg in self.MODE_CONFIGS.values():
			link_tiles.difference_update(cfg['tiles'])
			link_tiles.discard(cfg['spawn_tile'])
			link_tiles.difference_update(cfg.get('obstacles', []))
			link_tiles.difference_update(cfg.get('hidden_blocks', []))
		return link_tiles

	def _create_link_sprites(self):
		for sprite in self.link_sprites:
			sprite.kill()
		self.link_sprites.clear()
		self.link_tiles = self._build_link_tiles()
		for tx, ty in sorted(self.link_tiles):
			surf, layer = self._make_decor_surface('path')
			path = Generic(
				(tx * TILE_SIZE, ty * TILE_SIZE), surf,
				self.all_sprites,
				z=layer)
			self.link_sprites.append(path)

	def _all_demo_tiles(self):
		tiles = set(self.link_tiles)
		for cfg in self.MODE_CONFIGS.values():
			tiles.update(cfg['tiles'])
			tiles.update(cfg.get('terrain_costs', {}).keys())
		return sorted(tiles)

	def _make_area_ground_surface(self, mode):
		surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
		mode_color = MODE_COLORS.get(mode, (210, 180, 120))
		pygame.draw.rect(surf, (118, 78, 42, 88), (5, 5, 54, 54), border_radius=7)
		pygame.draw.rect(surf, (164, 112, 62, 58), (10, 10, 44, 44), border_radius=5)
		pygame.draw.rect(surf, (*mode_color, 96), (6, 6, 52, 52), 2, border_radius=7)
		pygame.draw.line(surf, (85, 54, 31, 56), (13, 47), (49, 15), 2)
		pygame.draw.line(surf, (198, 152, 92, 48), (16, 16), (46, 48), 2)
		return surf

	def _create_all_area_ground_visuals(self):
		for mode, cfg in self.MODE_CONFIGS.items():
			surf = self._make_area_ground_surface(mode)
			tiles = set(cfg['tiles'])
			tiles.update(cfg.get('terrain_costs', {}).keys())
			for tx, ty in tiles:
				ground = Generic(
					(tx * TILE_SIZE, ty * TILE_SIZE),
					surf.copy(),
					self.all_sprites,
					z=LAYERS['soil'])
				self.dynamic_decorations.append(ground)

	def _create_terrain_cost_visuals(self):
		for cfg in self.MODE_CONFIGS.values():
			for tx, ty in cfg.get('terrain_costs', {}):
				surf, layer = self._make_decor_surface('mud')
				terrain = Generic(
					(tx * TILE_SIZE, ty * TILE_SIZE),
					surf,
					self.all_sprites,
					z=layer)
				self.dynamic_decorations.append(terrain)

	def _calculate_overview_rect(self):
		points = []
		for cfg in self.MODE_CONFIGS.values():
			points.extend(cfg['tiles'])
			points.append(cfg['spawn_tile'])
			points.extend(cfg.get('obstacles', []))
			points.extend(cfg.get('hidden_blocks', []))
			enemy_spawn = cfg.get('enemy_spawn')
			if enemy_spawn:
				points.append(enemy_spawn)
		points.extend(self.link_tiles)
		if not points:
			return pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)

		padding = self.OVERVIEW_PADDING_TILES
		min_x = min(tile[0] for tile in points) - padding
		max_x = max(tile[0] for tile in points) + padding
		min_y = min(tile[1] for tile in points) - padding
		max_y = max(tile[1] for tile in points) + padding
		return pygame.Rect(
			min_x * TILE_SIZE,
			min_y * TILE_SIZE,
			(max_x - min_x + 1) * TILE_SIZE,
			(max_y - min_y + 1) * TILE_SIZE)

	def _start_overview(self):
		self.overview_timer = self.OVERVIEW_DURATION
		self.overview_active = True
		self.all_sprites.set_overview(self.overview_rect)

	def show_overview_once(self):
		if self.has_shown_overview:
			return
		self.has_shown_overview = True
		self._start_overview()

	# ----------------------------------------------------------------
	# Dynamic obstacles
	# ----------------------------------------------------------------
	def _clear_mode_obstacles(self):
		for sprite in self.dynamic_obstacles:
			sprite.kill()
		self.dynamic_obstacles.clear()
		for sprite in self.dynamic_decorations:
			sprite.kill()
		self.dynamic_decorations.clear()

	def _create_world_decorations(self):
		self._create_all_area_ground_visuals()
		self._create_terrain_cost_visuals()
		for mode in self.MODE_CONFIGS:
			self._create_mode_obstacles(mode)
			self._create_mode_decorations(mode)
		self._create_area_name_boards()

	def _load_scaled_asset(self, rel_path, size=(TILE_SIZE, TILE_SIZE)):
		path = os.path.join("..", rel_path)
		if not os.path.exists(path):
			return None
		image = pygame.image.load(path).convert_alpha()
		return pygame.transform.scale(image, size)

	def _make_decor_surface(self, kind):
		asset_paths = {
			'storm_debris': 'graphics/ai_visuals/storm_debris.png',
			'mud': 'graphics/ai_visuals/mud.png',
			'crow': 'graphics/ai_visuals/crow.png',
			'crow_damage': 'graphics/ai_visuals/crow_damage.png',
			'flower': 'graphics/objects/sunflower.png',
			'stump': 'graphics/objects/stump_medium.png',
			'stump_small': 'graphics/objects/stump_small.png',
			'bush': 'graphics/objects/bush.png',
			'mushroom': 'graphics/objects/mushroom.png',
		}
		if kind in asset_paths:
			image = self._load_scaled_asset(asset_paths[kind])
			if image:
				return image, LAYERS['main']

		surf = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
		if kind == 'puddle':
			pygame.draw.ellipse(surf, Colors.PUDDLE, (10, 26, 44, 18))
			pygame.draw.ellipse(surf, Colors.PUDDLE_HIGHLIGHT, (20, 30, 20, 6))
			pygame.draw.rect(surf, Colors.PUDDLE_SHADOW, (16, 38, 32, 3))
			return surf, LAYERS['soil water']
		if kind == 'mist':
			pygame.draw.rect(surf, Colors.MIST_DECOR, (10, 22, 42, 8), border_radius=4)
			pygame.draw.rect(surf, Colors.MIST_DECOR_DIM, (18, 34, 36, 7), border_radius=3)
			pygame.draw.rect(surf, Colors.MIST_DECOR_FAINT, (6, 44, 30, 6), border_radius=3)
			return surf, LAYERS['main']
		if kind == 'path':
			pygame.draw.rect(surf, Colors.PATH_DECOR, (4, 27, 56, 14), border_radius=5)
			pygame.draw.rect(surf, Colors.PATH_DECOR_LIGHT, (12, 30, 10, 5))
			pygame.draw.rect(surf, Colors.PATH_DECOR_DARK, (38, 32, 8, 4))
			pygame.draw.rect(surf, Colors.PATH_DECOR_HIGHLIGHT, (48, 29, 6, 4))
			return surf, LAYERS['soil']
		if kind == 'mud':
			pygame.draw.ellipse(surf, Colors.MUD, (7, 18, 50, 30))
			pygame.draw.ellipse(surf, Colors.MUD_LIGHT, (17, 24, 18, 7))
			pygame.draw.ellipse(surf, Colors.MUD_DARK, (28, 34, 20, 6))
			return surf, LAYERS['soil water']
		if kind == 'stake':
			pygame.draw.rect(surf, Colors.WOOD, (30, 18, 5, 34))
			pygame.draw.rect(surf, Colors.STAKE_TOP, (27, 12, 11, 8))
			pygame.draw.rect(surf, Colors.WOOD_DARK, (27, 12, 11, 8), 1)
			return surf, LAYERS['main']
		if kind == 'fallen_log':
			pygame.draw.ellipse(surf, Colors.WOOD_DARK, (6, 25, 52, 20))
			pygame.draw.ellipse(surf, Colors.WOOD, (8, 21, 48, 20))
			pygame.draw.line(surf, Colors.WOOD_DARK, (15, 29), (47, 29), 3)
			pygame.draw.circle(surf, Colors.WOOD_SIGN, (14, 31), 7, 2)
			pygame.draw.circle(surf, Colors.WOOD_SIGN, (49, 30), 6, 2)
			return surf, LAYERS['main']
		if kind == 'rock_pile':
			pygame.draw.circle(surf, Colors.DEBRIS_MID, (25, 35), 15)
			pygame.draw.circle(surf, Colors.DEBRIS, (38, 31), 13)
			pygame.draw.circle(surf, Colors.DEBRIS_DARK, (18, 27), 10)
			pygame.draw.rect(surf, Colors.DEBRIS_OUTLINE, (11, 20, 40, 28), 2, border_radius=8)
			return surf, LAYERS['main']
		if kind == 'broken_crate':
			pygame.draw.rect(surf, Colors.WOOD, (13, 18, 38, 32), border_radius=3)
			pygame.draw.rect(surf, Colors.WOOD_DARK, (13, 18, 38, 32), 3, border_radius=3)
			pygame.draw.line(surf, Colors.WOOD_DARK, (16, 22), (48, 47), 4)
			pygame.draw.line(surf, Colors.WOOD_SIGN, (19, 45), (47, 20), 3)
			return surf, LAYERS['main']

		# sign
		pygame.draw.rect(surf, Colors.WOOD, (30, 30, 5, 24))
		pygame.draw.rect(surf, Colors.WOOD_SIGN, (15, 13, 34, 18), border_radius=3)
		pygame.draw.rect(surf, Colors.WOOD_SIGN_DARK, (15, 13, 34, 18), 2, border_radius=3)
		pygame.draw.line(surf, Colors.SIGN_TEXT, (22, 20), (42, 20), 2)
		return surf, LAYERS['main']

	def _make_area_board_surface(self, mode):
		width = TILE_SIZE * 3
		height = TILE_SIZE
		surf = pygame.Surface((width, height), pygame.SRCALPHA)

		pygame.draw.rect(surf, Colors.WOOD_DARK, (20, 42, 8, 22))
		pygame.draw.rect(surf, Colors.WOOD_DARK, (width - 28, 42, 8, 22))
		pygame.draw.rect(surf, Colors.WOOD, (16, 38, 8, 26))
		pygame.draw.rect(surf, Colors.WOOD, (width - 24, 38, 8, 26))
		pygame.draw.rect(surf, Colors.WOOD_DARK, (8, 8, width - 8, 42), border_radius=6)
		pygame.draw.rect(surf, Colors.WOOD_SIGN, (4, 4, width - 12, 42), border_radius=6)
		pygame.draw.rect(surf, Colors.WOOD_SIGN_DARK, (4, 4, width - 12, 42), 2, border_radius=6)

		area_name = self.MODE_CONFIGS[mode].get('area_name', f'Khu {mode}')
		parts = area_name.split(' - ', 1)
		title = parts[0].upper()
		subtitle = parts[1] if len(parts) > 1 else area_name
		font_title = pygame.font.Font(None, 24)
		font_subtitle = pygame.font.Font(None, 18)

		text_center_x = width // 2
		icon = self._load_scaled_asset(
			f'graphics/ai_visuals/area_icon_{mode}.png', (34, 34))
		if icon:
			surf.blit(icon, (12, 9))
			text_center_x += 12

		title_surf = font_title.render(title, True, Colors.SIGN_TEXT)
		subtitle_surf = font_subtitle.render(subtitle[:26], True, Colors.STORY_TITLE_LIGHT)
		surf.blit(title_surf, title_surf.get_rect(center=(text_center_x, 17)))
		surf.blit(subtitle_surf, subtitle_surf.get_rect(center=(text_center_x, 34)))
		return surf

	def _create_area_name_boards(self):
		for mode, tile in self.AREA_BOARD_TILES.items():
			tx, ty = tile
			board = Generic(
				(tx * TILE_SIZE, ty * TILE_SIZE),
				self._make_area_board_surface(mode),
				self.all_sprites,
				z=LAYERS['main'])
			self.dynamic_decorations.append(board)

	def _create_mode_obstacles(self, mode):
		cfg = self.MODE_CONFIGS.get(mode, {})
		obstacle_kinds = cfg.get('obstacle_kinds', {})
		for tile in cfg.get('obstacles', []):
			tx, ty = tile
			pos = (tx * TILE_SIZE, ty * TILE_SIZE)
			kind = obstacle_kinds.get(tile, 'storm_debris')
			surf, layer = self._make_decor_surface(kind)
			obstacle = Generic(
				pos, surf,
				[self.all_sprites, self.collision_sprites],
				z=layer)
			self.dynamic_obstacles.append(obstacle)

	def _create_mode_decorations(self, mode):
		cfg = self.MODE_CONFIGS.get(mode, {})
		blocked_tiles = set(cfg.get('tiles', []))
		blocked_tiles.add(cfg.get('spawn_tile'))
		blocked_tiles.update(cfg.get('obstacles', []))
		blocked_tiles.update(cfg.get('hidden_blocks', []))
		enemy_spawn = cfg.get('enemy_spawn')
		if enemy_spawn:
			blocked_tiles.add(enemy_spawn)
		used_tiles = set()
		large_decoration_tiles = set()

		for kind, tx, ty in cfg.get('decorations', []):
			if kind == 'path':
				continue
			if (tx, ty) in blocked_tiles:
				continue
			if (tx, ty) in used_tiles:
				continue
			if kind not in self.COMPACT_DECORATION_KINDS:
				too_close = any(
					abs(tx - ux) <= 1 and abs(ty - uy) <= 1
					for ux, uy in large_decoration_tiles)
				if too_close:
					continue
				large_decoration_tiles.add((tx, ty))
			used_tiles.add((tx, ty))
			surf, layer = self._make_decor_surface(kind)
			groups = [self.all_sprites]
			if kind in self.SOLID_DECORATION_KINDS:
				groups.append(self.collision_sprites)
			decor = Generic(
				(tx * TILE_SIZE, ty * TILE_SIZE), surf,
				groups,
				z=layer)
			self.dynamic_decorations.append(decor)

	# ----------------------------------------------------------------
	# Core: khá»Ÿi táº¡o hoáº·c chuyá»ƒn mode
	# ----------------------------------------------------------------
	def _init_mode(self, mode, show_overview=False):
		"""Dá»n state cÅ© vĂ  thiáº¿t láº­p toĂ n bá»™ cho mode má»›i."""
		self.current_mode = mode
		cfg = self.MODE_CONFIGS[mode]
		if hasattr(self, 'sky'):
			self.sky.start_color = [255, 255, 255]

		# Dá»n
		self.clear_demo_soil()
		self._clear_mode_obstacles()

		self._create_link_sprites()
		self.overview_rect = self._calculate_overview_rect()

		# Tao ca 6 khu tren cung world de nguoi choi thay ban do lien ket.
		self.create_farm_from_tiles(self._all_demo_tiles())
		self.farm_tiles = self._task_tiles_for_config(cfg)

		# Tao vat can/trang tri cua tat ca khu, giu AI chi xu ly khu dang chon.
		self._create_world_decorations()

		# Spawn player
		spawn_tile = cfg['spawn_tile']
		spawn_pos = (spawn_tile[0] * TILE_SIZE + TILE_SIZE // 2,
					 spawn_tile[1] * TILE_SIZE + TILE_SIZE // 2)
		self.player.pos.update(spawn_pos)
		self.player.rect.center = spawn_pos
		self.player.hitbox.center = spawn_pos
		self.player.direction.update(0, 0)
		self.player.status = 'down_idle'

		# Hidden blocks cho mode 4 (override random cá»§a AI)
		hidden_blocks = cfg.get('hidden_blocks', None)
		# Enemy spawn cho mode 6
		enemy_spawn = cfg.get('enemy_spawn', None)
		terrain_costs = cfg.get('terrain_costs', {})
		extra_walkable_tiles = cfg.get('extra_walkable_tiles', [])

		# Khá»Ÿi táº¡o AI
		self.ai = FarmAIController(
			self.player, self.soil_layer, self.collision_sprites,
			self.farm_tiles, mode=mode,
			hidden_blocks=hidden_blocks,
			enemy_spawn=enemy_spawn,
			terrain_costs=terrain_costs,
			extra_walkable_tiles=extra_walkable_tiles,
			selected_algorithm=self.selected_algorithms.get(mode))
		if show_overview:
			self._start_overview()
		else:
			self.overview_timer = 0.0
			self.overview_active = False
			self.all_sprites.clear_overview()

	def set_ai_mode(self, mode, show_overview=False):
		if mode < 1 or mode > 6:
			return
		self._init_mode(mode, show_overview=show_overview)

		mode_names = {
			1: self.selected_algorithms.get(1, 'BFS'),
			2: self.selected_algorithms.get(2, 'A*'),
			3: self.selected_algorithms.get(3, 'Hill Climbing'),
			4: self.selected_algorithms.get(4, 'Online A*'),
			5: self.selected_algorithms.get(5, 'Backtrack'),
			6: 'Minimax'
		}
		area_name = self.MODE_CONFIGS[mode].get('area_name', f'Khu {mode}')
		pygame.display.set_caption(
			f'Smart Farm AI Robot - {area_name}: '
			f'{mode_names.get(mode, "")}')

	def cycle_algorithm(self, step=1):
		if self.current_mode not in (1, 2, 3, 4, 5) or not hasattr(self, 'ai'):
			return None
		name = self.ai.cycle_algorithm(step)
		self.selected_algorithms[self.current_mode] = name
		area_name = self.MODE_CONFIGS[self.current_mode].get(
			'area_name', f'Khu {self.current_mode}')
		pygame.display.set_caption(
			f'Smart Farm AI Robot - {area_name}: {name}')
		return name

	def handle_ui_click(self, pos):
		if not hasattr(self, 'ai'):
			return False
		action = self.ai.handle_panel_click(pos)
		if action == 'reset':
			self._init_mode(self.current_mode)
			return True
		return action is not None

	
	def player_add(self, item):
		self.player.item_inventory[item] += 1
		self.success.play()
	
	
	def toggle_shop(self):
		self.shop_active = not self.shop_active
	
	
	def reset(self):
		
		# plants
		self.soil_layer.update_plants()
		
		# soil
		self.soil_layer.remove_water()
		self.raining = True
		self.soil_layer.raining = self.raining

		if self.raining:
			self.soil_layer.water_all()

		# apples on the trees
		for tree in self.tree_sprites.sprites():
			for apple in tree.apple_sprites.sprites():
				apple.kill()
			tree.create_fruit()

		# sky
		self.sky.start_color = [255, 255, 255]
	
	
	def plant_collision(self):
		if self.soil_layer.plant_sprites:
			for plant in self.soil_layer.plant_sprites.sprites():
				if plant.harvestable and plant.rect.colliderect(self.player.hitbox):
					self.player_add(plant.plant_type)
					plant.kill()
					Particle(plant.rect.topleft, plant.image, self.all_sprites, LAYERS['main'])
					self.soil_layer.grid[plant.rect.centery // TILE_SIZE][plant.rect.x // TILE_SIZE].remove('P')

	
	def run(self,dt):

		# drawing logic
		self.display_surface.fill('black')
		if self.overview_active:
			self.overview_timer -= dt
			if self.overview_timer <= 0:
				self.overview_active = False
				self.all_sprites.clear_overview()
		self.all_sprites.customize_draw(self.player)

		# updates
		if self.shop_active:
			self.menu.update()
		else:
			if not self.overview_active:
				self.ai.update(dt)
			self.all_sprites.update(dt)
			self.plant_collision()

		# weather
		self.overlay.display()
		if not self.overview_active:
			self.ai.draw(self.display_surface, self.all_sprites.offset)

		# rain
		if self.raining and not self.shop_active:
			self.rain.update(light=True)
		if self.use_day_night:
			self.sky.display(dt)
		else:
			storm_tint = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
			storm_tint.fill(Colors.STORM_TINT)
			self.display_surface.blit(storm_tint, (0, 0))
		if self.current_mode == 4:
			fog_sky = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
			fog_sky.fill(Colors.FOG_SKY_TINT)
			self.display_surface.blit(fog_sky, (0, 0))

		# transition overlay
		if self.player.sleep:
			self.transition.play()



class CameraGroup(pygame.sprite.Group):
	def __init__(self):
		super().__init__()
		
		self.display_surface = pygame.display.get_surface()
		self.offset = pygame.math.Vector2()
		self.zoom = 1.0
		self.overview_rect = None

	def set_overview(self, overview_rect):
		self.overview_rect = overview_rect

	def clear_overview(self):
		self.overview_rect = None
		self.zoom = 1.0


	def customize_draw(self, player):
		if self.overview_rect:
			self.zoom = min(
				SCREEN_WIDTH / self.overview_rect.width,
				SCREEN_HEIGHT / self.overview_rect.height,
				1.0)
			center = pygame.math.Vector2(self.overview_rect.center)
			self.offset.x = center.x - (SCREEN_WIDTH / self.zoom) / 2
			self.offset.y = center.y - (SCREEN_HEIGHT / self.zoom) / 2
		else:
			self.zoom = 1.0
			self.offset.x = player.rect.centerx - SCREEN_WIDTH / 2
			self.offset.y = player.rect.centery - SCREEN_HEIGHT / 2

		for layer in LAYERS.values():
			for sprite in sorted(self.sprites(), key = lambda sprite: sprite.rect.centery):
				if sprite.z == layer:
					offset_rect = sprite.rect.copy()
					screen_center = (
						(offset_rect.centerx - self.offset.x) * self.zoom,
						(offset_rect.centery - self.offset.y) * self.zoom)
					if self.zoom == 1.0:
						offset_rect.center = screen_center
						self.display_surface.blit(sprite.image, offset_rect)
					else:
						width = max(1, int(sprite.image.get_width() * self.zoom))
						height = max(1, int(sprite.image.get_height() * self.zoom))
						image = pygame.transform.smoothscale(sprite.image, (width, height))
						draw_rect = image.get_rect(center=screen_center)
						self.display_surface.blit(image, draw_rect)








