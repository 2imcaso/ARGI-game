"""Shared visual tokens for the AGRI-1 AI demo.

The original token generator is web-oriented. This file keeps the same idea
but stores pygame-friendly RGB/RGBA tuples so map, fog, story, and HUD styling
can be tuned from one place.
"""


class Colors:
    # Overall post-storm grading. Keep these cool and muted.
    STORM_TINT = (70, 94, 118, 56)
    FOG_SKY_TINT = (96, 120, 145, 88)

    # Mode 4 fog-of-war.
    FOG_DARK = (22, 27, 36, 168)
    FOG_EDGE = (22, 27, 36, 104)
    FOG_MID = (22, 27, 36, 56)
    FOG_CLEAR = (0, 0, 0, 0)
    MIST_BASE = (210, 220, 225, 18)
    MIST_BAND = (220, 230, 235)
    MIST_HIGHLIGHT = (235, 240, 245)

    # HUD surfaces.
    PANEL_BG = (15, 15, 25, 210)
    PANEL_SEPARATOR = (60, 60, 80)
    TEXT_PRIMARY = (220, 220, 220)
    TEXT_SUBTITLE = (220, 220, 255)
    TEXT_MUTED = (140, 140, 160)
    TEXT_STAT = (200, 220, 255)
    TEXT_WARNING = (255, 235, 150)
    TEXT_SUCCESS = (180, 255, 180)
    TEXT_DIFFICULTY = (255, 220, 140)

    # Map overlays.
    TASK_MARKER_FILL = (255, 138, 196, 34)
    TASK_MARKER_BORDER = (255, 175, 220, 120)
    TASK_MARKER_CORNER = (255, 205, 235, 150)
    PATH = (255, 230, 0)
    BLOCK_LABEL = (255, 255, 255)
    ENEMY = (255, 80, 80)
    ENEMY_FALLBACK = (220, 50, 50)
    ENEMY_FALLBACK_OUTLINE = (255, 100, 100)
    CSP_CORN = (255, 220, 80)
    CSP_TOMATO = (255, 100, 100)
    CSP_CORN_BG = (80, 60, 0, 50)
    CSP_TOMATO_BG = (80, 0, 0, 50)
    SCORE_TEXT = (255, 255, 255)

    # Decorative fallback surfaces.
    PUDDLE = (54, 105, 160, 105)
    PUDDLE_HIGHLIGHT = (122, 174, 218, 80)
    PUDDLE_SHADOW = (35, 78, 125, 55)
    MIST_DECOR = (220, 228, 238, 36)
    MIST_DECOR_DIM = (220, 228, 238, 30)
    MIST_DECOR_FAINT = (220, 228, 238, 26)
    PATH_DECOR = (144, 116, 72, 54)
    PATH_DECOR_LIGHT = (176, 148, 92, 70)
    PATH_DECOR_DARK = (104, 82, 54, 55)
    PATH_DECOR_HIGHLIGHT = (190, 166, 112, 60)
    MUD = (76, 54, 40, 135)
    MUD_LIGHT = (112, 82, 56, 105)
    MUD_DARK = (45, 34, 28, 90)
    WOOD = (96, 64, 38)
    WOOD_DARK = (72, 46, 30)
    WOOD_SIGN = (176, 118, 64)
    WOOD_SIGN_DARK = (76, 48, 31)
    SIGN_TEXT = (242, 216, 132)
    STAKE_TOP = (210, 188, 104)
    DEBRIS = (110, 100, 95)
    DEBRIS_OUTLINE = (150, 140, 130)
    DEBRIS_DARK = (90, 82, 78)
    DEBRIS_MID = (130, 120, 112)
    DEBRIS_SHADOW = (95, 88, 82)

    # Story screens.
    STORY_NIGHT_BG = (8, 10, 18)
    STORY_PANEL_BG = (12, 14, 22, 220)
    STORY_PANEL_FLOAT = (12, 14, 22, 215)
    STORY_TITLE = (225, 205, 130)
    STORY_SUBTITLE = (170, 160, 120)
    STORY_RULE = (95, 88, 60)
    STORY_BODY = (210, 205, 185)
    STORY_QUOTE = (255, 235, 140)
    STORY_HINT = (155, 150, 125)
    STORY_HINT_DARK = (150, 145, 125)
    STORY_TITLE_LIGHT = (240, 235, 210)
    STORY_TEXT_LIGHT = (230, 230, 210)
    STORY_TEXT_MUTED = (220, 220, 205)
    STORY_BODY_MUTED = (185, 180, 160)
    STAR = (130, 130, 150)


MODE_COLORS = {
    1: (100, 160, 255),  # BFS
    2: (100, 255, 130),  # A*
    3: (255, 200, 80),   # Hill Climbing
    4: (180, 120, 255),  # Online Search
    5: (255, 160, 80),   # CSP
    6: (255, 80, 80),    # Minimax
}


MODE_EXPLORE_FILLS = {
    1: (100, 160, 255, 45),
    2: (100, 255, 130, 45),
}
