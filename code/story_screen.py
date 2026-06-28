import random

import pygame

from design_tokens import Colors, MODE_COLORS


STORY_DATA = {
    "intro": {
        "title": "MUA HE CUOI CUNG",
        "subtitle": "Cau chuyen cua AGRI-1",
        "lines": [
            "Trang trai ho Minh da nuoi song gia dinh ba doi.",
            "Nhung tran bao lon da lam moi thu tan hoang.",
            "",
            "Hoa mau do rap. Vat can chan loi di.",
            "Suong mu che phu nhung manh dat chua ai dam den.",
            "",
            "Ong Minh khoi dong lai AGRI-1 de di qua tung khu dat.",
            "\"Trang trai nay can may.\"",
        ],
        "hint": "Nhan SPACE hoac ENTER de bat dau",
    },
    1: {
        "day": "NGAY 1 - KHU 1",
        "title": "Vuon Truoc Trai",
        "algo": "BFS - Uninformed Search",
        "color": MODE_COLORS[1],
        "lines": [
            "AGRI-1 bat dau o khu vuon truoc trai, chua co ban do.",
            "Robot chi co the tham do deu theo tung lop.",
            "",
            "Muc tieu dau tien: tim manh dat gan nhat va bat dau cuoc.",
            "\"Khong thong minh hon, chi kien nhan hon.\"",
        ],
        "hint": "Nhan SPACE de vao Khu 1",
    },
    2: {
        "day": "NGAY 2 - KHU 2",
        "title": "Loi Vao Nha Kho",
        "algo": "A* - Informed Search",
        "color": MODE_COLORS[2],
        "lines": [
            "Sang ngay hai, AGRI-1 duoc dieu toi loi vao nha kho.",
            "Manh vo va da lon chan ngang duong vao khu dat.",
            "AGRI-1 da biet vi tri dich va biet vat can.",
            "",
            "Lan nay robot dung uoc luong Manhattan de di vong ngan hon.",
            "\"Hom nay, toi di thong minh hon.\"",
        ],
        "hint": "Nhan SPACE de vao Khu 2",
    },
    3: {
        "day": "NGAY 3 - KHU 3",
        "title": "Vuon Cay Phia Dong",
        "algo": "Hill Climbing - Local Search",
        "color": MODE_COLORS[3],
        "lines": [
            "Khu vuon phia dong bi nang hon, nhieu cay dang heo kho.",
            "AGRI-1 khong the cuu tat ca trong mot ngay.",
            "",
            "Robot chon o co gia tri cham soc cao nhat trong vung lan can.",
            "Doi khi tham lam cuc bo se khong toi uu toan cuc.",
        ],
        "hint": "Nhan SPACE de vao Khu 3",
    },
    4: {
        "day": "NGAY 4 - KHU 4",
        "title": "Bai Dat Suong Mu",
        "algo": "Online Search - Partial Observability",
        "color": MODE_COLORS[4],
        "lines": [
            "Ngay thu tu, AGRI-1 vao bai dat thap bi suong mu phu kin.",
            "AGRI-1 chi thay duoc cac o sat ben canh.",
            "",
            "Hai buc tuong da an tao ngo cut tren duong toi cay o goc xa.",
            "Khi phat hien da chan, robot phai quay lai va re-plan.",
            "\"Ke hoach tot nhat la ke hoach biet thay doi.\"",
        ],
        "hint": "Nhan SPACE de vao Khu 4",
    },
    5: {
        "day": "NGAY 5 - KHU 5",
        "title": "O Quy Hoach Trung Tam",
        "algo": "CSP Backtracking - Constraint Satisfaction",
        "color": MODE_COLORS[5],
        "lines": [
            "Khu trung tam duoc don sach de quy hoach lai.",
            "Ong Minh muon trong lai theo quy tac moi.",
            "Hai o ke nhau khong duoc trong cung loai cay.",
            "",
            "AGRI-1 phai gan crop cho tung o, sai thi lui lai va thu tiep.",
            "\"Co viec phai nghi truoc khi hanh dong.\"",
        ],
        "hint": "Nhan SPACE de vao Khu 5",
    },
    6: {
        "day": "NGAY 6 - KHU 6",
        "title": "Hang Cay Can Bao Ve",
        "algo": "Minimax / Alpha-Beta / Expectimax / Expectiminimax",
        "color": MODE_COLORS[6],
        "lines": [
            "Ngay cuoi, AGRI-1 toi hang cay phia dong nam.",
            "Khi vuon gan hoi sinh, ke pha hoai da xuat hien.",
            "Cay khoe, cay kho, cay sau benh va cay quy co muc uu tien khac nhau.",
            "",
            "Gia tri va rui ro cua tung cay quyet dinh nuoc di cua AI.",
            "\"Doi mat voi ke thu thong minh, va dung de thua.\"",
        ],
        "hint": "Nhan SPACE de vao Khu 6",
    },
    "ending": {
        "title": "MUA HE CUOI CUNG",
        "subtitle": "Ket thuc",
        "lines": [
            "Sau sau ngay, sau khu dat da duoc cuu lai.",
            "",
            "AGRI-1 da hoc cach tim duong, lap ke hoach, thich nghi,",
            "giai rang buoc va doi dau voi doi thu.",
            "",
            "\"Mua he nay, chung ta da khong bo cuoc.\"",
        ],
        "hint": "Nhan SPACE de choi lai | ESC de thoat",
    },
}


def draw_wrapped_text(surface, text, font, color, x, y, max_width):
    if text == "":
        return y + font.get_linesize() // 2

    line = ""
    for word in text.split(" "):
        test = line + word + " "
        if line and font.size(test)[0] > max_width:
            surface.blit(font.render(line.rstrip(), True, color), (x, y))
            y += font.get_linesize()
            line = word + " "
        else:
            line = test

    if line:
        surface.blit(font.render(line.rstrip(), True, color), (x, y))
        y += font.get_linesize()
    return y


class TypewriterScreen:
    TEXT_SPEED = 0.025
    HINT_BLINK = 0.65

    def _init_typewriter(self, lines):
        self.lines = lines
        self.full_text = "\n".join(lines)
        self.total_chars = len(self.full_text)
        self.chars_shown = 0
        self.char_elapsed = 0.0
        self.hint_timer = 0.0
        self.hint_visible = True
        self.elapsed = 0.0

    def update(self, dt):
        self.elapsed += dt
        self.char_elapsed += dt
        self.hint_timer += dt

        new_chars = int(self.char_elapsed / self.TEXT_SPEED)
        if new_chars:
            self.chars_shown = min(self.total_chars, self.chars_shown + new_chars)
            self.char_elapsed = 0.0

        if self.hint_timer >= self.HINT_BLINK:
            self.hint_visible = not self.hint_visible
            self.hint_timer = 0.0

    def _handle_continue(self):
        if self.chars_shown < self.total_chars:
            self.chars_shown = self.total_chars
            return False
        return True

    def _draw_lines(self, surface, font, x, y, width, accent=Colors.STORY_QUOTE):
        remaining = self.chars_shown
        for line in self.lines:
            if remaining <= 0:
                break
            chunk = line[:remaining]
            remaining -= len(line) + 1
            if line == "":
                y += font.get_linesize() // 2
                continue
            color = accent if line.startswith('"') else Colors.STORY_BODY
            y = draw_wrapped_text(surface, chunk, font, color, x, y, width)
        return y


class IntroScreen(TypewriterScreen):
    def __init__(self, screen):
        self.screen = screen
        self.W, self.H = screen.get_size()
        self.done = False
        data = STORY_DATA["intro"]
        self.title = data["title"]
        self.subtitle = data["subtitle"]
        self.hint = data["hint"]
        self.font_title = pygame.font.Font(None, 66)
        self.font_subtitle = pygame.font.Font(None, 32)
        self.font_body = pygame.font.Font(None, 27)
        self.font_hint = pygame.font.Font(None, 24)
        self._init_typewriter(data["lines"])
        self.stars = [
            (random.randint(0, self.W), random.randint(0, self.H), random.randint(1, 2))
            for _ in range(90)
        ]

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_RETURN):
            self.done = self._handle_continue()

    def draw(self):
        self.screen.fill(Colors.STORY_NIGHT_BG)
        for x, y, r in self.stars:
            pygame.draw.circle(self.screen, Colors.STAR, (x, y), r)
        title = self.font_title.render(self.title, True, Colors.STORY_TITLE)
        self.screen.blit(title, title.get_rect(centerx=self.W // 2, top=58))
        subtitle = self.font_subtitle.render(self.subtitle, True, Colors.STORY_SUBTITLE)
        self.screen.blit(subtitle, subtitle.get_rect(centerx=self.W // 2, top=128))
        pygame.draw.line(self.screen, Colors.STORY_RULE, (430, 168), (850, 168), 1)
        self._draw_lines(self.screen, self.font_body, self.W // 2 - 310, 195, 620)
        if self.chars_shown >= self.total_chars and self.hint_visible:
            hint = self.font_hint.render(self.hint, True, Colors.STORY_HINT)
            self.screen.blit(hint, hint.get_rect(centerx=self.W // 2, bottom=self.H - 30))


class DayTransitionScreen(TypewriterScreen):
    def __init__(self, screen, mode):
        self.screen = screen
        self.W, self.H = screen.get_size()
        self.done = False
        data = STORY_DATA.get(mode, STORY_DATA[1])
        self.day = data["day"]
        self.title = data["title"]
        self.algo = data["algo"]
        self.color = data["color"]
        self.hint = data["hint"]
        self.font_day = pygame.font.Font(None, 36)
        self.font_title = pygame.font.Font(None, 58)
        self.font_algo = pygame.font.Font(None, 27)
        self.font_body = pygame.font.Font(None, 26)
        self.font_hint = pygame.font.Font(None, 23)
        self._init_typewriter(data["lines"])

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_RETURN):
            self.done = self._handle_continue()

    def draw(self):
        tint = (max(8, self.color[0] // 18), max(8, self.color[1] // 18), max(12, self.color[2] // 16))
        self.screen.fill(tint)
        panel = pygame.Rect((self.W - 720) // 2, (self.H - 500) // 2, 720, 500)
        panel_surf = pygame.Surface(panel.size, pygame.SRCALPHA)
        panel_surf.fill(Colors.STORY_PANEL_BG)
        self.screen.blit(panel_surf, panel.topleft)
        pygame.draw.rect(self.screen, self.color, panel, 2, border_radius=10)

        day = self.font_day.render(self.day, True, self.color)
        self.screen.blit(day, day.get_rect(centerx=self.W // 2, top=panel.top + 22))
        title = self.font_title.render(self.title, True, Colors.STORY_TITLE_LIGHT)
        self.screen.blit(title, title.get_rect(centerx=self.W // 2, top=panel.top + 60))
        algo = self.font_algo.render(self.algo, True, Colors.STORY_TEXT_LIGHT)
        self.screen.blit(algo, algo.get_rect(centerx=self.W // 2, top=panel.top + 120))
        pygame.draw.line(self.screen, self.color, (panel.left + 44, panel.top + 154), (panel.right - 44, panel.top + 154), 1)
        self._draw_lines(self.screen, self.font_body, panel.left + 55, panel.top + 175, panel.width - 110)

        if self.chars_shown >= self.total_chars and self.hint_visible:
            hint = self.font_hint.render(self.hint, True, Colors.STORY_HINT_DARK)
            self.screen.blit(hint, hint.get_rect(centerx=self.W // 2, bottom=panel.bottom - 18))


class EndingScreen(IntroScreen):
    def __init__(self, screen):
        super().__init__(screen)
        self.restart = False
        self.quit = False
        data = STORY_DATA["ending"]
        self.title = data["title"]
        self.subtitle = data["subtitle"]
        self.hint = data["hint"]
        self._init_typewriter(data["lines"])

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.quit = True
            elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                self.restart = self._handle_continue()


class StoryPanel:
    SHOW_TIME = 6.0

    def __init__(self, mode):
        data = STORY_DATA.get(mode, STORY_DATA[1])
        self.day = data["day"]
        self.title = data["title"]
        self.algo = data["algo"]
        self.color = data["color"]
        self.mission = next((line for line in data["lines"] if line and not line.startswith('"')), "")
        self.elapsed = 0.0
        self.visible = True
        self.font_day = pygame.font.Font(None, 28)
        self.font_title = pygame.font.Font(None, 32)
        self.font_algo = pygame.font.Font(None, 21)
        self.font_body = pygame.font.Font(None, 20)

    def update(self, dt):
        if self.visible:
            self.elapsed += dt
            if self.elapsed >= self.SHOW_TIME:
                self.visible = False

    def dismiss(self):
        self.visible = False

    def draw(self, surface):
        if not self.visible:
            return
        panel = pygame.Rect(surface.get_width() - 450, 20, 430, 122)
        panel_surf = pygame.Surface(panel.size, pygame.SRCALPHA)
        panel_surf.fill(Colors.STORY_PANEL_FLOAT)
        surface.blit(panel_surf, panel.topleft)
        pygame.draw.rect(surface, self.color, panel, 2, border_radius=8)
        surface.blit(self.font_day.render(self.day, True, self.color), (panel.left + 14, panel.top + 10))
        surface.blit(self.font_title.render(self.title, True, Colors.STORY_TITLE_LIGHT), (panel.left + 14, panel.top + 34))
        surface.blit(self.font_algo.render(self.algo, True, Colors.STORY_TEXT_MUTED), (panel.left + 14, panel.top + 66))
        surface.blit(self.font_body.render(self.mission[:60], True, Colors.STORY_BODY_MUTED), (panel.left + 14, panel.top + 92))
