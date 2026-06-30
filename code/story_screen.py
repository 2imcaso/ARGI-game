import random

import pygame

from design_tokens import Colors, MODE_COLORS


STORY_DATA = {
    "intro": {
        "title": "LENH KHOI PHUC",
        "subtitle": "Nhiem vu cua AGRI-1",
        "lines": [
            "Sau mot tran bao lon, he thong nong trai thong minh bi sap nguon.",
            "Cac khu dat mat ket noi. Cay trong thieu nuoc. Vat can chan loi di.",
            "Mot so khu bi suong che phu, mot so khu can gieo lai tu dau.",
            "",
            "Robot AGRI-1 duoc kich hoat trong che do tu dong.",
            "No khong co nguoi dan duong.",
            "No chi nhan duoc danh sach khu vuc can khoi phuc.",
            "",
            "Moi khu dat la mot kieu van de khac nhau.",
            "Moi van de can mot cach suy nghi khac nhau.",
            "",
            "\"AGRI-1 online.\"",
            "\"Bat dau tien trinh khoi phuc nong trai.\"",
        ],
        "hint": "Nhan SPACE hoac ENTER de bat dau",
    },

    1: {
        "day": "NGAY 1 - KHU 1",
        "title": "Khu Dat Mat Tin Hieu",
        "algo": "BFS / DFS / IDS / UCS - Uninformed Search",
        "color": MODE_COLORS[1],
        "lines": [
            "AGRI-1 duoc dua den khu dat dau tien.",
            "Du lieu cam bien tai day gan nhu trong rong.",
            "Robot khong biet o nao quan trong hon o nao.",
            "",
            "Nhiem vu cua no rat co ban:",
            "di toi tung o dat, khoi phuc dat, gieo hat va tuoi nuoc.",
            "",
            "Khi khong co thong tin uu tien, AGRI-1 chi co the tim kiem theo cach may moc nhat.",
            "No mo rong tung buoc, kiem tra tung nhanh, khong bo sot khu vuc nao.",
            "",
            "\"Khong can doan dung ngay tu dau.\"",
            "\"Chi can tiep tuc tim cho den khi thay duong.\"",
        ],
        "hint": "Nhan SPACE de vao Khu 1",
    },

    2: {
        "day": "NGAY 2 - KHU 2",
        "title": "Loi Vao Bi Chan",
        "algo": "Greedy / A* / IDA* - Informed Search",
        "color": MODE_COLORS[2],
        "lines": [
            "Khu thu hai nam sau mot loi vao bi cay do va manh vo chan ngang.",
            "Khac voi ngay dau tien, AGRI-1 da biet vi tri cac cay can cuu.",
            "Nhung neu di sai duong, no se mat qua nhieu thoi gian.",
            "",
            "Lan nay robot khong chi tim duong.",
            "No uoc luong khoang cach den muc tieu, tinh so buoc da di,",
            "roi chon con duong co kha nang ngan va hop ly hon.",
            "",
            "Moi quyet dinh cua AGRI-1 deu dua tren cau hoi:",
            "duong nay co dua minh den cay can cuu nhanh hon khong?",
            "",
            "\"Du lieu da co.\"",
            "\"Bat dau tinh duong toi uu.\"",
        ],
        "hint": "Nhan SPACE de vao Khu 2",
    },

    3: {
        "day": "NGAY 3 - KHU 3",
        "title": "Vuon Cay Can Uu Tien",
        "algo": "Hill Climbing / Local Beam / Annealing / Restart Hill",
        "color": MODE_COLORS[3],
        "lines": [
            "Khu vuon thu ba khong bi chan duong.",
            "Van de nam o chinh nhung cay trong.",
            "",
            "Co cay kho.",
            "Co cay nguy cap.",
            "Co cay bi sau benh.",
            "Co cay da gan nhu khong the cuu.",
            "",
            "AGRI-1 khong con chi hoi: di duong nao ngan nhat?",
            "No phai hoi: cay nao dang dang duoc xu ly truoc nhat?",
            "",
            "Robot danh diem tung cay theo tinh trang cua no,",
            "roi chon buoc di co gia tri cham soc cao hon trong khu vuc hien tai.",
            "",
            "Nhung lua chon cuc bo co the lam no bi ket.",
            "Vi vay, mot so module se thu nhieu ung vien, chap nhan rui ro,",
            "hoac khoi dong lai khi duong hien tai khong con tot.",
            "",
            "\"Khong phai cay gan nhat nao cung la cay can cuu truoc.\"",
        ],
        "hint": "Nhan SPACE de vao Khu 3",
    },

    4: {
        "day": "NGAY 4 - KHU 4",
        "title": "Khu Suong Mu",
        "algo": "Online Search / Belief-State / AND-OR Search",
        "color": MODE_COLORS[4],
        "lines": [
            "Khu thu tu bi suong day bao phu.",
            "Ban do khong con dang tin.",
            "Mot so o trong nhu loi di, nhung co the la vat can an.",
            "",
            "AGRI-1 chi nhin thay vung rat gan quanh minh.",
            "Moi khi buoc sang mot o moi, no cap nhat lai nhung gi da biet.",
            "",
            "Neu phat hien duong bi chan, robot khong tiep tuc co chap.",
            "No danh dau vat can, sua lai niem tin ve ban do,",
            "roi tinh lai duong di tu vi tri hien tai.",
            "",
            "O khu nay, ke hoach khong phai thu co dinh.",
            "Ke hoach la thu duoc sua sau moi lan quan sat.",
            "",
            "\"Tam nhin bi gioi han.\"",
            "\"Kich hoat che do vua di vua hoc.\"",
        ],
        "hint": "Nhan SPACE de vao Khu 4",
    },

    5: {
        "day": "NGAY 5 - KHU 5",
        "title": "Luoi Gieo Trong",
        "algo": "Backtracking / Forward Checking / AC-3 / Min-Conflicts",
        "color": MODE_COLORS[5],
        "lines": [
            "Khu thu nam da duoc don sach de gieo trong lai.",
            "Nhung AGRI-1 khong the gieo hat tuy tien.",
            "",
            "Moi o dat can mot loai cay.",
            "Hai o ke nhau khong duoc trung loai.",
            "Mot so cap cay khong duoc dung canh nhau vi xung dot dinh duong.",
            "",
            "Lan nay, robot khong chi di toi muc tieu.",
            "No phai gan gia tri cho tung o dat, kiem tra rang buoc,",
            "neu sai thi lui lai va thu cach khac.",
            "",
            "Moi hat giong dat xuong la mot quyet dinh.",
            "Moi xung dot phat hien la mot lan robot phai sua lai ke hoach.",
            "",
            "\"Khong phai dat nao cung trong cay nao cung duoc.\"",
            "\"Bat dau giai rang buoc gieo trong.\"",
        ],
        "hint": "Nhan SPACE de vao Khu 5",
    },

    6: {
        "day": "NGAY 6 - KHU 6",
        "title": "Vong Bao Ve Cuoi",
        "algo": "Minimax / Alpha-Beta / Expectimax / Expectiminimax",
        "color": MODE_COLORS[6],
        "lines": [
            "Khu cuoi cung la vung cay gia tri cao nhat cua nong trai.",
            "Nhung day cung la noi sinh vat pha hoai xam nhap manh nhat.",
            "",
            "AGRI-1 xuat phat tu mot phia.",
            "Ke pha hoai xuat phat tu phia doi dien.",
            "",
            "Cay kho, cay nguy cap, cay bi sau benh va cay dac biet",
            "deu co gia tri va muc rui ro khac nhau.",
            "",
            "Neu robot chi chay den cay gan nhat, doi thu co the pha huy cay quan trong hon.",
            "Neu robot dang sua mot cay, no phai quyet dinh nen sua tiep hay bo de chan muc tieu gia tri cao hon.",
            "",
            "Moi luot la mot van dau:",
            "MAX co gang bao ve, MIN co gang pha huy,",
            "va CHANCE mo phong nhung rui ro khong chac chan.",
            "",
            "\"Khong chi can tim duong.\"",
            "\"Phai doan nuoc di tiep theo cua doi thu.\"",
        ],
        "hint": "Nhan SPACE de vao Khu 6",
    },

    "ending": {
        "title": "LENH KHOI PHUC",
        "subtitle": "Hoan tat nhiem vu",
        "lines": [
            "Sau sau khu vuc, nong trai bat dau hoat dong tro lai.",
            "",
            "Nhung luong dat dau tien da duoc khoi phuc.",
            "Nhung cay nguy cap da duoc cuu.",
            "Nhung vung suong mu da duoc khao sat.",
            "Luoi gieo trong da duoc sap xep dung rang buoc.",
            "Va khu cay cuoi cung da vuot qua dot tan cong.",
            "",
            "AGRI-1 khong chi hoan thanh nhiem vu.",
            "No da lan luot kich hoat sau cach suy nghi:",
            "tim kiem khi khong biet gi,",
            "uoc luong khi da co thong tin,",
            "uu tien khi tai nguyen co han,",
            "thich nghi khi moi truong khong chac chan,",
            "kiem tra rang buoc khi moi o dat lien quan den nhau,",
            "va doi dau khi co ke chong lai minh.",
            "",
            "\"Tien trinh khoi phuc hoan tat.\"",
            "\"AGRI-1 chuyen sang che do bao tri nong trai.\"",
        ],
        "hint": "Nhan SPACE de choi lai | ESC de thoat",
    },
}


def wrap_text(text, font, max_width):
    if text == "":
        return [""]

    wrapped = []
    line = ""
    for word in text.split(" "):
        test = line + word + " "
        if line and font.size(test)[0] > max_width:
            wrapped.append(line.rstrip())
            line = word + " "
        else:
            line = test

    if line:
        wrapped.append(line.rstrip())
    return wrapped


def wrapped_lines_height(lines, font, max_width, line_gap=0):
    height = 0
    for text in lines:
        if text == "":
            height += font.get_linesize() // 2
            continue
        height += len(wrap_text(text, font, max_width)) * (font.get_linesize() + line_gap)
    return height


def fit_story_font(lines, max_width, max_height, base_size, min_size):
    for size in range(base_size, min_size - 1, -1):
        font = pygame.font.Font(None, size)
        if wrapped_lines_height(lines, font, max_width) <= max_height:
            return font
    return pygame.font.Font(None, min_size)


def draw_wrapped_text(surface, text, font, color, x, y, max_width, line_gap=0):
    if text == "":
        return y + font.get_linesize() // 2

    for line in wrap_text(text, font, max_width):
        surface.blit(font.render(line, True, color), (x, y))
        y += font.get_linesize() + line_gap
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

    def _draw_lines(self, surface, font, x, y, width, accent=Colors.STORY_QUOTE, line_gap=0):
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
            y = draw_wrapped_text(surface, chunk, font, color, x, y, width, line_gap)
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
        self._fit_layout()
        self.stars = [
            (random.randint(0, self.W), random.randint(0, self.H), random.randint(1, 2))
            for _ in range(90)
        ]

    def _fit_layout(self):
        self.content_width = min(760, self.W - 180)
        self.content_x = (self.W - self.content_width) // 2
        self.body_top = 190
        self.hint_bottom = self.H - 28
        max_body_height = self.hint_bottom - self.body_top - 34
        self.font_body = fit_story_font(self.lines, self.content_width, max_body_height, 27, 20)

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
        rule_half = min(230, self.content_width // 2)
        pygame.draw.line(
            self.screen,
            Colors.STORY_RULE,
            (self.W // 2 - rule_half, 168),
            (self.W // 2 + rule_half, 168),
            1,
        )
        self._draw_lines(self.screen, self.font_body, self.content_x, self.body_top, self.content_width, line_gap=1)
        if self.chars_shown >= self.total_chars and self.hint_visible:
            hint = self.font_hint.render(self.hint, True, Colors.STORY_HINT)
            self.screen.blit(hint, hint.get_rect(centerx=self.W // 2, bottom=self.hint_bottom))


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
        self._fit_layout()

    def _fit_layout(self):
        self.panel_width = min(self.W - 180, 860)
        self.panel_height = min(self.H - 70, 620)
        self.panel = pygame.Rect(
            (self.W - self.panel_width) // 2,
            (self.H - self.panel_height) // 2,
            self.panel_width,
            self.panel_height,
        )
        self.body_x = self.panel.left + 58
        self.body_y = self.panel.top + 170
        self.body_width = self.panel.width - 116
        self.hint_bottom = self.panel.bottom - 20
        body_height = self.hint_bottom - self.body_y - 34
        self.font_body = fit_story_font(self.lines, self.body_width, body_height, 26, 18)

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_RETURN):
            self.done = self._handle_continue()

    def draw(self):
        tint = (max(8, self.color[0] // 18), max(8, self.color[1] // 18), max(12, self.color[2] // 16))
        self.screen.fill(tint)
        panel = self.panel
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
        self._draw_lines(self.screen, self.font_body, self.body_x, self.body_y, self.body_width, line_gap=1)

        if self.chars_shown >= self.total_chars and self.hint_visible:
            hint = self.font_hint.render(self.hint, True, Colors.STORY_HINT_DARK)
            self.screen.blit(hint, hint.get_rect(centerx=self.W // 2, bottom=self.hint_bottom))


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
        self._fit_layout()

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
        panel_width = min(460, surface.get_width() - 40)
        panel = pygame.Rect(surface.get_width() - panel_width - 20, 18, panel_width, 148)
        panel_surf = pygame.Surface(panel.size, pygame.SRCALPHA)
        panel_surf.fill(Colors.STORY_PANEL_FLOAT)
        surface.blit(panel_surf, panel.topleft)
        pygame.draw.rect(surface, self.color, panel, 2, border_radius=8)
        x = panel.left + 16
        width = panel.width - 32
        surface.blit(self.font_day.render(self.day, True, self.color), (x, panel.top + 10))
        surface.blit(self.font_title.render(self.title, True, Colors.STORY_TITLE_LIGHT), (x, panel.top + 36))
        draw_wrapped_text(surface, self.algo, self.font_algo, Colors.STORY_TEXT_MUTED, x, panel.top + 70, width)
        draw_wrapped_text(surface, self.mission, self.font_body, Colors.STORY_BODY_MUTED, x, panel.top + 104, width)
