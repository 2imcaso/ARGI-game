import os
import sys

os.chdir(os.path.dirname(__file__))

import pygame

from settings import *
from level import Level
from story_screen import IntroScreen, DayTransitionScreen, EndingScreen, StoryPanel


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Smart Farm AI Robot")
        self.clock = pygame.time.Clock()
        self.level = Level()

        self.mode = 1
        self.state = "intro"
        self.intro_screen = IntroScreen(self.screen)
        self.day_screen = None
        self.ending_screen = None
        self.story_panel = None

    def start_day(self, mode):
        self.mode = mode
        self.level.set_ai_mode(mode)
        self.day_screen = DayTransitionScreen(self.screen, mode)
        self.story_panel = None
        self.state = "day_transition"

        mode_names = {
            1: self.level.selected_algorithms.get(1, "BFS"),
            2: self.level.selected_algorithms.get(2, "A*"),
            3: self.level.selected_algorithms.get(3, "Hill Climbing"),
            4: self.level.selected_algorithms.get(4, "Online A*"),
            5: self.level.selected_algorithms.get(5, "Backtrack"),
            6: "Minimax",
        }
        area_names = {
            1: "Khu 1 - Vuon truoc trai",
            2: "Khu 2 - Loi vao nha kho",
            3: "Khu 3 - Vuon cay phia dong",
            4: "Khu 4 - Bai dat suong mu",
            5: "Khu 5 - O quy hoach trung tam",
            6: "Khu 6 - Hang cay can bao ve",
        }
        area_name = area_names.get(mode, "Khu " + str(mode))
        pygame.display.set_caption(
            f"Smart Farm AI Robot - {area_name}: {mode_names.get(mode, '')}")

    def enter_gameplay(self):
        self.story_panel = StoryPanel(self.mode)
        self.state = "game"

    def show_ending(self):
        self.ending_screen = EndingScreen(self.screen)
        self.state = "ending"

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if self.state == "intro":
            self.intro_screen.handle_event(event)
            return

        if self.state == "day_transition" and self.day_screen:
            self.day_screen.handle_event(event)
            return

        if self.state == "ending" and self.ending_screen:
            self.ending_screen.handle_event(event)
            return

        if self.state != "game":
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.level.handle_ui_click(event.pos):
                return

        if event.type != pygame.KEYDOWN:
            return

        if self.story_panel and self.story_panel.visible:
            self.story_panel.dismiss()

        if pygame.K_1 <= event.key <= pygame.K_6:
            self.start_day(event.key - pygame.K_0)
        elif event.key == pygame.K_q and self.mode in (1, 2, 3, 4, 5):
            self.level.cycle_algorithm(-1)
        elif event.key == pygame.K_e and self.mode in (1, 2, 3, 4, 5):
            self.level.cycle_algorithm(1)
        elif event.key == pygame.K_e:
            self.show_ending()

    def update_and_draw(self, dt):
        if self.state == "intro":
            self.intro_screen.update(dt)
            self.intro_screen.draw()
            if self.intro_screen.done:
                self.start_day(1)
            return

        if self.state == "day_transition" and self.day_screen:
            self.day_screen.update(dt)
            self.day_screen.draw()
            if self.day_screen.done:
                self.enter_gameplay()
            return

        if self.state == "ending" and self.ending_screen:
            self.ending_screen.update(dt)
            self.ending_screen.draw()
            if self.ending_screen.quit:
                pygame.quit()
                sys.exit()
            if self.ending_screen.restart:
                self.intro_screen = IntroScreen(self.screen)
                self.state = "intro"
            return

        self.level.run(dt)
        if self.story_panel:
            self.story_panel.update(dt)
            self.story_panel.draw(self.screen)

    def run(self):
        while True:
            dt = self.clock.tick() / 1000
            for event in pygame.event.get():
                self.handle_event(event)
            self.update_and_draw(dt)
            pygame.display.update()


if __name__ == "__main__":
    game = Game()
    game.run()

