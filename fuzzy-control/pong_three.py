#!/usr/bin/env python3
# Based on https://python101.readthedocs.io/pl/latest/pygame/pong/#
import pygame
from typing import Type
import skfuzzy as fuzz
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl

FPS = 60  # TO DO FIX ME CHANGE ME TO 30
TIME_SCALE = 1.5 # TO DO FIX ME DELETE ME



class Board:
    def __init__(self, width: int, height: int):
        self.surface = pygame.display.set_mode((width, height), 0, 32)
        pygame.display.set_caption("AIFundamentals - PongGame")

    def draw(self, *args):
        background = (0, 0, 0)
        self.surface.fill(background)
        for drawable in args:
            drawable.draw_on(self.surface)

        pygame.display.update()


class Drawable:
    def __init__(self, x: int, y: int, width: int, height: int, color=(255, 255, 255)):
        self.width = width
        self.height = height
        self.color = color
        self.surface = pygame.Surface(
            [width, height], pygame.SRCALPHA, 32
        ).convert_alpha()
        self.rect = self.surface.get_rect(x=x, y=y)

    def draw_on(self, surface):
        surface.blit(self.surface, self.rect)


class Ball(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        radius: int = 20,
        color=(255, 10, 0),
        speed: int = 3,
    ):
        super(Ball, self).__init__(x, y, radius, radius, color)
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed = speed
        self.y_speed = speed
        self.start_speed = speed
        self.start_x = x
        self.start_y = y
        self.start_color = color
        self.last_collision = 0

    def bounce_y(self):
        self.y_speed *= -1

    def bounce_x(self):
        self.x_speed *= -1

    def bounce_y_power(self):
        self.color = (
            self.color[0],
            self.color[1] + 10 if self.color[1] < 255 else self.color[1],
            self.color[2],
        )
        pygame.draw.ellipse(self.surface, self.color, [0, 0, self.width, self.height])
        self.x_speed *= 1.1
        self.y_speed *= 1.1
        self.bounce_y()

    def reset(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y
        self.x_speed = self.start_speed
        self.y_speed = self.start_speed
        self.color = self.start_color
        self.bounce_y()

    def move(self, board: Board, *args):
        self.rect.x += round(self.x_speed * TIME_SCALE)
        self.rect.y += round(self.y_speed * TIME_SCALE)

        if self.rect.x < 0 or self.rect.x > (board.surface.get_width() - self.rect.width):
            self.bounce_x()

        if self.rect.y < 0 or self.rect.y > (board.surface.get_height() - self.rect.height):
            self.reset()

        timestamp = pygame.time.get_ticks()
        if FPS > 0 and timestamp - self.last_collision < (1000 // FPS) * 4:
            return

        for racket in args:
            if self.rect.colliderect(racket.rect):
                self.last_collision = pygame.time.get_ticks()
                if (self.rect.right < racket.rect.left + racket.rect.width // 4) or \
                   (self.rect.left > racket.rect.right - racket.rect.width // 4):
                    self.bounce_y_power()
                else:
                    self.bounce_y()

class Racket(Drawable):
    def __init__(
        self,
        x: int,
        y: int,
        width: int = 80,
        height: int = 20,
        color=(255, 255, 255),
        max_speed: int = 10,
    ):
        super(Racket, self).__init__(x, y, width, height, color)
        self.max_speed = max_speed
        self.surface.fill(color)

    def move(self, x: int, board: Board):
        # scale paddle’s max step by TIME_SCALE
        scaled_max = self.max_speed * TIME_SCALE

        delta = x - self.rect.x
        delta = scaled_max if delta > scaled_max else delta
        delta = -scaled_max if delta < -scaled_max else delta
        delta = 0 if (self.rect.x + delta) < 0 else delta
        delta = 0 if (self.rect.x + self.width + delta) > board.surface.get_width() else delta
        self.rect.x += int(round(delta))


class Player:
    def __init__(self, racket: Racket, ball: Ball, board: Board) -> None:
        self.ball = ball
        self.racket = racket
        self.board = board

    def move(self, x: int):
        self.racket.move(x, self.board)

    def move_manual(self, x: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass

    def act(self, x_diff: int, y_diff: int):
        """
        Do nothing, control is defined in derived classes
        """
        pass


class PongGame:
    def __init__(
        self, width: int, height: int, player1: Type[Player], player2: Type[Player]
    ):
        pygame.init()
        self.board = Board(width, height)
        self.fps_clock = pygame.time.Clock()
        self.ball = Ball(width // 2, height // 2)

        self.opponent_paddle = Racket(x=width // 2, y=0)
        self.oponent = player1(self.opponent_paddle, self.ball, self.board)

        self.player_paddle = Racket(x=width // 2, y=height - 20)
        self.player = player2(self.player_paddle, self.ball, self.board)

    def run(self):
        while not self.handle_events():
            self.ball.move(self.board, self.player_paddle, self.opponent_paddle)
            self.board.draw(
                self.ball,
                self.player_paddle,
                self.opponent_paddle,
            )
            self.oponent.act(
                self.oponent.racket.rect.centerx - self.ball.rect.centerx,
                self.oponent.racket.rect.centery - self.ball.rect.centery,
            )
            self.player.act(
                self.player.racket.rect.centerx - self.ball.rect.centerx,
                self.player.racket.rect.centery - self.ball.rect.centery,
            )
            self.fps_clock.tick(FPS)

    def handle_events(self):
        for event in pygame.event.get():
            if (event.type == pygame.QUIT) or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                pygame.quit()
                return True
        keys = pygame.key.get_pressed()
        if keys[pygame.constants.K_LEFT]:
            self.player.move_manual(0)
        elif keys[pygame.constants.K_RIGHT]:
            self.player.move_manual(self.board.surface.get_width())
        return False


class NaiveOponent(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(NaiveOponent, self).__init__(racket, ball, board)

    def act(self, x_diff: int, y_diff: int):
        x_cent = self.ball.rect.centerx
        self.move(x_cent)


class HumanPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(HumanPlayer, self).__init__(racket, ball, board)

    def move_manual(self, x: int):
        self.move(x)


class MamdaniPlayer(Player):

    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(MamdaniPlayer, self).__init__(racket, ball, board)

        W = self.board.surface.get_width()
        H = self.board.surface.get_height()
        vmax = float(self.racket.max_speed)

        # Universes
        self.x_universe = np.linspace(-W/2, W/2, 1001)
        self.y_universe = np.linspace(0, H, 1001)
        self.v_universe = np.linspace(-vmax, vmax, 801)

        self.xdiff = ctrl.Antecedent(self.x_universe, 'xdiff')
        self.ydist = ctrl.Antecedent(self.y_universe, 'ydist')  # |y_diff|
        self.vout  = ctrl.Consequent(self.v_universe, 'vout')

        # x_diff membership (5 sets)
        s = W/10
        self.xdiff['NL'] = fuzz.trapmf(self.x_universe, [-W/2, -W/2, -3*s, -1.5*s])
        self.xdiff['NS'] = fuzz.trimf(self.x_universe, [-2*s, -s, 0])
        self.xdiff['Z']  = fuzz.trimf(self.x_universe, [-s*0.6, 0, s*0.6])
        self.xdiff['PS'] = fuzz.trimf(self.x_universe, [0, s, 2*s])
        self.xdiff['PL'] = fuzz.trapmf(self.x_universe, [1.5*s, 3*s, W/2, W/2])

        # y (near vs far)
        y_near = H/5
        y_mid  = H/3
        self.ydist['near'] = fuzz.trapmf(self.y_universe, [0, 0, y_near, y_mid])
        self.ydist['far']  = fuzz.trapmf(self.y_universe, [y_mid, H*0.75, H, H])

        # x
        self.vout['FAST_LEFT']  = fuzz.trapmf(self.v_universe, [-vmax, -vmax, -0.85*vmax, -0.55*vmax])
        self.vout['SLOW_LEFT']  = fuzz.trimf(self.v_universe,  [-0.65*vmax, -0.40*vmax, -0.15*vmax])
        self.vout['STOP']       = fuzz.trimf(self.v_universe,  [-0.10*vmax, 0.0, 0.10*vmax])
        self.vout['SLOW_RIGHT'] = fuzz.trimf(self.v_universe,  [ 0.15*vmax,  0.40*vmax,  0.65*vmax])
        self.vout['FAST_RIGHT'] = fuzz.trapmf(self.v_universe, [ 0.55*vmax,  0.85*vmax,  vmax,      vmax])

        # rules
        rules = []
        # Strong moves <--> near
        rules.append(ctrl.Rule(self.xdiff['PL'] & self.ydist['near'], self.vout['FAST_LEFT']))
        rules.append(ctrl.Rule(self.xdiff['NL'] & self.ydist['near'], self.vout['FAST_RIGHT']))
        # gentle moves <--> far
        rules.append(ctrl.Rule(self.xdiff['PL'] & self.ydist['far'],  self.vout['SLOW_LEFT']))
        rules.append(ctrl.Rule(self.xdiff['NL'] & self.ydist['far'],  self.vout['SLOW_RIGHT']))
        # mid when mid
        rules.append(ctrl.Rule(self.xdiff['PS'], self.vout['SLOW_LEFT']))
        rules.append(ctrl.Rule(self.xdiff['NS'], self.vout['SLOW_RIGHT']))
        rules.append(ctrl.Rule(self.xdiff['Z'],  self.vout['STOP']))

        self.system = ctrl.ControlSystem(rules)

        self.sim = ctrl.ControlSystemSimulation(self.system, flush_after_run=50)

    def make_decision(self, x_diff: int, y_diff: int) -> int:
        # Inputs each frame
        self.sim.input['xdiff'] = float(x_diff)
        self.sim.input['ydist'] = float(abs(y_diff))

        self.sim.compute()
        v = float(self.sim.output['vout'])

        vmax = float(self.racket.max_speed)
        v = max(-vmax, min(vmax, v))

        return int(round(v))

    def act(self, x_diff: int, y_diff: int):
        v = self.make_decision(x_diff, y_diff)
        self.move(self.racket.rect.x + v)


if __name__ == "__main__":
    #game = PongGame(800, 400, NaiveOponent, HumanPlayer)
    game = PongGame(800, 400, NaiveOponent, MamdaniPlayer)
    game.run()