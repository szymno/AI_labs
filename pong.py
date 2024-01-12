import pygame
from typing import Type
import skfuzzy as fuzz
import skfuzzy.control as fuzzcontrol

FPS = 60


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
        print("DDDDDDDDDDDDDDDDDDDDDDDDD")
        self.bounce_y()

    def reset(self):
        self.rect.x = self.start_x
        self.rect.y = self.start_y
        self.x_speed = self.start_speed
        self.y_speed = self.start_speed
        self.color = self.start_color
        self.bounce_y()

    def move(self, board: Board, *args):
        self.rect.x += round(self.x_speed)
        self.rect.y += round(self.y_speed)

        if self.rect.x < 0 or self.rect.x > (
            board.surface.get_width() - self.rect.width
        ):
            self.bounce_x()

        if self.rect.y < 0 or self.rect.y > (
            board.surface.get_height() - self.rect.height
        ):
            self.reset()

        timestamp = pygame.time.get_ticks()
        if timestamp - self.last_collision < FPS * 4:
            return

        for racket in args:
            if self.rect.colliderect(racket.rect):
                self.last_collision = pygame.time.get_ticks()
                if (self.rect.right < racket.rect.left + racket.rect.width // 4) or (
                    self.rect.left > racket.rect.right - racket.rect.width // 4
                ):
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
        delta = x - self.rect.x
        delta = self.max_speed if delta > self.max_speed else delta
        delta = -self.max_speed if delta < -self.max_speed else delta
        delta = 0 if (self.rect.x + delta) < 0 else delta
        delta = (
            0
            if (self.rect.x + self.width + delta) > board.surface.get_width()
            else delta
        )
        self.rect.x += delta


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
        print(self.board.surface.get_width(), type(self.board.surface.get_width()))

    def move_manual(self, x: int):
        self.move(x)


# ----------------------------------
# DO NOT MODIFY CODE ABOVE THIS LINE
# ----------------------------------

import numpy as np
import matplotlib.pyplot as plt

from skfuzzy.control import Rule, Antecedent, Consequent, ControlSystem, ControlSystemSimulation
from skfuzzy.membership import trapmf, trimf


class FuzzyMamdamiPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(FuzzyMamdamiPlayer, self).__init__(racket, ball, board)
        self.racket_width = racket.width
        self.racket_height = racket.height
        self.ball_xy = ball.rect.width

        x_univ = np.arange(-self.board.surface.get_width(), self.board.surface.get_width() + 1)
        y_univ = np.arange(self.board.surface.get_height())
        v_univ = np.arange(-self.racket.max_speed, self.racket.max_speed + 1)

        x_dist = Antecedent(x_univ, "X")
        y_dist = Antecedent(y_univ, "Y")
        velocity = Consequent(v_univ, "Racket velocity", "centroid")

        x_dist["right"] = trapmf(x_univ, (-800, -800, -40, -32))
        x_dist["above"] = trapmf(x_univ, (-40, -32, 32, 40))
        x_dist["left"] = trapmf(x_univ, (32, 40, 800, 800))

        y_dist["low"] = trapmf(y_univ, (0, 0, 300, 350))
        y_dist["high"] = trapmf(y_univ, (300, 350, 399, 399))

        velocity["left fast"] = trapmf(v_univ, (-10, -10, -10, -9))
        velocity["left"] = trapmf(v_univ, (-10, -9, -6, -0))
        velocity["zero"] = trimf(v_univ, (-6, 0, 6))
        velocity["right"] = trapmf(v_univ, (0, 6, 9, 10))
        velocity["right fast"] = trapmf(v_univ, (9, 10, 10, 10))

        rules = (
            Rule(x_dist["left"] & y_dist["low"],  velocity["left fast"]),
            Rule(x_dist["left"] & y_dist["high"], velocity["left"]),

            Rule(x_dist["above"], velocity["zero"]),

            Rule(x_dist["right"] & y_dist["low"], velocity["right fast"]),
            Rule(x_dist["right"] & y_dist["high"], velocity["right"]),
        )
        self.racket_controller = ControlSystem(rules)
        self.racket_computing_controller = ControlSystemSimulation(self.racket_controller)

        velocity.view()
        x_dist.view()
        y_dist.view()
        self.racket_controller.view()
        plt.show()

    def act(self, x_diff: int, y_diff: int):
        velocity = self.make_decision(x_diff, y_diff)
        self.move(self.racket.rect.x + velocity)

    def make_decision(self, x_diff: int, y_diff: int):
        self.racket_computing_controller.inputs(
            {
                "X": x_diff,
                "Y": y_diff
            }
        )
        self.racket_computing_controller.compute()
        return self.racket_computing_controller.output["Racket velocity"]


class FuzzyTSKPlayer(Player):
    def __init__(self, racket: Racket, ball: Ball, board: Board):
        super(FuzzyTSKPlayer, self).__init__(racket, ball, board)
        self.x_univ = np.arange(-self.board.surface.get_width(), self.board.surface.get_width() + 1)
        self.y_univ = np.arange(self.board.surface.get_height())
        self.v_univ = np.arange(-self.racket.max_speed, self.racket.max_speed + 1)

        self.x_dist = {}
        self.y_dist = {}

        self.x_dist["right"] = trapmf(self.x_univ, (-800, -800, -40, -32))
        self.x_dist["above"] = trapmf(self.x_univ, (-40, -32, 32, 40))
        self.x_dist["left"] = trapmf(self.x_univ, (32, 40, 800, 800))

        self.y_dist["low"] = trapmf(self.y_univ, (0, 0, 300, 350))
        self.y_dist["high"] = trapmf(self.y_univ, (300, 350, 399, 399))

        self.velocity_f = {
            "left fast": lambda x_diff, y_diff: -0.6 * (abs(x_diff) + y_diff),
            "left": lambda x_diff, y_diff: -0.3 * (abs(x_diff) + y_diff),
            "zero": lambda x_diff, y_diff: 0.1 * abs(x_diff),
            "right": lambda x_diff, y_diff: 0.3 * (abs(x_diff) + y_diff),
            "right fast": lambda x_diff, y_diff: 0.6 * (abs(x_diff) + y_diff),
        }

        # visualize TSK
        plt.figure()
        for name, mf in self.x_dist.items():
            plt.plot(self.x_univ, mf, label=name)
        plt.legend()
        plt.show()

    def act(self, x_diff: int, y_diff: int):
        velocity = self.make_decision(x_diff, y_diff)
        self.move(self.racket.rect.x + velocity)

    def make_decision(self, x_diff: int, y_diff: int):
        x_vals = {
            name: fuzz.interp_membership(self.x_univ, mf, x_diff)
            for name, mf in self.x_dist.items()
        }

        y_vals = {
            name: fuzz.interp_membership(self.y_univ, mf, y_diff)
            for name, mf in self.y_dist.items()
        }

        activations = {
            "left fast": max(
                [
                    x_vals["left"],
                    y_vals["low"],
                ]
            ),
            "left": max(
                [
                    x_vals["left"],
                    y_vals["high"],
                ]
            ),
            "zero": max(
                [
                    x_vals["above"],
                ]
            ),
            "right": max(
                [
                    x_vals["right"],
                    y_vals["high"],
                ]
            ),
            "right fast": max(
                [
                    x_vals["right"],
                    y_vals["low"],
                ]
            ),
        }

        velocity = sum(
            activations[val] * self.velocity_f[val](x_diff, y_diff)
            for val in activations
        ) / sum(activations[val] for val in activations)
        return velocity


if __name__ == "__main__":
    # game = PongGame(800, 400, NaiveOponent, HumanPlayer)
    game = PongGame(800, 400, NaiveOponent, FuzzyTSKPlayer)
    game.run()