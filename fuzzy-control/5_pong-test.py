import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
import importlib

game = importlib.import_module("pong_five")

def run_simulation(rounds=100, width=800, height=400, time_scale=2.0, max_steps_per_round=20000):
    """
    Plays `rounds` rounds; each round ends when the ball crosses top (you win) or bottom (you lose).
    Returns (wins, losses).
    """
    if hasattr(game, "TIME_SCALE"):
        game.TIME_SCALE = time_scale
    if hasattr(game, "FPS"):
        game.FPS = 0

    pygame.init()

    wins = 0
    losses = 0

    for rnd in range(rounds):
        g = game.PongGame(width, height, game.NaiveOponent, game.FuzzyPlayer)

        def no_draw(*args, **kwargs):
            pass
        g.board.draw = no_draw

        last_y_before_reset = {"y": None}
        original_reset = g.ball.reset

        def reset_hook():
            y_before = g.ball.rect.y
            last_y_before_reset["y"] = y_before
            original_reset()

        g.ball.reset = reset_hook

        g.player_paddle.rect.x = width // 2 - g.player_paddle.width // 2
        g.opponent_paddle.rect.x = width // 2 - g.opponent_paddle.width // 2

        decided = False
        steps = 0

        while not decided and steps < max_steps_per_round:
            g.ball.move(g.board, g.player_paddle, g.opponent_paddle)

            g.oponent.act(
                g.oponent.racket.rect.centerx - g.ball.rect.centerx,
                g.oponent.racket.rect.centery - g.ball.rect.centery,
            )
            g.player.act(
                g.player.racket.rect.centerx - g.ball.rect.centerx,
                g.player.racket.rect.centery - g.ball.rect.centery,
            )

            if last_y_before_reset["y"] is not None:
                if last_y_before_reset["y"] < 0:
                    wins += 1
                else:
                    losses += 1
                decided = True

            steps += 1

        if not decided:
            wins += 1

        pygame.event.pump()

    pygame.quit()
    return wins, losses


if __name__ == "__main__":
    wins, losses = run_simulation(
        rounds=10,
        width=800,
        height=400,
        time_scale=3.0,
        max_steps_per_round=30000
    )
    total = wins + losses
    win_rate = (wins / total) * 100 if total else 0.0
    print(f"Rounds: {total}")
    print(f"Wins:   {wins}")
    print(f"Losses: {losses}")
    print(f"Win rate: {win_rate:.2f}%")
