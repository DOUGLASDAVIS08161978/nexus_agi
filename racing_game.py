#!/usr/bin/env python3
import curses
import random
import time

def main(stdscr):
    # Setup
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(1)   # Non-blocking input
    stdscr.timeout(100) # Refresh rate

    # Get screen dimensions
    height, width = stdscr.getmaxyx()

    # Game variables
    car_pos = width // 2
    score = 0
    speed = 100
    obstacles = []
    game_over = False

    # Colors
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)

    def draw_car(y, x):
        """Draw the player's car"""
        if y < height - 1 and x >= 0 and x < width:
            stdscr.addstr(y, x, "🏎", curses.color_pair(2))

    def draw_obstacle(y, x):
        """Draw an obstacle"""
        if y >= 0 and y < height and x >= 0 and x < width:
            stdscr.addstr(y, x, "💥", curses.color_pair(1))

    def draw_road():
        """Draw road lanes"""
        for i in range(0, height, 2):
            if width // 3 < width:
                stdscr.addstr(i, width // 3, "|", curses.color_pair(4))
            if 2 * width // 3 < width:
                stdscr.addstr(i, 2 * width // 3, "|", curses.color_pair(4))

    # Game loop
    while not game_over:
        stdscr.clear()

        # Draw road
        draw_road()

        # Handle input
        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key == curses.KEY_LEFT or key == ord('a'):
            car_pos = max(0, car_pos - 2)
        elif key == curses.KEY_RIGHT or key == ord('d'):
            car_pos = min(width - 2, car_pos + 2)

        # Spawn obstacles
        if random.randint(1, 20) == 1:
            obstacle_x = random.choice([width // 4, width // 2, 3 * width // 4])
            obstacles.append([0, obstacle_x])

        # Move and draw obstacles
        for obs in obstacles[:]:
            obs[0] += 1
            if obs[0] >= height:
                obstacles.remove(obs)
                score += 10
                # Increase difficulty
                if score % 50 == 0 and speed > 30:
                    speed -= 5
                    stdscr.timeout(speed)
            else:
                draw_obstacle(obs[0], obs[1])

                # Collision detection
                if obs[0] >= height - 2 and obs[0] <= height - 1:
                    if abs(obs[1] - car_pos) <= 1:
                        game_over = True

        # Draw player car
        draw_car(height - 2, car_pos)

        # Draw score
        score_text = f"SCORE: {score} | Press Q to quit | Use A/D or Arrow Keys"
        if len(score_text) < width:
            stdscr.addstr(0, (width - len(score_text)) // 2, score_text, curses.color_pair(3))

        stdscr.refresh()

    # Game over screen
    stdscr.clear()
    game_over_text = "GAME OVER!"
    final_score = f"Final Score: {score}"
    play_again = "Press any key to exit"

    stdscr.addstr(height // 2 - 1, (width - len(game_over_text)) // 2, game_over_text, curses.color_pair(1))
    stdscr.addstr(height // 2, (width - len(final_score)) // 2, final_score, curses.color_pair(3))
    stdscr.addstr(height // 2 + 1, (width - len(play_again)) // 2, play_again)

    stdscr.nodelay(0)
    stdscr.getch()

if __name__ == "__main__":
    curses.wrapper(main)
