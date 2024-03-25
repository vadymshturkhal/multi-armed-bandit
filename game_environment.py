import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Colors
background_color = (30, 30, 30)
text_color = (255, 255, 255)

# Fonts
font = pygame.font.Font(None, 36)

# Number of bandits
k = 5
# Reward probabilities for each bandit
reward_probabilities = [random.random() for _ in range(k)]


class MultiArmedGame:
    def __init__(self):
        self.total_score = 0

    def draw_text(self, text, position):
        text_surface = font.render(text, True, text_color)
        rect = text_surface.get_rect(center=position)
        screen.blit(text_surface, rect)

    def game_loop(self):
        last_reward = 0
        last_choice = None

        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if pygame.K_1 <= event.key <= pygame.K_9:
                        chosen_bandit = event.key - pygame.K_1
                        if chosen_bandit < k:
                            last_choice = chosen_bandit
                            # Simulate bandit pull
                            if random.random() < reward_probabilities[chosen_bandit]:
                                last_reward = 1
                            else:
                                last_reward = 0
                            self.total_score += last_reward

            # Fill the background
            screen.fill(background_color)

            # Display instructions
            self.draw_text("Press keys 1-{} to pull a bandit's lever".format(k), (400, 50))
            # Display scores
            self.draw_text("Total Score: {}".format(self.total_score), (400, 150))
            self.draw_text("Last Reward: {}".format(last_reward), (400, 200))
            if last_choice is not None:
                self.draw_text("Last Bandit Chosen: {}".format(last_choice + 1), (400, 250))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = MultiArmedGame()
    game.game_loop()
