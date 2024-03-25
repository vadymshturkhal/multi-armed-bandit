import pygame
import random
import sys


# Screen dimensions
WIDTH = 600
HEIGHT = 400

# Colors
background_color = (30, 30, 30)
text_color = (255, 255, 255)

# Number of bandits
k = 5
# Reward probabilities for each bandit
reward_probabilities = [random.random() for _ in range(k)]


class MultiArmedGame:
    def __init__(self, is_rendering=True):
        self.total_score = 0
        self.last_reward = 0
        self.last_choice = None
        self.is_rendering = is_rendering
    
        # init display
        if self.is_rendering:
            pygame.init()
            self.display = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption('Multi Armed')
            self.clock = pygame.time.Clock()

            # Fonts
            pygame.font.init()  # Initialize font module
            self.font = pygame.font.Font(None, 36)

    def game_loop(self):
        running = True
        while True:
            if self.is_rendering:
                self._handle_events()
                self._update_ui()
                self.clock.tick(60)
                pygame.display.flip()
    
    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('Quit')
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if pygame.K_1 <= event.key <= pygame.K_9:
                    chosen_bandit = event.key - pygame.K_1
                    if chosen_bandit < k:
                        self.last_choice = chosen_bandit
                        # Simulate bandit pull
                        if random.random() < reward_probabilities[chosen_bandit]:
                            self.last_reward = 1
                        else:
                            self.last_reward = 0
                        self.total_score += self.last_reward

    def _draw_text(self, text, position):
        text_surface = self.font.render(text, True, text_color)
        rect = text_surface.get_rect(center=position)
        self.display.blit(text_surface, rect)

    def _update_ui(self):
        # Fill the background
        self.display.fill(background_color)

        # Display instructions
        self._draw_text("Press keys 1-{} to pull a bandit's lever".format(k), (WIDTH // 2, 50))
        # Display scores
        self._draw_text("Total Score: {}".format(self.total_score), (WIDTH // 2, 150))
        self._draw_text("Last Reward: {}".format(self.last_reward), (WIDTH // 2, 200))
        if self.last_choice is not None:
            self._draw_text("Last Bandit Chosen: {}".format(self.last_choice + 1), (WIDTH // 2, 250))


if __name__ == "__main__":
    game = MultiArmedGame(is_rendering=True)
    game.game_loop()
