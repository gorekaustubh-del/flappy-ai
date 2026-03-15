import pygame
import random
import neat
import os
import pickle
import time

pygame.init()
SCREEN = pygame.display.set_mode((400, 600))
CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont("Arial", 18)

GRAVITY = 0.25
FLAP_STRENGTH = -6.5

class Bird:
    def __init__(self):
        self.rect = pygame.Rect(50, 300, 30, 30)
        self.velocity = 0

    def move(self):
        self.velocity += GRAVITY
        self.rect.y += self.velocity

    def jump(self):
        self.velocity = FLAP_STRENGTH

class Pipe:
    def __init__(self, x):
        self.x = x
        self.gap_y = random.randint(200, 400)
        self.top_pipe = pygame.Rect(x, 0, 50, self.gap_y - 100)
        self.bottom_pipe = pygame.Rect(x, self.gap_y + 100, 50, 600)

    def move(self):
        self.x -= 3
        self.top_pipe.x = self.x
        self.bottom_pipe.x = self.x

def watch_best_bird(config_file):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    with open("best_bird.pkl", "rb") as f:
        winner = pickle.load(f)

    print("\n" + "="*60)
    print("  BEST BIRD LOADED - STARTING PLAYBACK")
    print("="*60)

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    bird = Bird()
    pipes = [Pipe(400)]
    score = 0
    running = True

    # Metrics
    start_time = time.time()
    total_jumps = 0
    max_velocity = 0
    min_velocity = 0
    close_calls = 0
    highest_position = bird.rect.y
    lowest_position = bird.rect.y
    jump_frames = []
    last_jump_frame = 0
    total_frames = 0

    print(f"\n  Bird starting at Y position: {bird.rect.y}")
    print(f"  Screen height: 600px")
    print(f"  Pipe gap size: 200px")
    print(f"  Pipe speed: 3px per frame")
    print(f"  Gravity: {GRAVITY} per frame")
    print(f"  Jump strength: {FLAP_STRENGTH}")
    print(f"\n  {'='*56}")
    print(f"  Live updates every 100 pipes...")
    print(f"  {'='*56}\n")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        total_frames += 1

        # Find the pipe to look at
        pipe_ind = 0
        if len(pipes) > 1 and bird.rect.x > pipes[0].top_pipe.right:
            pipe_ind = 1

        # Move bird
        bird.move()

        # Track velocity stats
        if bird.velocity > max_velocity:
            max_velocity = bird.velocity
        if bird.velocity < min_velocity:
            min_velocity = bird.velocity

        # Track position stats
        if bird.rect.y < highest_position:
            highest_position = bird.rect.y
        if bird.rect.y > lowest_position:
            lowest_position = bird.rect.y

        # AI makes decision with 5 inputs
        output = net.activate((
            bird.rect.y,
            abs(bird.rect.y - pipes[pipe_ind].top_pipe.bottom),
            abs(bird.rect.y - pipes[pipe_ind].bottom_pipe.top),
            bird.velocity,
            pipes[pipe_ind].top_pipe.x - bird.rect.x
        ))

        if output[0] > 0.5:
            bird.jump()
            total_jumps += 1
            frames_since_last_jump = total_frames - last_jump_frame
            jump_frames.append(frames_since_last_jump)
            last_jump_frame = total_frames

        # Move pipes
        for pipe in pipes:
            pipe.move()

        # Check close calls (within 20px of a pipe edge)
        for pipe in pipes:
            if (abs(bird.rect.x - pipe.top_pipe.right) < 10 and
                (abs(bird.rect.y - pipe.top_pipe.bottom) < 20 or
                 abs(bird.rect.y - pipe.bottom_pipe.top) < 20)):
                close_calls += 1

        # Check collision
        for pipe in pipes:
            if (bird.rect.colliderect(pipe.top_pipe) or
                bird.rect.colliderect(pipe.bottom_pipe) or
                bird.rect.top < 0 or
                bird.rect.bottom > 600):

                elapsed = time.time() - start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                avg_jump_interval = sum(jump_frames) / len(jump_frames) if jump_frames else 0
                pipes_per_min = (score / elapsed) * 60 if elapsed > 0 else 0

                print(f"\n{'='*60}")
                print(f"  BIRD DIED - FINAL REPORT")
                print(f"{'='*60}")
                print(f"\n  SCORE & TIME")
                print(f"  Final score:          {score} pipes")
                print(f"  Total time alive:     {minutes}m {seconds}s")
                print(f"  Pipes per minute:     {pipes_per_min:.1f}")
                print(f"  Total frames:         {total_frames}")
                print(f"\n  MOVEMENT STATS")
                print(f"  Total jumps:          {total_jumps}")
                print(f"  Jumps per pipe:       {total_jumps/score:.2f}" if score > 0 else "  Jumps per pipe:      N/A")
                print(f"  Avg frames per jump:  {avg_jump_interval:.1f}")
                print(f"  Max downward speed:   {max_velocity:.2f} px/frame")
                print(f"  Max upward speed:     {abs(min_velocity):.2f} px/frame")
                print(f"\n  POSITION STATS")
                print(f"  Highest position:     Y={highest_position}px (0=top)")
                print(f"  Lowest position:      Y={lowest_position}px (600=bottom)")
                print(f"  Vertical range used:  {lowest_position - highest_position}px out of 600px")
                print(f"\n  DANGER STATS")
                print(f"  Close calls:          {close_calls}")
                print(f"\n{'='*60}\n")
                running = False

        # Remove off-screen pipes
        pipes = [p for p in pipes if p.x > -50]

        # Spawn new pipe
        if pipes[-1].x < 250:
            pipes.append(Pipe(400))
            score += 1

            # Print live update every 100 pipes
            if score % 100 == 0:
                elapsed = time.time() - start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                pipes_per_min = (score / elapsed) * 60 if elapsed > 0 else 0
                avg_jump_interval = sum(jump_frames) / len(jump_frames) if jump_frames else 0
                print(f"  Score: {score} | "
                      f"Time: {minutes}m {seconds}s | "
                      f"Pipes/min: {pipes_per_min:.1f} | "
                      f"Jumps: {total_jumps} | "
                      f"Avg jump: {avg_jump_interval:.1f}f | "
                      f"Close calls: {close_calls}")

        # Calculate live metrics for screen
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        pipes_per_min = (score / elapsed) * 60 if elapsed > 0 else 0
        avg_jump_interval = sum(jump_frames) / len(jump_frames) if jump_frames else 0
        vertical_range = lowest_position - highest_position

        # Draw everything
        SCREEN.fill((255, 255, 255))
        for pipe in pipes:
            pygame.draw.rect(SCREEN, (0, 200, 0), pipe.top_pipe)
            pygame.draw.rect(SCREEN, (0, 200, 0), pipe.bottom_pipe)
        pygame.draw.rect(SCREEN, (255, 0, 0), bird.rect)

        # Live metrics on screen
        SCREEN.blit(FONT.render(f"Score: {score}", True, (0, 0, 0)), (10, 10))
        SCREEN.blit(FONT.render(f"Time: {minutes}m {seconds}s", True, (0, 0, 0)), (10, 33))
        SCREEN.blit(FONT.render(f"Pipes/min: {pipes_per_min:.1f}", True, (0, 0, 0)), (10, 56))
        SCREEN.blit(FONT.render(f"Jumps: {total_jumps}", True, (0, 0, 0)), (10, 79))
        SCREEN.blit(FONT.render(f"Jumps/pipe: {total_jumps/score:.2f}" if score > 0 else "Jumps/pipe: 0", True, (0, 0, 0)), (10, 102))
        SCREEN.blit(FONT.render(f"Avg jump: {avg_jump_interval:.1f}f", True, (0, 0, 0)), (10, 125))
        SCREEN.blit(FONT.render(f"Velocity: {bird.velocity:.2f}", True, (0, 100, 255)), (10, 148))
        SCREEN.blit(FONT.render(f"Y pos: {bird.rect.y}", True, (0, 100, 255)), (10, 171))
        SCREEN.blit(FONT.render(f"V.range: {vertical_range}px", True, (0, 100, 255)), (10, 194))
        SCREEN.blit(FONT.render(f"Close calls: {close_calls}", True, (255, 0, 0)), (10, 217))

        pygame.display.flip()
        CLOCK.tick(60)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    watch_best_bird(config_path)
