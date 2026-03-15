import pygame
import random
import neat
import os
import pickle
import copy
import glob

pygame.init()
SCREEN = pygame.display.set_mode((400, 600))
CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont("Arial", 24)

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

generation = 0
best_genome_ever = None
best_fitness_ever = -1
SAVE_INTERVAL = 10000
MAX_SCORE = 50000

def save_best(genome, fitness, generation, score):
    global best_genome_ever, best_fitness_ever
    best_fitness_ever = fitness
    best_genome_ever = copy.deepcopy(genome)
    with open("best_bird_tmp.pkl", "wb") as f:
        pickle.dump(best_genome_ever, f)
    os.replace("best_bird_tmp.pkl", "best_bird.pkl")
    print(f"  *** SAVED! Fitness: {best_fitness_ever:.0f} (gen {generation}, score {score}) ***")

def eval_genomes(genomes, config):
    global generation, best_genome_ever, best_fitness_ever
    generation += 1

    print(f"\n{'='*50}")
    print(f"  GENERATION {generation} STARTING")
    print(f"  Birds alive: {len(genomes)}")
    print(f"  All time best fitness so far: {best_fitness_ever:.0f}")
    print(f"{'='*50}")

    nets = []
    ge = []
    birds = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird())
        ge.append(genome)

    pipes = [Pipe(400)]
    score = 0
    best_score_this_gen = 0
    birds_started = len(birds)
    last_save_threshold = 0

    while len(birds) > 0:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        pipe_ind = 0
        if len(pipes) > 1 and birds[0].rect.x > pipes[0].top_pipe.right:
            pipe_ind = 1

        for x, bird in enumerate(birds):
            ge[x].fitness += 0.05
            bird.move()

            output = nets[x].activate((
                bird.rect.y,
                abs(bird.rect.y - pipes[pipe_ind].top_pipe.bottom),
                abs(bird.rect.y - pipes[pipe_ind].bottom_pipe.top),
                bird.velocity,
                pipes[pipe_ind].top_pipe.x - bird.rect.x
            ))

            if output[0] > 0.5:
                bird.jump()

        to_remove = []
        for pipe in pipes:
            pipe.move()
            for x, bird in enumerate(birds):
                if (bird.rect.colliderect(pipe.top_pipe) or
                    bird.rect.colliderect(pipe.bottom_pipe) or
                    bird.rect.top < 0 or
                    bird.rect.bottom > 600):
                    ge[x].fitness -= 1

                    # Save if this bird beats the all time best
                    if ge[x].fitness > best_fitness_ever:
                        save_best(ge[x], ge[x].fitness, generation, score)

                    to_remove.append(x)

        for x in sorted(set(to_remove), reverse=True):
            birds.pop(x)
            nets.pop(x)
            ge.pop(x)

        if len(birds) == 0:
            break

        pipes = [p for p in pipes if p.x > -50]

        if pipes[-1].x < 250:
            pipes.append(Pipe(400))
            score += 1
            if score > best_score_this_gen:
                best_score_this_gen = score
            for g in ge:
                g.fitness += 10

            if score % 100 == 0:
                print(f"  Milestone: {len(birds)} birds still alive at score {score}!")

        # Save every 10000 fitness for any surviving bird
        for x, g in enumerate(ge):
            if g.fitness >= last_save_threshold + SAVE_INTERVAL:
                last_save_threshold = (g.fitness // SAVE_INTERVAL) * SAVE_INTERVAL
                if g.fitness > best_fitness_ever:
                    save_best(g, g.fitness, generation, score)
                    print(f"  *** INTERVAL SAVE at fitness {g.fitness:.0f}! ***")

        # End generation at max score
        if score >= MAX_SCORE:
            print(f"  *** SCORE {MAX_SCORE} REACHED! Ending generation. ***")
            # Save any surviving birds if they beat the record
            for g in ge:
                if g.fitness > best_fitness_ever:
                    save_best(g, g.fitness, generation, score)
            break

        SCREEN.fill((255, 255, 255))
        for pipe in pipes:
            pygame.draw.rect(SCREEN, (0, 200, 0), pipe.top_pipe)
            pygame.draw.rect(SCREEN, (0, 200, 0), pipe.bottom_pipe)
        for bird in birds:
            pygame.draw.rect(SCREEN, (255, 0, 0), bird.rect)

        SCREEN.blit(FONT.render(f"Score: {score}", True, (0, 0, 0)), (10, 10))
        SCREEN.blit(FONT.render(f"Alive: {len(birds)}", True, (0, 0, 0)), (10, 40))
        SCREEN.blit(FONT.render(f"Gen: {generation}", True, (0, 0, 0)), (10, 70))
        SCREEN.blit(FONT.render(f"Best: {best_fitness_ever:.0f}", True, (255, 0, 0)), (10, 100))

        pygame.display.flip()
        CLOCK.tick(60)

    print(f"\n  --- Generation {generation} Summary ---")
    print(f"  Birds that started: {birds_started}")
    print(f"  Best score reached this generation: {best_score_this_gen}")
    print(f"  All time best fitness: {best_fitness_ever:.0f}")

def run(config_file):
    global best_fitness_ever, generation

    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # Load previous best fitness so we never overwrite a good bird
    if os.path.exists("best_bird.pkl"):
        try:
            with open("best_bird.pkl", "rb") as f:
                old_winner = pickle.load(f)
                best_fitness_ever = old_winner.fitness
                print(f"Loaded previous best fitness: {best_fitness_ever:.0f}")
        except Exception:
            print("Could not load best_bird.pkl, starting fresh!")
            best_fitness_ever = -1
    else:
        print("No previous best bird found!")

    # Load latest checkpoint with proper numerical sorting
    checkpoints = sorted(
        glob.glob("neat-checkpoint-*"),
        key=lambda x: int(x.split("-")[-1])
    )

    if checkpoints:
        latest = checkpoints[-1]
        checkpoint_gen = int(latest.split("-")[-1])
        generation = checkpoint_gen
        print(f"Resuming from checkpoint: {latest} (generation {checkpoint_gen})")
        p = neat.Checkpointer.restore_checkpoint(latest)
    else:
        print("No checkpoint found, starting fresh population!")
        p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(neat.Checkpointer(1))

    print("\n" + "="*50)
    print("  FLAPPY BIRD AI TRAINING STARTED")
    print(f"  Continuing from generation {generation}")
    print("  100 birds per generation")
    print("  200 generations max")
    print(f"  Generation ends at score {MAX_SCORE}")
    print(f"  Best bird saves every {SAVE_INTERVAL} fitness AND on new records")
    print("="*50 + "\n")

    try:
        p.run(eval_genomes, 200)
    except Exception as e:
        print(f"\n{e}")

    print(f"\n{'='*50}")
    print(f"  TRAINING COMPLETE!")
    print(f"  Best fitness ever achieved: {best_fitness_ever:.0f}")
    print(f"  Best bird has been saved to best_bird.pkl")
    print(f"  Run watch_bird.py to watch it play!")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
