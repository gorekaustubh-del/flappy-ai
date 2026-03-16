# Flappy Bird NEAT AI

A self-learning Flappy Bird AI built with Python and NEAT (NeuroEvolution of Augmenting Topologies).
The AI trains itself by evolving a population of birds across generations, getting better over time
without any hardcoded rules.

## Results
- Best fitness achieved: 320,000+
- Highest score in a single run: 57,000+ pipes

## How it works
Each bird has a neural network brain with 5 inputs:
- Bird Y position
- Distance to top pipe
- Distance to bottom pipe
- Current velocity
- Horizontal distance to next pipe

The output is a single number — if it's above 0.5 the bird jumps, otherwise it falls.

NEAT evolves the population by:
1. Spawning 100 random birds
2. Letting them all play at once
3. Killing bad birds, breeding good ones
4. Adding random mutations
5. Repeating until the birds are skilled enough to play indefinitely

## Files
- `flappy_ai.py` — training script, runs 100 birds per generation and saves the best brain
- `watch_bird.py` — loads the best saved brain and watches it play with live metrics
- `config-feedforward.txt` — NEAT configuration file defining evolution rules

## Requirements
```
pip3 install pygame neat-python
```

## How to train
```
python3 flappy_ai.py
```
Training continues from the latest checkpoint automatically.
The best bird is saved to `best_bird.pkl` whenever a new record is set.

## How to watch the best bird play
```
python3 watch_bird.py
```
Shows live metrics including score, pipes per minute, jumps, velocity, and close calls.

## Built with
- Python 3.10
- Pygame 2.6.1
- NEAT-Python
