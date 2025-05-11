# Maze Solver AI Challenge

![Maze Game Screenshot](screenshot.png)  <!-- Placeholder for screenshot -->

Welcome to the **Maze Solver AI Challenge**! This is an interactive maze game where you can either manually guide an agent to the goal or let AI algorithms solve the maze for you. The game features multiple pathfinding algorithms, dynamic obstacles, power-ups, and lifelines to enhance the gameplay experience.

## Features

- **Maze Generation**: Procedurally generated mazes with multiple paths.
- **Pathfinding Algorithms**: Choose from BFS, DFS, A*, or Greedy to solve the maze.
- **Lifelines**: Use special abilities like breaking walls, freezing obstacles, or revealing parts of the path.
- **Moving Obstacles**: Avoid dynamic obstacles that move around the maze.
- **Power-ups**: Collect power-ups to gain additional lifelines or bonus points.
- **Algorithm Comparison**: Real-time comparison of algorithm performance from the current state.
- **Graphical User Interface**: Intuitive GUI built with Tkinter for easy interaction.

## Requirements

- Python 3.x
- Tkinter (included in standard Python installations)

## How to Run

1. Ensure you have Python installed on your system.
2. Download the `AI_Project.py` file.
3. Open a terminal or command prompt.
4. Navigate to the directory containing `AI_Project.py`.
5. Run the command: `python AI_Project.py`

## Game Objective

- Guide the agent (blue square) to the goal (green square) after collecting the required number of keys (yellow squares).
- Avoid moving obstacles (red squares) that can collide with the agent and deduct points.
- Use lifelines strategically to overcome challenges.

## Controls

- **Arrow Keys**: Move the agent up, down, left, or right.
- **AI Controls**:
  - Select an algorithm from the dropdown menu.
  - Click "Run AI" to let the AI solve the maze.
  - Click "Pause AI" to pause or resume the AI's movement.
  - Click "Optimal Solver" to run the most efficient algorithm from the current state.
- **Lifelines**:
  - **Wall Break**: Click to enter wall-breaking mode, then click on a wall to break it.
  - **Freeze**: Freeze obstacles for 10 seconds.
  - **Reveal Path**: Reveal a segment of the path using the selected algorithm.
- **Game Controls**:
  - **Replay Maze**: Restart the current maze.
  - **New Maze**: Generate a new random maze.

## AI Algorithms

The game includes four pathfinding algorithms:

- **BFS (Breadth-First Search)**: Explores all possible paths level by level.
- **DFS (Depth-First Search)**: Explores as far as possible along each branch before backtracking.
- **A* (A-Star)**: Uses heuristics to find the shortest path efficiently.
- **Greedy**: Always chooses the path that seems best at the moment based on heuristics.

You can select any of these algorithms to see how they perform in solving the maze.

## Lifelines

- **Wall Break**: Allows you to break a wall in the maze, creating a new path. Limited uses.
- **Freeze Obstacles**: Freezes all obstacles for 10 seconds, preventing them from moving.
- **Reveal Path**: Reveals a segment of the path to the next key or the goal using the selected algorithm.

Lifelines can be gained by collecting power-ups (pink squares) in the maze.

## Algorithm Comparison

The game features a live comparison table that shows the performance of each algorithm from the agent's current position and state. The table updates periodically and highlights the best-performing algorithm based on moves, simulated score, power-ups collected, and time taken.

## Feature Key

The right panel of the game window includes a feature key that explains the symbols used in the maze:

- **Agent**: Blue square
- **Goal**: Green square
- **Key**: Yellow square
- **Power-up**: Pink square
- **Obstacle**: Red square
- **Wall**: Dark gray square
- **Path**: Light gray square
- **AI Trace**: Light blue square (shows the path taken by the AI)
- **Revealed Path**: Orange square (shows the path revealed by the lifeline)

## Educational Purpose

This game is designed not only for entertainment but also to educate players about pathfinding algorithms. By visualizing how BFS, DFS, A*, and Greedy algorithms navigate the maze, players can gain insights into their strengths and weaknesses.

## Code Structure

The project is structured into two main classes:

- **`MazeEnvironment`**: Handles the game logic, including maze generation, agent movement, obstacle movement, and pathfinding algorithms.
- **`MazeApp`**: Manages the graphical user interface using Tkinter, including drawing the maze, handling user inputs, and displaying statistics.

## Additional Notes

- The game ends when the agent reaches the goal after collecting the required keys.
- You can replay the same maze or generate a new one after the game ends.
- Moving obstacles move at regular intervals and can collide with the agent, deducting points.

Enjoy the challenge and see if you can outsmart the AI!