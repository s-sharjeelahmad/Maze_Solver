import tkinter as tk
from tkinter import messagebox, ttk
import random
import heapq
from collections import deque
import time
import copy #replay keliye

# Constants
GRID_WIDTH = 30
GRID_HEIGHT = 20
CELL_SIZE = 30
WALL = '#'
PATH = ' '
AGENT = 'A'
GOAL = 'G'
KEY = 'K'
POWERUP = 'P'
VISITED_PATH = '·' #AI path visualization
REVEALED_STEP = '*' # Revealed steps
OBSTACLE = 'O'

COLOR_MAP = {
    WALL: "#1C1C1C",           
    PATH: "#F0F0F0",            
    AGENT: "#007ACC",           
    GOAL: "#2ECC71",           
    KEY: "#FFD700",             
    POWERUP: "#FF69B4",        
    VISITED_PATH: "#ADD8E6",   
    REVEALED_STEP: "#FFA500",   
    OBSTACLE: "#B22222", 
}

#--1st class: Game Environment and Logic--
class MazeEnvironment:
    #1. Constructor (declaring)
    def __init__(self, width, height):
        #dimensions
        self.width = width
        self.height = height

        self.initial_grid_config = None #storing maze for replay
        self.initial_agent_pos = (1,1) #agent at top left
        self.initial_goal_pos = (width-2, height-2) #goal at bottom right

        #lists to store initial positions of keys, powerups and obstacles
        self.initial_keys = []
        self.initial_powerups = []
        self.initial_obstacles = []
    
        self.initial_total_keys_required = 0 #number of keys required to reach goal
        
        self.grid = []#current state of maze grid
        self.agent_pos = self.initial_agent_pos #current position of agent
        self.goal_pos = self.initial_goal_pos #position of goal
        #list to track dynamic entities of the game
        self.keys = []
        self.powerups = []
        self.obstacles = []

        self.obstacle_direction = {}    #dictionary for tracking obstacle movement
        self.revealed_path_segments = []    # stores path segments revealed by the player (if allowed)
        
        self.algorithm_paths = {} # storing path used by different algos
        self.algorithm_stats = {} # storing states for each algo
        
        # Game state
        self.wall_breaks = 0    # Number of wall-breaks the player can use
        self.freeze_available = 0    # Powerup to freeze obstacles
        self.reveal_path_available = 0  # Powerup to reveal part of path
        self.frozen_obstacles = False   # For checking obstacles are currently frozen
        self.frozen_time = 0    # Time remaining for freeze effect
        self.collected_keys = 0 # Number of keys collected so far
        self.collected_powerups = 0 # Number of powerups collected
        self.moves_taken = 0    #number of moves taken
        self.start_time = None  #track time when game starts
        self.score = 0           #track score
        self.game_finished = False  #check for game completion(goal reached)
        self.total_keys_required = 0    # Number of keys required to reach goal
        
        self.generate_maze() # Generate the maze structure and populate it with entities
    #2. Initializing
    def _initialize_game_state(self):
        self.wall_breaks = 2
        self.freeze_available = 2
        self.reveal_path_available = 2
        self.frozen_obstacles = False
        self.frozen_time = 0
        self.collected_keys = 0
        self.collected_powerups = 0
        self.moves_taken = 0
        self.score = 0
        self.start_time = None
        self.revealed_path_segments = []
        self.game_finished = False
        #initial values after resetting/replaying the game
        self.agent_pos = self.initial_agent_pos 
        self.goal_pos = self.initial_goal_pos
        self.keys = copy.deepcopy(self.initial_keys)
        self.powerups = copy.deepcopy(self.initial_powerups)
        self.obstacles = copy.deepcopy(self.initial_obstacles)
        self.obstacle_direction = {} 
        for pos in self.obstacles:
            self.obstacle_direction[pos] = random.choice([
                (0, 1),  # right
                (1, 0),  # down
                (0, -1), # left
                (-1, 0)  # up
            ])
        self.total_keys_required = self.initial_total_keys_required
        self.grid = copy.deepcopy(self.initial_grid_config) if self.initial_grid_config else []

    #3. Generation of maze
    def generate_maze(self):
        
        #generating multiple paths so user/AI can pursue optimal one
        self._generate_maze_with_multiple_paths()
        
        #blocking all side of maze
        for x in range(self.width):
            self.grid[0][x] = 1
            self.grid[self.height - 1][x] = 1
        for y in range(self.height):
            self.grid[y][0] = 1
            self.grid[y][self.width - 1] = 1
        
        #storing initial config for replay
        self.initial_grid_config = copy.deepcopy(self.grid)

        #setting initial position for agent and goal
        self.initial_agent_pos = (1,1) 
        self.initial_goal_pos = (self.width - 2, self.height - 2)

        #placing keys
        self.initial_keys = []
        key_count = random.randint(2, 3)
        self.initial_total_keys_required = key_count
        
        # avoiding (powerups,keys,obstacles) to be placed on similar position of agent and goal
        temp_exclude = {self.initial_agent_pos, self.initial_goal_pos}

        for _ in range(key_count):
            pos = self.get_random_empty_cell(exclude=temp_exclude)
            if pos:
                self.initial_keys.append(pos)
                temp_exclude.add(pos)
        
        # placing powerups
        self.initial_powerups = []
        for _ in range(random.randint(4, 5)):
            pos = self.get_random_empty_cell(exclude=temp_exclude)
            if pos:
                self.initial_powerups.append(pos)
                temp_exclude.add(pos)
        
        # placing obstacles
        self.initial_obstacles = []
        # obstacle_direction is part of dynamic state, initialized in _initialize_game_state
        for _ in range(random.randint(5, 6)):
            pos = self.get_random_empty_cell(exclude=temp_exclude)
            if pos:
                self.initial_obstacles.append(pos)
        
        #initialize the game based on these newly generated initial settings
        self._initialize_game_state()
        
        # checks the maze is solvable (at least to the first key or goal if no keys)
        initial_target = self.initial_goal_pos
        if self.initial_keys:
            initial_target = self.initial_keys[0]

        if not self._has_path(self.initial_agent_pos, initial_target):
            print("Warning: Initial path check failed. Regenerating maze.")
            self.generate_maze()

    #4. applies when user want to replay the same maze
    def reset_state_for_replay(self):
        if self.initial_grid_config is None:
            print("Error: No initial configuration to replay. Generating new maze.")
            self.generate_maze() # This will set up initial_grid_config and then call _initialize_game_state
            return
        self._initialize_game_state()

    #5. create atleast 2 paths in maze within 20 attempts
    def _generate_maze_with_multiple_paths(self, max_attempts=20):
        for _ in range(max_attempts):
            self.grid = [[1 for _ in range(self.width)] for _ in range(self.height)]
            self._create_path_with_dfs((1,1)) 
            self._create_additional_path()
            self.grid[1][1] = 0 
            self.grid[self.height - 2][self.width - 2] = 0
            
            if self._count_paths((1,1), (self.width - 2, self.height - 2)) >= 2:
                return
        print("Warning: Could not guarantee multiple paths within attempts.")

    #6. used to create paths in maze
    def _create_path_with_dfs(self, start_pos):
        stack = [start_pos]
        visited = {start_pos}
        self.grid[start_pos[1]][start_pos[0]] = 0
        
        while stack:
            x, y = stack[-1]
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = x + dx, y + dy 
                if (1 <= nx < self.width - 1 and 1 <= ny < self.height - 1 and
                    (nx, ny) not in visited):
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = random.choice(neighbors)
                self.grid[y + (ny - y) // 2][x + (nx - x) // 2] = 0
                self.grid[ny][nx] = 0
                visited.add((nx, ny))
                stack.append((nx, ny))
            else:
                stack.pop()
        
        #confirms that there is a path to goal
        gx, gy = self.width - 2, self.height - 2
        if self.grid[gy][gx] == 1:
            self.grid[gy][gx] = 0
            for dx_goal, dy_goal in [(0,1), (1,0), (0,-1), (-1,0)]:
                nnx, nny = gx + dx_goal, gy + dy_goal
                if 0 <= nnx < self.width and 0 <= nny < self.height and self.grid[nny][nnx] == 0:
                    break 
            else: 
                curr_x, curr_y = gx, gy
                while curr_x > 1 or curr_y > 1: 
                    if curr_x > 1 and (curr_x % 2 == 0 or self.grid[curr_y][curr_x-1] == 1):
                        self.grid[curr_y][curr_x-1] = 0
                        curr_x -=1
                    elif curr_y > 1 and (curr_y % 2 == 0 or self.grid[curr_y-1][curr_x] == 1):
                         self.grid[curr_y-1][curr_x] = 0
                         curr_y -=1
                    else: 
                        if random.choice([True, False]) and curr_x > 1: curr_x -=1
                        elif curr_y > 1: curr_y -=1
                        else: break
                        if 0 <= curr_y < self.height and 0 <= curr_x < self.width:
                            self.grid[curr_y][curr_x] = 0
                        else: break

    #7. create more paths in maze by removing walls to increase complexity
    def _create_additional_path(self):
        num_walls_to_remove = int((self.width * self.height) * 0.05)
        
        for _ in range(num_walls_to_remove):
            wall_candidates = []
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.grid[y][x] == 1: 
                        if x > 0 and x < self.width -1 and self.grid[y][x-1] == 0 and self.grid[y][x+1] == 0:
                             wall_candidates.append((x,y))
                             continue
                        if y > 0 and y < self.height -1 and self.grid[y-1][x] == 0 and self.grid[y+1][x] == 0:
                             wall_candidates.append((x,y))
                             continue
            
            if wall_candidates:
                wx, wy = random.choice(wall_candidates)
                self.grid[wy][wx] = 0
    
    #8.count path to be atleast 2
    def _count_paths(self, start, goal, max_paths_to_find=2):
        path1 = self.bfs_to_target(start, goal, ignore_keys=True)
        if not path1:
            return 0
        if len(path1) <= 3: 
            return 1

        original_grid_val = self.grid[path1[len(path1)//2][1]][path1[len(path1)//2][0]]
        self.grid[path1[len(path1)//2][1]][path1[len(path1)//2][0]] = 1 
        
        path2 = self.bfs_to_target(start, goal, ignore_keys=True)
        
        self.grid[path1[len(path1)//2][1]][path1[len(path1)//2][0]] = original_grid_val
        
        return 2 if path2 else 1
    
    #9. verifies that path exists between start and goal 
    def _has_path(self, start, goal):
        return bool(self.bfs_to_target(start, goal, ignore_keys=True))
    
    #10. for getting random empty cell     
    def get_random_empty_cell(self, exclude=None):
        if exclude is None: exclude = set()
        empty_cells = []
        for y in range(1, self.height - 1): 
            for x in range(1, self.width - 1):
                if self.grid and y < len(self.grid) and x < len(self.grid[y]) and \
                   self.grid[y][x] == 0 and (x, y) not in exclude:
                    empty_cells.append((x, y))
        return random.choice(empty_cells) if empty_cells else None
    
    #11. agent,goal, keys and powerups assigned to special cells    
    def all_special_cells(self):
        special_cells = {self.agent_pos, self.goal_pos}
        special_cells.update(self.keys)
        special_cells.update(self.powerups)
        return special_cells
    
    #12. break wall functionality
    def break_wall(self, wall_pos):
        x, y = wall_pos
        if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
            return False, "Cannot break boundary walls."

        if 0 < x < self.width-1 and 0 < y < self.height-1 and self.grid[y][x] == 1:
            if self.wall_breaks > 0:
                self.grid[y][x] = 0
                self.wall_breaks -= 1
                return True, "Wall broken!"
            return False, "No wall breaks left."
        return False, "Not a valid wall to break."
    
    #13. possible movements for agent and obstacles
    def get_valid_neighbors(self, pos, for_obstacle=False):
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                is_obstacle_collision = (nx,ny) in self.obstacles and not self.frozen_obstacles and not for_obstacle
                if self.grid[ny][nx] == 0 and not is_obstacle_collision:
                    neighbors.append((nx, ny))
        return neighbors
    
    #14. confirms valid move
    def is_valid_move(self, pos):
        x, y = pos
        if not (0 <= x < self.width and 0 <= y < self.height): return False
        if self.grid[y][x] == 1: return False 
        if (x,y) in self.obstacles and not self.frozen_obstacles: return False 
        return True
    
    #15. freeze obstacles functionality 
    def use_freeze_obstacles(self):
        if self.freeze_available > 0 and not self.frozen_obstacles:
            self.frozen_obstacles = True
            self.freeze_available -= 1
            self.frozen_time = time.time()
            return True, "Obstacles frozen for 10 seconds!"
        elif self.frozen_obstacles:
            return False, "Obstacles are already frozen."
        return False, "No freeze power-ups left."
    
    #16. reveal path functionality (uses a*)         
    def use_reveal_path(self, algorithm_name="a_star"):
        if self.reveal_path_available <= 0:
            return False, "No reveal path lifelines left."
        path = []
        if algorithm_name == "bfs": path = self._bfs_solve_path(self.agent_pos, self.collected_keys, copy.deepcopy(self.keys))
        elif algorithm_name == "dfs": path = self._dfs_solve_path(self.agent_pos, self.collected_keys, copy.deepcopy(self.keys))
        elif algorithm_name == "greedy": path = self._greedy_solve_path(self.agent_pos, self.collected_keys, copy.deepcopy(self.keys))
        else: path = self._a_star_solve_path(self.agent_pos, self.collected_keys, copy.deepcopy(self.keys))
            
        if path:
            self.revealed_path_segments = path[:min(10, len(path))] 
            self.reveal_path_available -= 1
            return True, f"Path revealed using {algorithm_name}!"
        return False, "Could not find a path to reveal."
    
    #17. Agent movement tracking         
    def move_agent(self, new_pos):
        if self.game_finished: return None

        if not self.start_time:
            self.start_time = time.time()
            
        pickup_message = None
        if new_pos in self.keys:
            self.keys.remove(new_pos)
            self.collected_keys += 1
            self.score += 50
            pickup_message = f"Key collected! ({self.collected_keys}/{self.total_keys_required})"
            
        if new_pos in self.powerups:
            self.powerups.remove(new_pos)
            self.collected_powerups += 1
            self.score += 30
            rand_effect = random.randint(0, 2)
            if rand_effect == 0 and self.wall_breaks < 3:#--
                self.wall_breaks += 1
                pickup_message = "Gained a Wall Break!"
            elif rand_effect == 1 and self.freeze_available < 3: #--
                self.freeze_available += 1
                pickup_message = "Gained a Freeze Obstacles!"
            elif self.reveal_path_available < 3:#--
                self.reveal_path_available += 1
                pickup_message = "Gained a Reveal Path!"
            else: 
                self.score += 20 
                pickup_message = "Bonus points!"
            
        self.agent_pos = new_pos
        self.moves_taken += 1
        self.revealed_path_segments = [] 
        
        if new_pos == self.goal_pos:
            if self.collected_keys >= self.total_keys_required:
                self.finish_game()
                return "Goal Reached! You Win!"
            else:
                return f"Reached goal, but need {self.total_keys_required - self.collected_keys} more key(s)!"
            
        return pickup_message 
    
    #18. finish game logic        
    def finish_game(self, simulated_time_override=None):
        if not self.game_finished:
            self.game_finished = True
            
            if simulated_time_override is not None:
                completion_time = simulated_time_override
            elif self.start_time:
                completion_time = time.time() - self.start_time
            else:
                completion_time = 0

            #bonus calculation initialized
            time_bonus = max(0, 1000 - int(completion_time * 5)) 
            move_bonus = max(0, 500 - self.moves_taken * 5)
            self.score += 500 + time_bonus + move_bonus 
    
    #19. Lifelines timer logic (for freeze obstacles)             
    def update_lifeline_timers(self):
        if self.frozen_obstacles and (time.time() - self.frozen_time > 10):
            self.frozen_obstacles = False
            return "Obstacles unfrozen."
        return None
    
    #20. obtacles movement logic
    def move_obstacles(self):
        if self.frozen_obstacles or self.game_finished:
            return False

        new_obstacles = []
        new_directions = {}
        collision_occurred = False

        for pos in self.obstacles:
            x, y = pos
            dx, dy = self.obstacle_direction.get(pos, random.choice([(0,1),(1,0),(0,-1),(-1,0)]))
            
            moved_this_turn = False
            for attempt in range(4): 
                nx, ny = x + dx, y + dy
                
                if (0 < nx < self.width - 1 and 0 < ny < self.height - 1 and 
                    self.grid[ny][nx] == 0 and 
                    (nx, ny) != self.goal_pos and   # obstacles shouldn't destroy goal
                    (nx, ny) not in self.keys and   # obstacles shouldn't destroy keys
                    (nx, ny) not in self.powerups and # Obstacles shouldn't destroy powerups
                    (nx, ny) not in new_obstacles): # Avoid collision with other moving obstacles in same step
                    
                    new_obstacles.append((nx, ny))
                    new_directions[(nx, ny)] = (dx, dy)
                    moved_this_turn = True
                    
                    if (nx,ny) == self.agent_pos: 
                        collision_occurred = True
                    break 
                else:   #turn right
                    dx, dy = -dy, dx 
                    if attempt == 3:
                        new_obstacles.append(pos)
                        new_directions[pos] = self.obstacle_direction.get(pos, (dx,dy)) 
            
            if not moved_this_turn and pos not in new_obstacles :
                 new_obstacles.append(pos)
                 new_directions[pos] = self.obstacle_direction.get(pos, (dx,dy))


        self.obstacles = new_obstacles
        self.obstacle_direction = new_directions
        
        if collision_occurred:
            self.score = max(0, self.score - 100) # if collision occured, -100 is the penalty but score doesn't go below 0 from this
            return True
        return False


    # --- Path-finding Algorithms ---
    
    #21. will be called after collection of all keys
    def bfs_to_target(self, start, target, ignore_keys=False): 
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            if current == target:
                return path + [current]
                
            for neighbor in self.get_valid_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [current]))
        return []
    
    #22. complete logic of bfs
    def _bfs_solve_path(self, start_pos, current_keys_collected, keys_list_remaining):
        path_to_keys = []
        temp_agent_pos = start_pos
        temp_keys_collected_count = current_keys_collected
        temp_keys_map_list = copy.deepcopy(keys_list_remaining)

        current_path_segment_starts_with_start_pos = True

        while temp_keys_collected_count < self.total_keys_required:
            if not temp_keys_map_list: break 
            
            shortest_path_to_key_segment = []
            best_key_pos = None

            for key_pos in temp_keys_map_list:
                path_segment = self.bfs_to_target(temp_agent_pos, key_pos)
                if path_segment and (not shortest_path_to_key_segment or len(path_segment) < len(shortest_path_to_key_segment)):
                    shortest_path_to_key_segment = path_segment
                    best_key_pos = key_pos
            
            if shortest_path_to_key_segment:
                to_extend = shortest_path_to_key_segment if current_path_segment_starts_with_start_pos else shortest_path_to_key_segment[1:]
                path_to_keys.extend(to_extend)
                current_path_segment_starts_with_start_pos = False 
                
                temp_agent_pos = best_key_pos
                temp_keys_map_list.remove(best_key_pos)
                temp_keys_collected_count +=1 
            else:
                return [] 

        path_to_goal_segment = self.bfs_to_target(temp_agent_pos, self.goal_pos)
        if path_to_goal_segment:
            to_extend = path_to_goal_segment if current_path_segment_starts_with_start_pos else path_to_goal_segment[1:]
            full_path = path_to_keys + to_extend
            
            if not full_path or full_path[0] != start_pos:
                if start_pos == self.goal_pos and temp_keys_collected_count >= self.total_keys_required: return [start_pos]
            return full_path if full_path else ([start_pos] if start_pos == self.goal_pos and temp_keys_collected_count >= self.total_keys_required else [])

        elif not path_to_keys and temp_keys_collected_count >= self.total_keys_required and start_pos == self.goal_pos:
             return [start_pos]
        return []

    #23.  will be called after collection of all keys
    def dfs_to_target(self, start, target):
        stack = [(start, [])]
        visited = set()
        while stack:
            current, path = stack.pop()
            if current == target: return path + [current]
            if current in visited: continue
            visited.add(current)
            neighbors = self.get_valid_neighbors(current)
            random.shuffle(neighbors)
            for neighbor in neighbors: 
                if neighbor not in visited:
                    stack.append((neighbor, path + [current]))
        return []
    #24. DFS: mostly not optimal ,just for visualizing that how it works
    def _dfs_solve_path(self, start_pos, current_keys_collected_count_player, keys_list_player_needs):
        # State for memorization: (current_position, setof_remaining_initial_keys_to_find) for caching
        memo = {}
        
        player_has_already_collected_mask = 0
        for i, ik_pos in enumerate(self.initial_keys):
            if ik_pos not in keys_list_player_needs:
                player_has_already_collected_mask |= (1 << i)

        def solve_recursive_dfs(current_pos_sim, collected_mask_sim):
            state = (current_pos_sim, collected_mask_sim)
            if state in memo: return memo[state]

            # check that all keys are in the mask
            num_keys_in_mask = bin(collected_mask_sim).count('1')
            if num_keys_in_mask >= self.total_keys_required:
                path_to_goal_segment = self.dfs_to_target(current_pos_sim, self.goal_pos)
                memo[state] = path_to_goal_segment
                return path_to_goal_segment

            # find paths to remaining (unmasked) initial keys
            potential_next_keys_to_target = []
            for i, key_pos_initial in enumerate(self.initial_keys):
                if not (collected_mask_sim & (1 << i)): # If key 'i' is not in mask yet
                    potential_next_keys_to_target.append((key_pos_initial, i)) # (pos, index)
            
            for key_pos_target, key_idx_target in potential_next_keys_to_target:
                path_to_this_key_segment = self.dfs_to_target(current_pos_sim, key_pos_target)
                if path_to_this_key_segment:
                    remaining_path_segment = solve_recursive_dfs(key_pos_target, collected_mask_sim | (1 << key_idx_target))
                    if remaining_path_segment:
                        full_segment_path = path_to_this_key_segment[:-1] + remaining_path_segment
                        memo[state] = full_segment_path 
                        return full_segment_path
            
            memo[state] = [] 
            return []

        full_simulated_path = solve_recursive_dfs(start_pos, player_has_already_collected_mask)

        if not full_simulated_path:
            if start_pos == self.goal_pos and bin(player_has_already_collected_mask).count('1') >= self.total_keys_required:
                return [start_pos]
            return []
        
        if full_simulated_path[0] != start_pos :
             pass 

        return full_simulated_path

    #25. distance calculation
    def manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    #26.  will be called after collection of all keys
    def a_star_to_target(self, start, target):
        open_set = [(0 + self.manhattan_distance(start, target), 0, start, [])] 
        g_scores = {start: 0}

        queue = [(0 + self.manhattan_distance(start, target), 0, start, [start])] 
        visited_g_scores = {start: 0}

        while queue:
            f, g, current, path = heapq.heappop(queue)

            if g > visited_g_scores.get(current, float('inf')):
                continue

            if current == target: return path

            for neighbor in self.get_valid_neighbors(current):
                tentative_g = g + 1
                if tentative_g < visited_g_scores.get(neighbor, float('inf')):
                    visited_g_scores[neighbor] = tentative_g
                    h = self.manhattan_distance(neighbor, target)
                    heapq.heappush(queue, (tentative_g + h, tentative_g, neighbor, path + [neighbor]))
        return []

    #27. A* algorithm
    def _a_star_solve_path(self, start_pos, current_keys_collected_count, keys_list_remaining):
        path_accumulator = []
        current_pos_sim = start_pos
        keys_collected_sim_count = current_keys_collected_count
        remaining_keys_sim_list = copy.deepcopy(keys_list_remaining)
        
        first_segment = True

        while keys_collected_sim_count < self.total_keys_required:
            if not remaining_keys_sim_list: break
            
            best_path_to_key_segment = []
            chosen_key_pos = None

            for key_pos_target in remaining_keys_sim_list:
                segment = self.a_star_to_target(current_pos_sim, key_pos_target)
                if segment and (not best_path_to_key_segment or len(segment) < len(best_path_to_key_segment)):
                    best_path_to_key_segment = segment
                    chosen_key_pos = key_pos_target
            
            if best_path_to_key_segment:
                path_accumulator.extend(best_path_to_key_segment if first_segment else best_path_to_key_segment[1:])
                first_segment = False
                current_pos_sim = chosen_key_pos
                remaining_keys_sim_list.remove(chosen_key_pos)
                keys_collected_sim_count += 1
            else:
                return [] 

        path_to_goal_segment = self.a_star_to_target(current_pos_sim, self.goal_pos)
        if path_to_goal_segment:
            path_accumulator.extend(path_to_goal_segment if first_segment else path_to_goal_segment[1:])
            
            if not path_accumulator and start_pos == self.goal_pos and keys_collected_sim_count >= self.total_keys_required:
                return [start_pos] # Already at goal, all keys collected
            return path_accumulator
        
        elif not path_accumulator and current_pos_sim == self.goal_pos and keys_collected_sim_count >= self.total_keys_required:
            if start_pos == self.goal_pos: return [start_pos] # Started at goal, no keys needed
        return [] if not path_accumulator and not (start_pos == self.goal_pos and keys_collected_sim_count >= self.total_keys_required) else path_accumulator

    #28.  will be called after collection of all keys
    def greedy_to_target(self, start, target): 
        queue = [(self.manhattan_distance(start, target), start, [start])] 
        visited = {start} # Greedy only needs to visit each node once

        while queue:
            _, current, path = heapq.heappop(queue)

            if current == target: return path

            # Sort neighbors by heuristic to target for Greedy
            neighbors_sorted = sorted(
                self.get_valid_neighbors(current),
                key=lambda n: self.manhattan_distance(n, target)
            )

            for neighbor in neighbors_sorted:
                if neighbor not in visited:
                    visited.add(neighbor)
                    heapq.heappush(queue, (self.manhattan_distance(neighbor, target), neighbor, path + [neighbor]))
        return []
    
    #29. Greedy: pick closest key by heuristic first
    def _greedy_solve_path(self, start_pos, current_keys_collected_count, keys_list_remaining):
        path_accumulator = []
        current_pos_sim = start_pos
        keys_collected_sim_count = current_keys_collected_count
        remaining_keys_sim_list = copy.deepcopy(keys_list_remaining)
        
        first_segment = True

        while keys_collected_sim_count < self.total_keys_required:
            if not remaining_keys_sim_list: break
            remaining_keys_sim_list.sort(key=lambda k_pos: self.manhattan_distance(current_pos_sim, k_pos))
            
            if not remaining_keys_sim_list: break 
            
            target_key = remaining_keys_sim_list[0] 
            segment = self.greedy_to_target(current_pos_sim, target_key)

            if segment:
                path_accumulator.extend(segment if first_segment else segment[1:])
                first_segment = False
                current_pos_sim = target_key
                remaining_keys_sim_list.remove(target_key)
                keys_collected_sim_count += 1
            else:
                return []

        path_to_goal_segment = self.greedy_to_target(current_pos_sim, self.goal_pos)
        if path_to_goal_segment:
            path_accumulator.extend(path_to_goal_segment if first_segment else path_to_goal_segment[1:])
            if not path_accumulator and start_pos == self.goal_pos and keys_collected_sim_count >= self.total_keys_required:
                return [start_pos]
            return path_accumulator
        
        elif not path_accumulator and current_pos_sim == self.goal_pos and keys_collected_sim_count >= self.total_keys_required :
             if start_pos == self.goal_pos: return [start_pos]
        
        return [] if not path_accumulator and not (start_pos == self.goal_pos and keys_collected_sim_count >= self.total_keys_required) else path_accumulator

    #30. Save paths of all algos
    def get_path_for_algorithm(self, algorithm_name):
        current_remaining_keys = copy.deepcopy(self.keys)
        if algorithm_name == "bfs":
            return self._bfs_solve_path(self.agent_pos, self.collected_keys, current_remaining_keys)
        elif algorithm_name == "dfs":
            return self._dfs_solve_path(self.agent_pos, self.collected_keys, current_remaining_keys)
        elif algorithm_name == "a_star":
            return self._a_star_solve_path(self.agent_pos, self.collected_keys, current_remaining_keys)
        elif algorithm_name == "greedy":
            return self._greedy_solve_path(self.agent_pos, self.collected_keys, current_remaining_keys)
        return []

    #31. finding optimal one
    def find_optimal_algorithm(self):
        algorithms = ["bfs", "a_star", "greedy"] 
        best_algo = None
        best_moves = float('inf')
        
        original_env_state = self.snapshot_state()

        for algo in algorithms:
            path = self.get_path_for_algorithm(algo) 
            moves = len(path) -1 if path else float('inf')
            
            self.algorithm_stats[algo] = {"moves": moves if moves != float('inf') else -1, "path_length": moves if moves != float('inf') else -1}
            
            if moves != float('inf') and moves < best_moves:
                best_moves = moves
                best_algo = algo
        
        self.restore_state(original_env_state) 
        return best_algo if best_algo else "a_star" # Default to A* if no path found byany
    #32. For saving the state of the game
    def snapshot_state(self):
        return {
            'agent_pos': self.agent_pos,
            'keys': copy.deepcopy(self.keys),
            'collected_keys': self.collected_keys,
            'total_keys_required': self.total_keys_required,
            'powerups': copy.deepcopy(self.powerups),
            'obstacles': copy.deepcopy(self.obstacles),
            'obstacle_direction': copy.deepcopy(self.obstacle_direction),
            'game_finished': self.game_finished,
            'score': self.score,
            'moves_taken': self.moves_taken,
            'wall_breaks': self.wall_breaks,
            'freeze_available': self.freeze_available,
            'reveal_path_available': self.reveal_path_available,
            'frozen_obstacles': self.frozen_obstacles,
            'frozen_time': self.frozen_time,
            'start_time': self.start_time, 
            'grid': copy.deepcopy(self.grid),
            'initial_grid_config': copy.deepcopy(self.initial_grid_config),
            'initial_agent_pos': self.initial_agent_pos,
            'initial_goal_pos': self.initial_goal_pos,
            'initial_keys': copy.deepcopy(self.initial_keys),
            'initial_powerups': copy.deepcopy(self.initial_powerups),
            'initial_obstacles': copy.deepcopy(self.initial_obstacles),
            'initial_total_keys_required': self.initial_total_keys_required,
        }
    #33. for restoring the state of the game
    def restore_state(self, snapshot):
        self.agent_pos = snapshot['agent_pos']
        self.keys = snapshot['keys']
        self.collected_keys = snapshot['collected_keys']
        self.total_keys_required = snapshot['total_keys_required']
        self.powerups = snapshot['powerups']
        self.obstacles = snapshot['obstacles']
        self.obstacle_direction = snapshot['obstacle_direction']
        self.game_finished = snapshot['game_finished']
        self.score = snapshot['score']
        self.moves_taken = snapshot['moves_taken']
        self.wall_breaks = snapshot['wall_breaks']
        self.freeze_available = snapshot['freeze_available']
        self.reveal_path_available = snapshot['reveal_path_available']
        self.frozen_obstacles = snapshot['frozen_obstacles']
        self.frozen_time = snapshot['frozen_time']
        self.start_time = snapshot['start_time']
        self.grid = snapshot['grid']

        self.initial_grid_config = snapshot.get('initial_grid_config', self.initial_grid_config)
        self.initial_agent_pos = snapshot.get('initial_agent_pos', self.initial_agent_pos)
        self.initial_goal_pos = snapshot.get('initial_goal_pos', self.initial_goal_pos)
        self.initial_keys = snapshot.get('initial_keys', self.initial_keys)
        self.initial_powerups = snapshot.get('initial_powerups', self.initial_powerups)
        self.initial_obstacles = snapshot.get('initial_obstacles', self.initial_obstacles)
        self.initial_total_keys_required = snapshot.get('initial_total_keys_required', self.initial_total_keys_required)

    #34. user can see the path of the algo (lifelines are not used and obstacles become static)
    def simulate_ai(self, algorithm):
        original_state_snapshot = self.snapshot_state()
        
        sim_collected_keys_count = 0
        sim_collected_powerups_count = 0
        
        self.moves_taken = 0 
        self.game_finished = False 
        self.score = 0
        
       

        calculated_path = self.get_path_for_algorithm(algorithm) 
        
        sim_path_followed = []

        if calculated_path and len(calculated_path) > 1:
            sim_path_followed.append(calculated_path[0]) 
            for i in range(1, len(calculated_path)):
                pos = calculated_path[i]
                self.agent_pos = pos 
                self.moves_taken += 1
                sim_path_followed.append(pos)

                if pos in self.keys: 
                    self.keys.remove(pos) 
                    self.collected_keys += 1 
                    sim_collected_keys_count +=1
                
                if pos in self.powerups: 
                    self.powerups.remove(pos)
                    sim_collected_powerups_count +=1

                if pos == self.goal_pos and self.collected_keys >= self.total_keys_required:
                    self.finish_game(simulated_time_override=0) 
                    break 
        elif calculated_path and len(calculated_path) == 1 and calculated_path[0] == self.goal_pos: 
             if self.collected_keys >= self.total_keys_required:
                self.finish_game(simulated_time_override=0)
             sim_path_followed.append(calculated_path[0])

        # Calculate score for simulation:
        # Score = (keys_score) + (powerups_score) + (win_bonus if won) + (move_bonus if won)
        sim_score_final = 0
        sim_score_final += sim_collected_keys_count * 50
        sim_score_final += sim_collected_powerups_count * 30
        
        if self.game_finished: 
            sim_score_final += 500 
            move_bonus = max(0, 500 - self.moves_taken * 5)
            sim_score_final += move_bonus
        
        sim_moves_final = self.moves_taken
        
        self.restore_state(original_state_snapshot) 
        
        return sim_moves_final, sim_score_final, sim_path_followed

#--2nd class: GUI--
class MazeApp:
    #1.Constructor (initial window setup)
    def __init__(self, root_tk):
        self.root = root_tk # Store the main Tkinter root window
        self.root.title("Maze Solver AI Challenge") # Set the window title
        self.root.configure(bg="#101010")  # Black background

        self.env = MazeEnvironment(GRID_WIDTH, GRID_HEIGHT) # Create the maze logic environment (backend)

        # AI control states
        self.ai_animating = False   #checks whether AI is currently moving step-by-step
        self.ai_path_to_animate = []     # The full path AI will follow
        self.ai_animation_paused = False     # Is AI movement paused
        self.ai_current_path_taken = [] # Stores positions already taken by AI
        self.wall_break_mode = False      # If player is currently in wall-breaking mode
        self.obstacle_move_interval = 2000  # Time interval (ms) for obstacle movement
        
        # Comparison UI window variables
        self.compare_window = None # Placeholder for comparison popup window
        self.compare_tree = None #Table view for algorithm comparison
        self.compare_update_id = None   # Used to manage periodic updates of the table
                
        self.status_message = tk.StringVar(value="Welcome to the Maze Challenge!") # Message showed at the bottom text box

        # --- Main Layout ---
        self.main_frame = tk.Frame(root_tk, bg="#101010")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.left_frame = tk.Frame(self.main_frame, bg="#101010")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.right_frame = tk.Frame(self.main_frame, width=500, bg="#181c1b")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        self.right_frame.pack_propagate(False)
        
        # --- Right Panel: Stats & feature_key ---
        self.right_center_container = tk.Frame(self.right_frame, bg="#181c1b")
        self.right_center_container.pack(expand=True, fill=tk.BOTH)

        # --- Top Controls (Left Frame) ---
        self.controls_frame = tk.LabelFrame(
            self.left_frame, text="AI Controls", bg="#181c1b", fg="#22c55e",
            padx=10, pady=10, font=("Arial", 10, "bold"), relief=tk.GROOVE
        )
        self.controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # --- AI Control Buttons ---
        tk.Label(self.controls_frame, text="Algorithm:", bg="#181c1b", fg="#22c55e").pack(side=tk.LEFT, padx=(0, 5))
        self.algorithm_var = tk.StringVar(value="a_star")
        self.algorithm_menu = ttk.Combobox(
            self.controls_frame, textvariable=self.algorithm_var,
            values=["bfs", "dfs", "a_star", "greedy"], state="readonly", width=10, font=("Arial", 10)
        )
        self.algorithm_menu.pack(side=tk.LEFT, padx=(0, 10))
        self.algorithm_menu.bind("<<ComboboxSelected>>", lambda e: self.canvas.focus_set())
        self.run_ai_button = tk.Button(
            self.controls_frame, text="Run AI", command=self.run_ai_agent,
            bg="#22c55e", fg="#101010", width=9, font=("Arial", 10, "bold"),
            relief=tk.RAISED, borderwidth=2, activebackground="#16a34a"
        )
        self.run_ai_button.pack(side=tk.LEFT, padx=3)

        self.pause_ai_button = tk.Button(
            self.controls_frame, text="Pause AI", command=self.toggle_ai_pause, state=tk.DISABLED,
            bg="#a3e635", fg="#101010", width=9, font=("Arial", 10, "bold"),
            relief=tk.RAISED, borderwidth=2, activebackground="#bef264"
        )
        self.pause_ai_button.pack(side=tk.LEFT, padx=3)

        self.optimal_button = tk.Button(
            self.controls_frame, text="Optimal Solver", command=self.run_optimal_solver,
            bg="#166534", fg="#e5ffe5", width=12, font=("Arial", 10, "bold"),
            relief=tk.RAISED, borderwidth=2, activebackground="#22c55e"
        )
        self.optimal_button.pack(side=tk.LEFT, padx=3)

        

        # --- Canvas (Left Frame) ---
        self.canvas_frame = tk.Frame(self.left_frame, bg="#101010", bd=2, relief=tk.SUNKEN)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=GRID_WIDTH * CELL_SIZE, height=GRID_HEIGHT * CELL_SIZE,
            bg="#181c1b", highlightthickness=0
        )
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.handle_canvas_click)

        # --- Lifelines (Left Frame) ---
        self.lifelines_frame = tk.LabelFrame(
            self.left_frame, text="Player Lifelines", bg="#181c1b", fg="#22c55e",
            padx=10, pady=10, font=("Arial", 10, "bold"), relief=tk.GROOVE
        )
        self.lifelines_frame.pack(fill=tk.X, pady=(0, 10))

        btn_font = ("Arial", 10, "bold")
        self.wall_break_button = tk.Button(
            self.lifelines_frame, text="Wall Break (0)", command=self.activate_wall_break_mode,
            bg="#22c55e", fg="#101010", font=btn_font, relief=tk.RAISED, borderwidth=2, activebackground="#16a34a"
        )
        self.wall_break_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.freeze_button = tk.Button(
            self.lifelines_frame, text="Freeze (0)", command=self.use_freeze_ui,
            bg="#22c55e", fg="#101010", font=btn_font, relief=tk.RAISED, borderwidth=2, activebackground="#16a34a"
        )
        self.freeze_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.reveal_path_button = tk.Button(
            self.lifelines_frame, text="Reveal Path (0)", command=self.use_reveal_path_ui,
            bg="#22c55e", fg="#101010", font=btn_font, relief=tk.RAISED, borderwidth=2, activebackground="#16a34a"
        )
        self.reveal_path_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # --- Game/Reset Controls (Left Frame - Bottom) ---
        self.game_controls_frame = tk.Frame(self.left_frame, bg="#101010")
        self.game_controls_frame.pack(fill=tk.X, pady=(0, 5))

        self.replay_button = tk.Button(
            self.game_controls_frame, text="Replay Maze", command=self.replay_maze,
            bg="#166534", fg="#e5ffe5", font=btn_font, relief=tk.RAISED, borderwidth=2, activebackground="#22c55e"
        )
        self.replay_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        self.reset_button = tk.Button(
            self.game_controls_frame, text="New Maze", command=self.reset_maze_ui,
            bg="#dc2626", fg="#e5ffe5", font=btn_font, relief=tk.RAISED, borderwidth=2, activebackground="#991b1b"
        )
        self.reset_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # --- Status Message (Left Frame - Bottom) ---
        self.status_label = tk.Label(
            self.left_frame, textvariable=self.status_message,
            bg="#181c1b", fg="#22c55e", relief=tk.SUNKEN, anchor="w", padx=5, font=("Arial", 10)
        )
        self.status_label.pack(fill=tk.X, ipady=3)

        

        stat_font = ("Arial", 11)
        self.stats_frame = tk.LabelFrame(
            self.right_center_container, text="Statistics", bg="#181c1b", padx=10, pady=10,
            relief=tk.RIDGE, bd=2, font=("Arial", 10, "bold"), fg="#22c55e"
        )
        self.stats_frame.pack(fill=tk.X, pady=(0, 10), expand=True)

        self.score_label = tk.Label(self.stats_frame, text="Score: 0", bg="#181c1b", font=stat_font, fg="#e5ffe5")
        self.score_label.pack(anchor="w", pady=1)
        self.moves_label = tk.Label(self.stats_frame, text="Moves: 0", bg="#181c1b", font=stat_font, fg="#e5ffe5")
        self.moves_label.pack(anchor="w", pady=1)
        self.keys_label = tk.Label(self.stats_frame, text="Keys: 0/0", bg="#181c1b", font=stat_font, fg="#e5ffe5")
        self.keys_label.pack(anchor="w", pady=1)
        self.powerups_label = tk.Label(self.stats_frame, text="Powerups: 0", bg="#181c1b", font=stat_font, fg="#e5ffe5")
        self.powerups_label.pack(anchor="w", pady=1)
        self.time_label = tk.Label(self.stats_frame, text="Time: 0s", bg="#181c1b", font=stat_font, fg="#e5ffe5")
        self.time_label.pack(anchor="w", pady=1)

        # Feature Key
        self.feature_key_frame = tk.LabelFrame(
            self.right_center_container, text="Feature Key", bg="#181c1b", padx=10, pady=10,
            relief=tk.RIDGE, bd=2, font=("Arial", 10, "bold"), fg="#22c55e"
        )
        self.feature_key_frame.pack(fill=tk.X, pady=5, expand=True)

        feature_key_items = {AGENT: "Agent", GOAL: "Goal", KEY: "Key", POWERUP: "Power-up",
                        OBSTACLE: "Obstacle", WALL: "Wall", PATH: "Path",
                        VISITED_PATH: "AI Trace", REVEALED_STEP: "Revealed Path"}
        feature_key_font = ("Arial", 10)
        for item, desc in feature_key_items.items():
            item_frame = tk.Frame(self.feature_key_frame, bg="#181c1b")
            item_frame.pack(anchor="w", fill=tk.X, pady=1)
            tk.Label(item_frame, text="  ", bg=COLOR_MAP.get(item, "gray"), relief=tk.SOLID, borderwidth=1).pack(side=tk.LEFT, padx=(0, 5))
            tk.Label(item_frame, text=f": {desc}", bg="#181c1b", font=feature_key_font, fg="#e5ffe5").pack(side=tk.LEFT)

        # --- Live Comparison Table (Bottom Right, vertically centered with others) ---
        compare_frame = tk.LabelFrame(
            self.right_center_container, text="Live Algorithm Comparison", bg="#181c1b", fg="#22c55e",
            padx=8, pady=6, font=("Arial", 10, "bold"), relief=tk.GROOVE
        )
        compare_frame.pack(fill=tk.X, pady=(10, 0), padx=2, expand=True)

        cols = ["Algorithm", "Moves", "Sim. Score", "Powerups", "Time (s)"]
        
        # Treeview Table Style
        style = ttk.Style(self.right_frame)
        style.theme_use('clam')
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"), background="#22c55e", foreground="#101010")
        style.configure("Treeview", rowheight=22, font=("Arial", 9), background="#101010", fieldbackground="#101010", foreground="#e5ffe5")

        self.compare_tree = ttk.Treeview(compare_frame, columns=cols, show="headings", selectmode="none", height=5)
        for col in cols:
            self.compare_tree.heading(col, text=col)
            self.compare_tree.column(col, width=90, anchor='center', stretch=tk.YES)
        self.compare_tree.pack(fill=tk.BOTH, expand=True)
        self.compare_tree.tag_configure("highlight", background="#166534", font=("Arial", 9, "bold"), foreground="#e5ffe5")

        self.update_compare_results()  # Start periodic update for embedded table        
        self.root.bind("<KeyPress>", self.handle_keypress)
        self.draw_maze()
        self.update_stats_display()
        self._obstacle_move_id = self.root.after(self.obstacle_move_interval, self.periodic_obstacle_move)
        self._periodic_update_id = self.root.after(100, self.periodic_update)

        self.status_message.set("Maze generated. Use arrow keys or AI to solve!")
    #2. Stay window updated (timer,lifelines, etc)
    def periodic_update(self):
        if not self.env.game_finished:
            if self.env.start_time:
                elapsed_time = time.time() - self.env.start_time
                self.time_label.config(text=f"Time: {int(elapsed_time)}s")
            
            unfreeze_msg = self.env.update_lifeline_timers()
            if unfreeze_msg:
                self.status_message.set(unfreeze_msg)
                self.draw_maze() 
                self.update_stats_display()
            
            self._periodic_update_id = self.root.after(100, self.periodic_update)
    #3. Crating maze
    def draw_maze(self):
        self.canvas.delete("all")
        for y in range(self.env.height):
            for x in range(self.env.width):
                cell_type = PATH 
                if self.env.grid[y][x] == 1: cell_type = WALL
                
                color_to_use = COLOR_MAP[cell_type]
                if cell_type == PATH: 
                    if (x,y) in self.ai_current_path_taken :
                        color_to_use = COLOR_MAP[VISITED_PATH]
                    elif (x,y) in self.env.revealed_path_segments:
                        color_to_use = COLOR_MAP[REVEALED_STEP]

                self.canvas.create_rectangle(x * CELL_SIZE, y * CELL_SIZE,
                                             (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                                             fill=color_to_use, outline="#CCCCCC", width=0.5) 
        
        items_to_draw = [
            (self.env.keys, KEY),
            (self.env.powerups, POWERUP),
            (self.env.obstacles, OBSTACLE),
            ([self.env.goal_pos], GOAL), 
            ([self.env.agent_pos], AGENT),
        ]
        item_margin = CELL_SIZE * 0.15 
        for item_list, item_type in items_to_draw:
            for pos_x, pos_y in item_list:
                if self.env.grid[pos_y][pos_x] == 0 or item_type == AGENT or item_type == OBSTACLE : 
                    self.canvas.create_rectangle(pos_x * CELL_SIZE + item_margin, 
                                                 pos_y * CELL_SIZE + item_margin,
                                                 (pos_x + 1) * CELL_SIZE - item_margin, 
                                                 (pos_y + 1) * CELL_SIZE - item_margin,
                                                 fill=COLOR_MAP[item_type], outline=COLOR_MAP[item_type])
        
        if self.env.game_finished:
            center_x = self.env.width * CELL_SIZE / 2
            center_y = self.env.height * CELL_SIZE / 2
            rect_width = 480
            rect_height = 90
            rect_color = "#166534"  
            outline_color = "#22c55e"  
            text_color = "#e5ffe5"     

            self.canvas.create_rectangle(
                center_x - rect_width/2, center_y - rect_height/2,
                center_x + rect_width/2, center_y + rect_height/2,
                fill=rect_color, outline=outline_color, width=6
            )
            self.canvas.create_text(
                center_x, center_y,
                text="GOAL Reached!",
                font=("Impact", 36, "bold"),
                fill=text_color
            )
        

    #4.
    def update_stats_display(self):
        self.score_label.config(text=f"Score: {self.env.score}")
        self.moves_label.config(text=f"Moves: {self.env.moves_taken}")
        self.keys_label.config(text=f"Keys: {self.env.collected_keys}/{self.env.total_keys_required}")
        self.powerups_label.config(text=f"Powerups: {self.env.collected_powerups}")
        
        can_use_lifelines = not self.ai_animating and not self.env.game_finished

        self.wall_break_button.config(text=f"Wall Break ({self.env.wall_breaks})", 
                                      state=tk.NORMAL if self.env.wall_breaks > 0 and can_use_lifelines else tk.DISABLED)
        
        if self.env.frozen_obstacles:
            remaining_freeze_time = 10 - (time.time() - self.env.frozen_time)
            self.freeze_button.config(text=f"Frozen ({int(max(0,remaining_freeze_time))}s)", state=tk.DISABLED)
        else:
            self.freeze_button.config(text=f"Freeze ({self.env.freeze_available})",
                                      state=tk.NORMAL if self.env.freeze_available > 0 and can_use_lifelines else tk.DISABLED)
        
        self.reveal_path_button.config(text=f"Reveal Path ({self.env.reveal_path_available})",
                                       state=tk.NORMAL if self.env.reveal_path_available > 0 and can_use_lifelines else tk.DISABLED)
    #5.  
    def handle_keypress(self, event):
        if self.env.game_finished or self.ai_animating or self.wall_break_mode: #diable keypresses if ai is running or game is finished or wall break mode is active
            return

        dx, dy = 0, 0
        if event.keysym == "Up": dy = -1
        elif event.keysym == "Down": dy = 1
        elif event.keysym == "Left": dx = -1
        elif event.keysym == "Right": dx = 1
        else: return

        ax, ay = self.env.agent_pos
        new_pos = (ax + dx, ay + dy)

        if self.env.is_valid_move(new_pos):
            message = self.env.move_agent(new_pos)
            if message: self.status_message.set(message)
            else: self.status_message.set(f"Moved to ({new_pos[0]}, {new_pos[1]})")
            
            self.draw_maze()
            self.update_stats_display()

            if self.env.game_finished:
                self.handle_game_end()
        else:
            self.status_message.set("Invalid move or blocked.")
    #6.
    def handle_canvas_click(self, event):
        if not self.wall_break_mode or self.ai_animating or self.env.game_finished:
            return

        grid_x, grid_y = event.x // CELL_SIZE, event.y // CELL_SIZE
        
        success, message = self.env.break_wall((grid_x, grid_y))
        self.status_message.set(message)
        
        if success:
            self.draw_maze()
        
        self.wall_break_mode = False 
        self.canvas.config(cursor="")
        self.wall_break_button.config(relief=tk.RAISED)
        self.update_stats_display() 
    #7.
    def activate_wall_break_mode(self):
        if self.env.wall_breaks > 0 and not self.ai_animating and not self.env.game_finished:
            self.wall_break_mode = not self.wall_break_mode 
            if self.wall_break_mode:
                self.status_message.set("WALL BREAK: Click on a wall to break. Press key to cancel.")
                self.canvas.config(cursor="hand2")
                self.wall_break_button.config(relief=tk.SUNKEN)
            else:
                self.status_message.set("Wall break mode deactivated.")
                self.canvas.config(cursor="")
                self.wall_break_button.config(relief=tk.RAISED)
        else:
            self.status_message.set("Cannot use Wall Break now.")
    #8.       
    def use_freeze_ui(self):
        if self.ai_animating or self.env.game_finished: return
        success, message = self.env.use_freeze_obstacles()
        self.status_message.set(message)
        if success:
            self.update_stats_display()
            self.draw_maze() 
    #9.
    def use_reveal_path_ui(self):
        if self.ai_animating or self.env.game_finished: return
        selected_algo = self.algorithm_var.get()
        success, message = self.env.use_reveal_path(selected_algo)
        self.status_message.set(message)
        if success:
            self.update_stats_display()
            self.draw_maze()
    #10.
    def reset_maze_ui(self, show_prompt=True):
        if self.ai_animating:
            self.status_message.set("AI is running. Cannot reset now.")
            return

        if show_prompt:
            if not messagebox.askyesno("Reset Maze", "Start a new random maze? Your progress will be lost."):
                return
        
        self.env.generate_maze() 
        self.common_reset_ui_actions("New maze generated. Good luck!")

    #11.  
    def replay_maze(self):
        if self.ai_animating:
            self.status_message.set("AI is running. Cannot replay now.")
            return
        if self.env.initial_grid_config is None:
            messagebox.showinfo("Replay Maze", "No maze has been generated yet to replay.")
            return
        if not messagebox.askyesno("Replay Maze", "Restart the current maze? Your progress will be lost."):
            return

        self.env.reset_state_for_replay() 
        self.common_reset_ui_actions("Maze replayed. Try again!")
    
    #12.
    def common_reset_ui_actions(self, status_msg):
        self.ai_current_path_taken = []
        self.status_message.set(status_msg)
        self.draw_maze()
        self.update_stats_display() 
        self.time_label.config(text="Time: 0s")
        self.enable_controls() 

        if hasattr(self, '_periodic_update_id') and self._periodic_update_id:
            self.root.after_cancel(self._periodic_update_id)
        self._periodic_update_id = self.root.after(100, self.periodic_update)
        
        if hasattr(self, '_obstacle_move_id') and self._obstacle_move_id:
            self.root.after_cancel(self._obstacle_move_id)
        self._obstacle_move_id = self.root.after(self.obstacle_move_interval, self.periodic_obstacle_move)
        
        self.wall_break_mode = False 
        self.canvas.config(cursor="")
        self.wall_break_button.config(relief=tk.RAISED)

    #13.
    def run_ai_agent(self, algorithm_name=None):
        if self.ai_animating or self.env.game_finished:
            self.status_message.set("Cannot run AI now (already running or game over).")
            return

        if algorithm_name is None:
            algorithm_name = self.algorithm_var.get()
        
        self.status_message.set(f"AI ({algorithm_name.upper()}) is calculating path...")
        self.root.update_idletasks() 

        path = self.env.get_path_for_algorithm(algorithm_name)
        
        if path and len(path) > 1:
            self.ai_path_to_animate = path 
            self.ai_animating = True
            self.ai_animation_paused = False
            self.ai_current_path_taken = [self.env.agent_pos] 
            self.disable_controls_for_ai()
            self.status_message.set(f"AI ({algorithm_name.upper()}) running. Path moves: {len(path)-1}")
            self.animate_ai_step()
        elif path and len(path) == 1 and path[0] == self.env.agent_pos:
            self.status_message.set(f"AI ({algorithm_name.upper()}): Already at destination or no moves needed.")
        else:
            self.status_message.set(f"AI ({algorithm_name.upper()}) could not find a path from current state.")
    
    #14.
    def animate_ai_step(self):
        if not self.ai_animating: return 
        if self.ai_animation_paused:
            self.root.after(100, self.animate_ai_step) 
            return

        if not self.ai_path_to_animate: 
            self.ai_animating = False
            is_game_won = self.env.agent_pos == self.env.goal_pos and self.env.collected_keys >= self.env.total_keys_required
            self.status_message.set(f"AI finished. Moves: {self.env.moves_taken}. {'Goal Reached!' if is_game_won else ''}")
            self.enable_controls()
            if self.env.game_finished: self.handle_game_end()
            return

        if self.ai_path_to_animate[0] == self.env.agent_pos:
            self.ai_path_to_animate.pop(0) 
            if not self.ai_path_to_animate: 
                self.animate_ai_step()
                return
        
        next_pos = self.ai_path_to_animate.pop(0)
        
        if not self.env.is_valid_move(next_pos):
            self.status_message.set(f"AI stopped: Path blocked at {next_pos}. Re-planning needed (not implemented).")
            self.ai_animating = False
            self.enable_controls()
            return

        message = self.env.move_agent(next_pos) 
        self.ai_current_path_taken.append(next_pos)
        
        if message: self.status_message.set(f"AI: {message.split('!')[0]}") 
        
        self.draw_maze()
        self.update_stats_display()

        if self.env.game_finished:
            self.ai_animating = False
            self.handle_game_end()
            return

        self.root.after(180, self.animate_ai_step) #
    
    #15.
    def toggle_ai_pause(self):
        if not self.ai_animating: return
        self.ai_animation_paused = not self.ai_animation_paused
        if self.ai_animation_paused:
            self.pause_ai_button.config(text="Resume AI", bg="#4CAF50", fg="white") 
            self.status_message.set("AI Paused. Obstacles may still move.")
        else:
            self.pause_ai_button.config(text="Pause AI", bg="#FFC107", fg="black") 
            self.status_message.set("AI Resumed.")
            
    #16.
    def run_optimal_solver(self):   
        if self.ai_animating or self.env.game_finished: return
        self.status_message.set("Finding optimal algorithm for current state...")
        self.root.update_idletasks()
        optimal_algo = self.env.find_optimal_algorithm()
        if optimal_algo:
            self.status_message.set(f"Optimal from here: {optimal_algo.upper()}. Running...")
            self.algorithm_var.set(optimal_algo)
            self.run_ai_agent(optimal_algo)
        else:
            self.status_message.set("Could not determine optimal algorithm or no path found.")

    #17.
    def update_compare_results(self):
        results = []
        algorithms_to_compare = ["bfs", "a_star", "greedy", "dfs"]
        current_player_state_snapshot = self.env.snapshot_state()
        best_idx = -1
        best_moves = float('inf')
        best_score = float('-inf')
        best_time = float('inf')

        for idx, algo_name in enumerate(algorithms_to_compare):
            start_sim_time = time.perf_counter()
            moves, sim_score, sim_path = self.env.simulate_ai(algo_name)
            sim_duration = time.perf_counter() - start_sim_time
            powerups_collected = 0
            if sim_path:
                powerups_collected = len([pos for pos in sim_path if pos in self.env.initial_powerups])
                powerups_collected = min(powerups_collected, len(self.env.initial_powerups))
            results.append({
                "Algorithm": algo_name.upper(),
                "Moves": moves if moves >= 0 else "N/A",
                "Sim. Score": sim_score if moves >= 0 else "N/A",
                "Powerups": powerups_collected if moves >= 0 else "N/A",
                "Time (s)": f"{sim_duration:.3f}" if moves >= 0 else "N/A"
            })
            if moves >= 0:
                if (moves < best_moves or
                    (moves == best_moves and sim_score > best_score) or
                    (moves == best_moves and sim_score == best_score and sim_duration < best_time)):
                    best_moves = moves
                    best_score = sim_score
                    best_time = sim_duration
                    best_idx = idx
        self.env.restore_state(current_player_state_snapshot)

        # Update treeview
        for i in self.compare_tree.get_children():
            self.compare_tree.delete(i)
        for idx, res_item in enumerate(results):
            tags = ()
            if best_idx is not None and idx == best_idx:
                tags = ("highlight",)
            self.compare_tree.insert("", "end", values=tuple(res_item[c] for c in ["Algorithm", "Moves", "Sim. Score", "Powerups", "Time (s)"]), tags=tags)

        self.compare_update_id = self.root.after(500, self.update_compare_results)
    #18.
    def close_compare_window(self):
        if self.compare_update_id:
            self.root.after_cancel(self.compare_update_id)
            self.compare_update_id = None
        if self.compare_window:
            self.compare_window.destroy()
        self.compare_window = None
    
    #19.
    def display_comparison_results(self, results, highlight_idx=None):
        top = tk.Toplevel(self.root)
        top.title("Algorithm Comparison Results (Current State)")
        top.geometry("520x250")
        top.configure(bg="#E0E0E0")

        tk.Label(top, text="Comparison based on solving from the agent's current position and state.",
                font=("Arial", 9), bg="#E0E0E0", fg="#333333").pack(pady=(5,0))

        cols = ["Algorithm", "Moves", "Sim. Score", "Powerups", "Time (s)"]
        tree_frame = tk.Frame(top)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        style = ttk.Style(top)
        style.theme_use('clam')
        style.configure("Treeview.Heading", font=("Arial", 10, "bold"), background="#3da9fc", foreground="white")
        style.configure("Treeview", rowheight=25, font=("Arial", 9), background="#f7f7ff", fieldbackground="#f7f7ff")
        style.map("Highlight.Treeview", background=[('selected', '#f6c90e')])

        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="none")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center', stretch=tk.YES)

        for idx, res_item in enumerate(results):
            tags = ()
            if highlight_idx is not None and idx == highlight_idx:
                tags = ("highlight",)
            tree.insert("", "end", values=tuple(res_item[c] for c in cols), tags=tags)

        tree.tag_configure("highlight", background="#B2FFB2", font=("Arial", 9, "bold"))

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Button(top, text="Close", command=top.destroy, bg="#D3D3D3", font=("Arial",9,"bold"), relief=tk.RAISED, borderwidth=2).pack(pady=10)
        top.transient(self.root)
        top.grab_set()

    #20.
    def periodic_obstacle_move(self):
        can_obstacles_move = not self.env.game_finished and \
                             (not self.ai_animating or self.ai_animation_paused)

        if can_obstacles_move:
            collision = self.env.move_obstacles()
            if collision: 
                self.status_message.set(f"Ouch! Obstacle collision! Score -100. Score: {self.env.score}")
            self.draw_maze()
            self.update_stats_display() 
        
        
        if not self.env.game_finished : 
             self._obstacle_move_id = self.root.after(self.obstacle_move_interval, self.periodic_obstacle_move)

    #21.
    def handle_game_end(self, custom_message=None):
        self.ai_animating = False 
        self.ai_animation_paused = False 
        self.ai_path_to_animate = [] 
        
        self.enable_controls() 
        
        final_message = f"Game Over! Final Score: {self.env.score}, Moves: {self.env.moves_taken}."
        if self.env.start_time:
            final_time = int(time.time() - self.env.start_time)
            final_message += f" Time: {final_time}s."

        if custom_message: final_message = custom_message + "\n" + final_message
        
        self.status_message.set(final_message.split('\n')[0]) 
        self.draw_maze() 
        self.update_stats_display() 

        if hasattr(self, '_periodic_update_id') and self._periodic_update_id:
            self.root.after_cancel(self._periodic_update_id)
            self._periodic_update_id = None 
        if hasattr(self, '_obstacle_move_id') and self._obstacle_move_id:
            self.root.after_cancel(self._obstacle_move_id)
            self._obstacle_move_id = None 

        action = messagebox.askquestion("Game Over", f"{final_message}\n\nReplay this maze?", 
                                        icon='info', type=messagebox.YESNOCANCEL, parent=self.root)
        if action == 'yes':
            self.replay_maze()
        elif action == 'no':
            self.reset_maze_ui(show_prompt=False) 
        else: 
            self.status_message.set("Game ended. Choose Replay or New Maze to continue.")
    #22.
    def disable_controls_for_ai(self):
        self.run_ai_button.config(state=tk.DISABLED)
        self.optimal_button.config(state=tk.DISABLED)
        self.reset_button.config(state=tk.DISABLED)
        self.replay_button.config(state=tk.DISABLED)
        
        # Disable lifelines as AI path is pre-computed
        self.wall_break_button.config(state=tk.DISABLED)
        self.freeze_button.config(state=tk.DISABLED)
        self.reveal_path_button.config(state=tk.DISABLED)
        
        # If wall break mode was active, deactivate it
        if self.wall_break_mode:
            self.wall_break_mode = False
            self.canvas.config(cursor="")
            self.wall_break_button.config(relief=tk.RAISED) 

        self.algorithm_menu.config(state=tk.DISABLED)
        self.pause_ai_button.config(state=tk.NORMAL, text="Pause AI", bg="#FFC107", fg="black")
    #23.    
    def enable_controls(self):
        self.run_ai_button.config(state=tk.NORMAL)
        self.optimal_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)
        self.replay_button.config(state=tk.NORMAL)
        self.algorithm_menu.config(state="readonly")
        self.pause_ai_button.config(state=tk.DISABLED, text="Pause AI", bg="#FFC107", fg="black")
        
        self.update_stats_display() 

#--Main Function--
if __name__ == "__main__":
    main_root = tk.Tk()
    try:
        s = ttk.Style()
        s.theme_use('clam') 
    except tk.TclError:
        print("Ttk themes not available or 'clam' theme failed. Using default.")
        
    app = MazeApp(main_root)
    main_root.mainloop()
