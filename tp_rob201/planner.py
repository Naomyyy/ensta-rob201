"""
Planner class
Implementation of A*
"""

import numpy as np
import heapq
import cv2

from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def get_neighbors(self, current_cell, goal_cell=None):
        """Get valid neighbors. If goal_cell provided, allow exploring to reach it."""
        x, y = current_cell
        neighbors = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue 
                nx, ny = x + dx, y + dy
                
                # Check grid boundaries
                if not (0 <= nx < self.grid.x_max_map and 0 <= ny < self.grid.y_max_map):
                    continue
                
                # Check inflated obstacles
                inflated = getattr(self, 'inflated_map', None)
                if inflated is not None:
                    is_obstacle = inflated[nx, ny] > 0
                else:
                    is_obstacle = self.grid.occupancy_map[nx, ny] >= 0.2
                
                # Allow reaching unexplored areas if it's the goal
                is_unexplored = np.abs(self.grid.occupancy_map[nx, ny]) < 0.1
                
                # For returning home: allow traversal of unexplored areas
                # (robô pode ter explorado mas o mapa estar incompleto)
                if is_unexplored and (goal_cell is None or (nx, ny) != goal_cell):
                    # Only block if very close to known obstacles
                    dist_to_obstacle = np.abs(self.grid.occupancy_map[nx, ny])
                    if dist_to_obstacle < 0.05:  # Very close edge
                        is_obstacle = True
                    # Otherwise allow exploration in unknowns
                
                if not is_obstacle:
                    neighbors.append((nx, ny))
        
        return neighbors

    def heuristic(self, cell_1, cell_2): #euclidean distance heuristic function for A*
        """ Euclidean distance between two map cells """
        return np.sqrt((cell_1[0] - cell_2[0])**2 + (cell_1[1] - cell_2[1])**2)

    def _reconstruct_path(self, cameFrom, current): # reconstructs the path from the start to the goal by backtracking through the cameFrom dictionary
        total_path = [current] # list to store the path, initialized with the goal cell
        while current in cameFrom:
            current = cameFrom[current] # update current to the cell it came from, effectively moving backwards through the path
            total_path.append(current)
        total_path.reverse() # reverse the path to get it from start to goal instead of goal to start
        return total_path

    
    def plan(self, start, goal, mu=1.0):
        """
        TD5
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates
        goal : [x, y, theta] nparray, goal pose in world coordinates
        mu : Weight for the heuristic (A* pondéré). Default is 1.0.
        """
        #-- PARAMETRES ---
        obstacles = (self.grid.occupancy_map >= 0.2).astype(np.uint8) # binary map of obstacles based on occupancy grid values
        kernel = np.ones((11, 11), np.uint8) # kernel for dilation, determines how much to inflate obstacles (11x11 means inflating by 5 cells in all directions)
        self.inflated_map = cv2.dilate(obstacles, kernel)

        start_map_x, start_map_y = self.grid.conv_world_to_map(start[0], start[1])
        goal_map_x, goal_map_y = self.grid.conv_world_to_map(goal[0], goal[1])

        start_node = (int(start_map_x), int(start_map_y))
        goal_node = (int(goal_map_x), int(goal_map_y))

        # BOUNDS CHECK
        if not (0 <= start_node[0] < self.grid.x_max_map and 0 <= start_node[1] < self.grid.y_max_map):
            print(f"Start out of bounds: {start_node}")
            return np.array([[], [], []])
        if not (0 <= goal_node[0] < self.grid.x_max_map and 0 <= goal_node[1] < self.grid.y_max_map):
            print(f"Goal out of bounds: {goal_node}")
            return np.array([[], [], []])

        # Inside Function to find the nearest free cell if the start or goal is in an obstacle
        def find_nearest_free(node):
            if self.inflated_map[node[0], node[1]] == 0:
                return node
            for r in range(1, 40):  # Increased search radius
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        nx, ny = node[0] + dx, node[1] + dy
                        if 0 <= nx < self.grid.x_max_map and 0 <= ny < self.grid.y_max_map:
                            if self.inflated_map[nx, ny] == 0:
                                return (nx, ny)
            print(f"Warning: No free cell found near {node}, using original anyway")
            return node

        # If start or goal is in an obstacle, find nearest free cell
        start_node = find_nearest_free(start_node)
        goal_node = find_nearest_free(goal_node)

        # Trivial case: start == goal
        if start_node == goal_node:
            wx, wy = self.grid.conv_map_to_world(start_node[0], start_node[1])
            return np.array([[wx], [wy], [0.0]])

        openSet = []
        openSet_dict = {}  # Map node -> fScore
        closed_set = set()  # Already processed nodes
        
        start_h = self.heuristic(start_node, goal_node)
        heapq.heappush(openSet, (mu * start_h, start_node))
        openSet_dict[start_node] = mu * start_h

        cameFrom = {}
        gScore = {start_node: 0}

        nodes_expanded = 0
        MAX_NODES = 100000

        while openSet:
            nodes_expanded += 1
            if nodes_expanded > MAX_NODES:
                print(f"A* limit reached ({nodes_expanded} nodes). Goal unreachable.")
                return np.array([[], [], []])

            current_f, current = heapq.heappop(openSet)
            
            # Skip if already processed (handle duplicates in heap)
            if current in closed_set:
                continue
            
            # Remove from dict
            if current in openSet_dict:
                del openSet_dict[current]
            closed_set.add(current)

            if current == goal_node:
                # Reconstruct path
                path_cells = self._reconstruct_path(cameFrom, current)
                
                # Convert back to world coordinates
                path_world = []
                for cell in path_cells:
                    wx, wy = self.grid.conv_map_to_world(cell[0], cell[1])
                    path_world.append([wx, wy, 0.0])
                
                print(f"✓ A* found path: {len(path_world)} waypoints, {nodes_expanded} nodes expanded")
                return np.array(path_world).T

            for neighbor in self.get_neighbors(current, goal_node):
                if neighbor in closed_set:
                    continue
                
                # Calculate actual movement cost (1 for adjacent, sqrt(2) for diagonal)
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                movement_cost = 1.0 if (dx + dy == 1) else np.sqrt(2)
                    
                tentative_gScore = gScore[current] + movement_cost

                if tentative_gScore < gScore.get(neighbor, float('inf')):
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    fScore = tentative_gScore + mu * self.heuristic(neighbor, goal_node)
                    
                    # Only add if not in open set or found better path
                    if neighbor not in openSet_dict or fScore < openSet_dict[neighbor]:
                        openSet_dict[neighbor] = fScore
                        heapq.heappush(openSet, (fScore, neighbor))

        print(f"✗ A* failed: No path found after {nodes_expanded} nodes")
        return np.array([[], [], []])

    def explore_frontiers(self, current_pose):
        # TD6
        
        grid = self.grid.occupancy_map
        
        unexplored = (np.abs(grid) < 0.1).astype(np.uint8)
        free = (grid < -0.5).astype(np.uint8)
        
        kernel = np.ones((3, 3), np.uint8)
        free_dilated = cv2.dilate(free, kernel)
        frontiers = cv2.bitwise_and(unexplored, free_dilated)
        
        # Inflate obstacles to create a safety buffer around them   
        obstacles = (grid >= 0.2).astype(np.uint8)
        obs_kernel = np.ones((11, 11), np.uint8)
        obstacles_dilated = cv2.dilate(obstacles, obs_kernel)
        
        safe_frontiers = cv2.bitwise_and(frontiers, cv2.bitwise_not(obstacles_dilated))
        
        
        # Remove frontiers too close to borders to avoid unreachable goals
        border_mask = np.ones_like(safe_frontiers)
        border_mask[:40, :] = 0  # Top border
        border_mask[-40:, :] = 0  # Bottom border
        border_mask[:, :40] = 0  # Left border
        border_mask[:, -40:] = 0  # Right border
        
        safe_frontiers = cv2.bitwise_and(safe_frontiers, border_mask.astype(np.uint8))
        
        fx, fy = np.where(safe_frontiers > 0)
        
        if len(fx) == 0:
            return None
            
        start_map_x, start_map_y = self.grid.conv_world_to_map(current_pose[0], current_pose[1])
        rx, ry = int(start_map_x), int(start_map_y)
        
        # Find nearest frontier to current position
        best_f = None
        min_dist = float('inf')
        
        # Iterate through all safe frontiers to find the nearest one    
        for x, y in zip(fx, fy):
            dist = (x - rx)**2 + (y - ry)**2
            if dist < min_dist:
                min_dist = dist
                best_f = (x, y)
                
        if best_f is None:
            return None
            
        goal_world_x, goal_world_y = self.grid.conv_map_to_world(best_f[0], best_f[1])
        return np.array([goal_world_x, goal_world_y, 0.0])
