""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np

class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0.0, 0.0, 0.0])

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        TD4
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        #--PARAMETRES --
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        ref_x, ref_y, ref_theta = odom_pose_ref
        ox, oy, ot = odom_pose

        # Convert from odom frame to map frame using the reference pose as the transformation
        cos_t = np.cos(ref_theta)
        sin_t = np.sin(ref_theta)

        abs_x = ref_x + cos_t * ox - sin_t * oy
        abs_y = ref_y + sin_t * ox + cos_t * oy
        abs_theta = ref_theta + ot

        # Normalize theta to [-pi, pi]
        abs_theta = (abs_theta + np.pi) % (2 * np.pi) - np.pi

        return np.array([abs_x, abs_y, abs_theta])
    
    def _score(self, lidar, pose):
        """
        TD4
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        #--PARAMETRES --
        distances = lidar.get_sensor_values()
        angles    = lidar.get_ray_angles()

        # Valid lidar points: finite and within sensor range
        valid = (distances > 0.1) & (distances < 490.0)
        laser_dist_valid = distances[valid][::5]
        angles_valid = angles[valid][::5]

        x, y, theta = pose
        angles_world = theta + angles_valid
        
        x_abs = x + laser_dist_valid * np.cos(angles_world)
        y_abs = y + laser_dist_valid * np.sin(angles_world)
        
        mx, my = self.grid.conv_world_to_map(x_abs, y_abs)
        
        valid_bounds = (mx >= 0) & (mx < self.grid.x_max_map) & (my >= 0) & (my < self.grid.y_max_map)
        mx = mx[valid_bounds].astype(int)
        my = my[valid_bounds].astype(int)
        
        # Penalize poses that have too few valid lidar points in the map to avoid instability
        if len(mx) <15:
            return -500.0
        
        # Score based on occupancy values at lidar endpoints: prefer occupied cells (positive values) and penalize free cells (negative values)
        occ_score = float(np.sum(self.grid.occupancy_map[mx, my]))
        
        # Penalize poses that predict many lidar endpoints in unknown space (values near 0) to encourage poses that explain the observations with known occupied cells
        empty_score = float(np.sum(self.grid.occupancy_map[mx, my] < -0.3))
        
        # Final score combines occupancy and empty cell penalties, with a weight to balance them. The exact values can be tuned for better performance.
        return occ_score - 0.1 * empty_score

    def localise(self, lidar, raw_odom):
        """
        TD4
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # --PARAMETRES --
        N_SAMPLES = 40 # number of samples per iteration
        N_ITER = 6 # number of iterations of CEM
        sigma = np.array([2.0, 2.0, 0.05]) 
        
        # IMPROVEMENT: CEM with elite update - iteratively sample around the best pose and update the mean
        mean = self.odom_pose_ref.copy()
        best_score = -np.inf
        best_ref = mean.copy()
        
        for iteration in range(N_ITER):
            samples = np.random.normal(mean, sigma, size=(N_SAMPLES, 3))
            scores = [] # store scores for all samples in this iteration
            
            for sample_ref in samples:
                cand_pose = self.get_corrected_pose(raw_odom, sample_ref)
                s = self._score(lidar, cand_pose)
                scores.append(s)
                
                if s > best_score: # keep track of the best pose across all iterations
                    best_score = s
                    best_ref = sample_ref.copy()
            
            # CEM elite update
            scores_arr = np.array(scores)
            elite_idx = np.argsort(scores_arr)[-10:]
            elite_samples = samples[elite_idx]
            mean = np.mean(elite_samples, axis=0)# update mean to elite mean for next iteration
        
        # IMPROVEMENT: Reject pose jumps that are too large compared to the current reference to avoid instability
        if best_score < -80.0:
            #print("CEM failed - keeping previous pose")
            return best_score
        
        pos_jump = np.linalg.norm(best_ref[:2] - self.odom_pose_ref[:2])
        
        #Reject jumps  to prevent instability, but still update the reference if the score is good
        if pos_jump < 2.0:
            self.odom_pose_ref = best_ref
        else:
            #print(f"Jump too large: {pos_jump:.1f}cm rejected")
            pass
        return best_score

    def update_map(self, lidar, pose):
        """
        #TD3
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
         
        # --PARAMETRES--
        MAX_RANGE = 400.0 # maximum lidar range 
        MARGIN = 5 # margin to consider a cell free
        STEP = 5 # subsampling step for lidar points to speed up updates

        #  -- CALCULATION OF THE MAP UPDATE --
        laser_dist = lidar.get_sensor_values()
        ray_angles = lidar.get_ray_angles()
        x, y, theta = pose
        
        valid = np.isfinite(laser_dist) & (laser_dist > 5) & (laser_dist < MAX_RANGE) # valid range and finite
        laser_dist_sub = laser_dist[valid][::STEP] #diving the lidar data into subsampled arrays for efficiency
        ray_angles_sub = ray_angles[valid][::STEP]
        ray_angles_world_sub = ray_angles_sub + theta

        # IMPROVEMENT: Use distance based probabilities 
        p_empty_array = -0.5 * (1 - laser_dist_sub / MAX_RANGE)
        p_occ_array = 0.5 * (1 - laser_dist_sub / MAX_RANGE)

        #convert lidar points to world coordinates
        x_abs = x + laser_dist_sub * np.cos(ray_angles_world_sub) 
        y_abs = y + laser_dist_sub * np.sin(ray_angles_world_sub)

        laser_dist_empty = np.maximum(0, laser_dist_sub - MARGIN) # reduce distance for empty cells to create a margin of free space
        x_empty = x + laser_dist_empty * np.cos(ray_angles_world_sub)
        y_empty = y + laser_dist_empty * np.sin(ray_angles_world_sub)

        #-- UPDATE THE MAP ---
        # Update map along the ray for empty cells, then update occupied cells at the end
        for pt_x, pt_y, p_empty in zip(x_empty, y_empty, p_empty_array):
            self.grid.add_value_along_line(x, y, pt_x, pt_y, p_empty)
    
        if len(x_abs) > 0:
            mx, my = self.grid.conv_world_to_map(x_abs, y_abs)
            valid_bounds = (mx >= 0) & (mx < self.grid.x_max_map) & (my >= 0) & (my < self.grid.y_max_map) # ensure points are within map bounds
            
            x_abs_v = x_abs[valid_bounds]
            y_abs_v = y_abs[valid_bounds]
            p_occ_v = p_occ_array[valid_bounds]
            mx_v = mx[valid_bounds]
            my_v = my[valid_bounds]

            #IMPROVEMENT: Avoid duplicate updates for points that fall in the same cell - group by cell and take max probability
            if len(x_abs_v) > 0:
                cells = np.column_stack((mx_v.astype(int), my_v.astype(int)))
                _, idx = np.unique(cells, axis=0, return_index=True)      
                self.grid.add_map_points(x_abs_v[idx], y_abs_v[idx], p_occ_v[idx])

        np.clip(self.grid.occupancy_map, -5.0, 5.0, out=self.grid.occupancy_map) # clamp values to prevent numerical issues
    
    def compute(self):
            """ Useless function, just for the exercise on using the profiler """
            # Remove after TP1 -- NEED TO KEEP THIS FUNCTION FOR INVIDUAL EXERCISE

            ranges = np.random.rand(3600)
            ray_angles = np.arange(-np.pi, np.pi, np.pi / 1800)

            # Poor implementation of polar to cartesian conversion
            points = []
            for i in range(3600):
                pt_x = ranges[i] * np.cos(ray_angles[i])
                pt_y = ranges[i] * np.sin(ray_angles[i])
                points.append([pt_x, pt_y])
