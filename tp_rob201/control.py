""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    TD 1: Obstacle Avoidance
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # -- PARAMETRES ---
    MIN_DIST = 30 # minimum distance to consider an obstacle ahead (in cm)
    FRONT_ANGLE = 0.5 # angle in radians for front lidar cone (±30°)
   
    # -- CALCULATION OF THE COMMANDS ---
    laser_dist = lidar.get_sensor_values() # returns the distances of the lidar
    ray_angles = lidar.get_ray_angles() # returns the angles of the lidar

    front_mask = np.abs(ray_angles) < FRONT_ANGLE # creates a mask for the front lidar based in the angle
    front_dist = laser_dist[front_mask] # creates a mask for the front lidar based in the distance
    obstacle_ahead = len(front_dist) > 0 and np.any(front_dist < MIN_DIST)

    #-- IMPROVEMENT: REACTIVE CONTROL LOGIC ---
    if obstacle_ahead:
        left_mask  = (ray_angles > 0) #separates the left and right lidar rays based on their angles
        right_mask = (ray_angles < 0) 

        left_space  = np.mean(laser_dist[left_mask])  if left_mask.any()  else 0.0
        right_space = np.mean(laser_dist[right_mask]) if right_mask.any() else 0.0

        if left_space >= right_space:
            #print("Obstacle ahead, turning left")
            rotation_speed = 0.5    
        else:
            #print("Obstacle ahead, turning right")
            rotation_speed = -0.5   

        speed = 0.0

    else:
        #print("Free path ahead")
        speed = 0.5
        rotation_speed = 0.0

    return {"forward": speed, "rotation": rotation_speed}

def potential_field_control(lidar, current_pose, goal_pose):
    """
    TD2
    Control using potential field for goal reaching and obstacle avoidance
    """
    #-- PARAMETERS ---
    K_GOAL = 2.0 # attractive gain
    K_OBS  = 1000 # repulsive gain
    D_SWITCH = 200 # switch linear to quadratic 
    D_SAFE = 30 # obstacle influence radius
    MAX_FORWARD = 0.5
    MAX_ROTATION = 0.5
    BASE_SPEED = 0.5

    x, y, theta = current_pose


    #-- COMPUTATION OF THE ATTRACTIVE FIELD ---

    q = np.array(current_pose, dtype=float)
    q_goal = np.array(goal_pose, dtype=float)
 
    delta_goal = q_goal[:2] - q[:2]
    d_goal = np.linalg.norm(delta_goal)
    
    if d_goal > D_SWITCH: # far from goal: use linear attractive field
        grad_att = (K_GOAL / d_goal) * delta_goal
    else:
        K_GOAL_Q = K_GOAL / (D_SWITCH ** 2) 
        grad_att = K_GOAL_Q * delta_goal
 
    # --- COMPUTATION OF THE REPULSIVE GRADIENT ---
    grad_rep = np.zeros(2)
 
    distances = lidar.get_sensor_values()   
    angles    = lidar.get_ray_angles()    
 
    valid = np.isfinite(distances) & (distances > 1e-3) #verify that the lidar readings are valid (not infinite or zero)

    if valid.any():

        d_valid = distances[valid]
        a_valid = angles[valid]
        grad_rep = np.zeros(2)

        #IMPROVEMENT: CLUSTERING OF LIDAR POINTS TO AVOID NOISE AND GET BETTER GRADIENT ESTIMATES
        clusters = []
        current_cluster = []

        THRESHOLD = 25 # distance threshold for clustering in cm

        for i in range(len(d_valid)):

            point = (d_valid[i], a_valid[i])

            if len(current_cluster) == 0:
                current_cluster.append(point)

            else:
                prev_d = current_cluster[-1][0]

                if abs(d_valid[i] - prev_d) < THRESHOLD:
                    current_cluster.append(point)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [point]

        if len(current_cluster) > 0:
            clusters.append(current_cluster)
    else:
        clusters = []

    theta = q[2]

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    for cluster in clusters:

        # representative point = closest point of cluster
        d_obs, phi_obs = min(cluster, key=lambda p: p[0])

        if d_obs >= D_SAFE:
            continue

        obs_robot = np.array([
            np.cos(phi_obs),
            np.sin(phi_obs)
        ])

        obs_world_dir = R @ obs_robot

        coeff = (
            (K_OBS / d_obs**3)
            * (1.0 / d_obs - 1.0 / D_SAFE)
        )

        grad_rep += -coeff * obs_world_dir

    # --- TOTAL GRADIENT  ---
    # Total potential field gradient
    grad_total = grad_att + grad_rep

    #--- CONTROL LOGIC ---

    # -- Normal navigation: follow the total gradient direction ---
    desired_angle = np.arctan2(grad_total[1], grad_total[0])
    angle_error   = np.arctan2(
        np.sin(desired_angle - theta),
        np.cos(desired_angle - theta)
    )

    # IMPROVEMENT: Rotation proportional to angle error (smooth, low gain)
    rotation = 0.1 * angle_error
    rotation = np.clip(rotation, -MAX_ROTATION, MAX_ROTATION)

    # Speed: scale by alignment, stop if facing away
    if abs(angle_error) > np.pi / 2:
        forward = 0.0
    else:
        forward = BASE_SPEED * np.cos(angle_error)

    forward = np.clip(forward, 0.0, MAX_FORWARD)

    # ---IMPROVEMENT: Reactive wall avoidance blending ---
    # When an obstacle is close ahead, use reactive avoidance and hold it
    # for REACT_HOLD iterations so the robot has time to actually turn away.
    REACT_DIST  = 18 # start blending at this distance
    REACT_ANGLE = 0.4  # frontal cone
    REACT_HOLD=30  # number of iterations to stay in reactive mode

    # Persistent timer stored as a function attribute
    if not hasattr(potential_field_control, '_react_timer'):
        potential_field_control._react_timer = 0

    # If timer is still counting down, keep using reactive
    if potential_field_control._react_timer > 0:
        potential_field_control._react_timer -= 1
        #print(f"reactive avoidance (timer={potential_field_control._react_timer})")
        reactive_cmd = reactive_obst_avoid(lidar)
        forward  = reactive_cmd["forward"]
        rotation = reactive_cmd["rotation"]
    else:
        # Check if we need to trigger reactive mode
        front_mask = np.abs(angles) < REACT_ANGLE
        if front_mask.any():
            front_dists = distances[front_mask]
            min_front   = np.min(front_dists)

            if min_front < REACT_DIST or np.linalg.norm(grad_total) < 0.001:
                potential_field_control._react_timer = REACT_HOLD
                #print(f"reactive avoidance TRIGGERED (timer={REACT_HOLD})")
                reactive_cmd = reactive_obst_avoid(lidar)
                forward  = reactive_cmd["forward"]
                rotation = reactive_cmd["rotation"]

    return {"forward": forward, "rotation": rotation}


def trajectory_following_control(lidar, current_pose, goal_pose):
    """
    Simple trajectory following controller (for A* paths)
    Focuses on following waypoints smoothly
    """
    #-- PARAMETERS ---
    MAX_FORWARD = 0.5
    MAX_ROTATION = 0.5
    BASE_SPEED = 0.5

    x, y, theta = current_pose
    q = np.array(current_pose, dtype=float)
    q_goal = np.array(goal_pose, dtype=float)

    #-- COMPUTE DIRECTION TO GOAL ---
    delta_goal = q_goal[:2] - q[:2]
    d_goal = np.linalg.norm(delta_goal)
    
    if d_goal < 5:  # Very close to goal
        return {"forward": 0.0, "rotation": 0.0}
    
    desired_angle = np.arctan2(delta_goal[1], delta_goal[0])
    angle_error = np.arctan2(
        np.sin(desired_angle - theta),
        np.cos(desired_angle - theta)
    )

    # Very smooth rotation control (low gain to avoid overshooting)
    rotation = 0.2 * angle_error
    rotation = np.clip(rotation, -MAX_ROTATION, MAX_ROTATION)

    # Forward speed: maintain speed even if slightly misaligned
    if abs(angle_error) > np.pi / 2:  # 90 degrees = opposite direction
        forward = 0.0
    else:
        forward = BASE_SPEED * max(0.3, np.cos(angle_error))  # Minimum 30% speed
    
    forward = np.clip(forward, 0.0, MAX_FORWARD)

    # Simple obstacle check: if obstacle ahead, stop
    distances = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()
    
    front_mask = np.abs(angles) < 0.5
    if front_mask.any():
        min_front = np.min(distances[front_mask])
        if min_front < 35:  # Obstacle closer than 35cm
            forward = 0.0
            rotation = 0.5 if angle_error > 0 else -0.5  # Turn away

    return {"forward": forward, "rotation": rotation}
