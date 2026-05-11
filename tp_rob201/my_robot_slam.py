"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid, trajectory_following_control
from occupancy_grid import OccupancyGrid
from planner import Planner


# --Shared waypoints  ---
DEFAULT_WAYPOINTS = [
    np.array([0,    140.0, 0.0]),
    np.array([-230,  30.0, 0.0]),
    np.array([-400, -500.0, 0.0]),
    np.array([0,   -500.0, 0.0]),
    np.array([0,      0.0, 0.0]),
]

# Waypoints used by TP5 (exploration subset — no return point, handled by A*)
TP5_WAYPOINTS = [
    np.array([0,   100.0, 0.0]),
    np.array([-300,  30.0, 0.0]),]


DIST_STOP = 30 # distance threshold to consider a waypoint reached 


class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        super().__init__(lidar_params=lidar_params,
                         odometer_params=odometer_params)

        self.counter = 0

        size_area      = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(
            x_min=-(size_area[0] / 2 + robot_position[0]),
            x_max=size_area[0] / 2 - robot_position[0],
            y_min=-(size_area[1] / 2 + robot_position[1]),
            y_max=size_area[1] / 2 - robot_position[1],
            resolution=2,
        )

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner   = Planner(self.occupancy_grid)

        self.corrected_pose = np.array([0, 0, 0])


    def _init_waypoints(self, waypoints):
        """Initialise waypoint state if not already done."""
        if not hasattr(self, "_waypoints"):
            self._waypoints = waypoints
            self._goal_idx  = 0

    def _navigate_waypoints(self, pose, waypoints=None):
        """
        Generic waypoint follower using potential_field_control.
        """
        if waypoints is None:
            waypoints = DEFAULT_WAYPOINTS

        self._init_waypoints(waypoints)

        if self._goal_idx >= len(self._waypoints):
            return None                         

        goal    = self._waypoints[self._goal_idx]
        command = potential_field_control(self.lidar(), pose, goal)

        if np.linalg.norm(pose[:2] - goal[:2]) < DIST_STOP:
            print(f"Goal {self._goal_idx + 1} reached.")
            self._goal_idx += 1
            if self._goal_idx >= len(self._waypoints):
                print("All goals reached, stopping the robot.")
                return None

        return command

    def control_follow_traj(self):
        """Local controller to follow an A* trajectory."""
        pose = self.corrected_pose

        # Drop waypoints we have already passed
        while self.traj.shape[1] > 0 and np.linalg.norm(pose[:2] - self.traj[:2, 0]) < 15:
            self.traj = self.traj[:, 1:]

        if self.traj.shape[1] == 0:
            return {"forward": 0.0, "rotation": 0.0}

        # Check if we reached the goal (last waypoint in trajectory)
        dist_to_last = np.linalg.norm(pose[:2] - self.traj[:2, -1])
        if dist_to_last < DIST_STOP:
          
            self.traj = np.array([[], [], []])
            return {"forward": 0.0, "rotation": 0.0}

        # Always target next waypoint (no lookahead) for accurate path following
        return trajectory_following_control(self.lidar(), pose, self.traj[:, 0])

    def control(self):
        """Main control function executed at each time step."""
        self.counter += 1

        # return self.control_tp1()
        # return self.control_tp2()
        # return self.control_tp3()
        # return self.control_tp4()
        # return self.control_tp5()
        return self.control_tp6()

    def control_tp1(self):
        self.tiny_slam.compute()
        return reactive_obst_avoid(self.lidar())

    def control_tp2(self):
        pose = self.odometer_values()
        cmd  = self._navigate_waypoints(pose)
        return cmd if cmd is not None else {"forward": 0.0, "rotation": 0.0}

    def control_tp3(self):
        pose = self.odometer_values()
        self.tiny_slam.update_map(self.lidar(), pose)

        if self.counter % 10 == 0: # only display every 10 iterations for efficiency
            self.tiny_slam.grid.display_cv(pose)

        cmd = self._navigate_waypoints(pose)
        return cmd if cmd is not None else {"forward": 0.0, "rotation": 0.0}

    def control_tp4(self):
        raw_odom = self.odometer_values()
        self.tiny_slam.localise(self.lidar(), raw_odom)
        self.corrected_pose = self.tiny_slam.get_corrected_pose(raw_odom) # get corrected pose from SLAM instead of raw odometry
        self.tiny_slam.update_map(self.lidar(), self.corrected_pose)

        if self.counter % 10 == 0: # only display every 10 iterations for efficiency
            self.tiny_slam.grid.display_cv(self.corrected_pose)

        cmd = self._navigate_waypoints(self.corrected_pose) 
        return cmd if cmd is not None else {"forward": 0.0, "rotation": 0.0}

    def control_tp5(self):
        raw_odom = self.odometer_values()
        self.tiny_slam.localise(self.lidar(), raw_odom)
        self.corrected_pose = self.tiny_slam.get_corrected_pose(raw_odom)
        self.tiny_slam.update_map(self.lidar(), self.corrected_pose)

   
        if not hasattr(self, "returning_home"):
            self.returning_home = False
            self.traj = np.array([[], [], []])

        if self.counter % 10 == 0:
            self.tiny_slam.grid.display_cv(self.corrected_pose, traj=self.traj)

        if not self.returning_home:
            cmd = self._navigate_waypoints(self.corrected_pose, TP5_WAYPOINTS)

            if cmd is None:
                # All exploration waypoints done → plan A* return
                print("Exploration complete! Planning A* return to origin…")
                self.returning_home = True
                self.traj = self.planner.plan(
                    self.corrected_pose, np.array([0.0, 0.0, 0.0]), mu=1.0
                )
                return {"forward": 0.0, "rotation": 0.0}

            return cmd

        # Returning home along A* trajectory
        if self.traj.shape[1] > 0:
            return self.control_follow_traj()
        return {"forward": 0.0, "rotation": 0.0}

    def control_tp6(self):
        """TP6: Frontier-Based Exploration — autonomous mapping then return home."""
        raw_odom = self.odometer_values()
        self.tiny_slam.localise(self.lidar(), raw_odom)
        self.corrected_pose = self.tiny_slam.get_corrected_pose(raw_odom)
        self.tiny_slam.update_map(self.lidar(), self.corrected_pose)

        if not hasattr(self, "state"):
            self.state          = "EXPLORING"
            self.traj           = np.array([[], [], []])
            self.frontier_timer = 40  # Initial wait for map to build
            self.exploration_attempts = 0
            self.visited_frontiers = []
            self.no_frontier_count = 0  # Counter for consecutive "no frontier" detections

        if self.counter % 10 == 0:
            goal_disp = None if self.traj.shape[1] == 0 else self.traj[:, -1]
            if self.state == "RETURNING":
                goal_disp = np.array([0.0, 0.0, 0.0])
            self.tiny_slam.grid.display_cv(self.corrected_pose, goal=goal_disp, traj=self.traj)

        if self.state == "EXPLORING":
            # When trajectory is done, try to find next frontier
            if self.traj.shape[1] == 0:
                self.frontier_timer -= 1
                
                if self.frontier_timer <= 0:
                    # Try up to 5 times to find a frontier that hasn't been visited
                    goal = None
                    for attempt in range(5):
                        candidate = self.planner.explore_frontiers(self.corrected_pose)
                        if candidate is None:
                            self.no_frontier_count += 1
                            break
                        
                        # Check if this frontier was recently visited
                        is_revisit = False
                        for visited in self.visited_frontiers[-15:]:  # Check last 15 frontiers
                            dist_to_visited = np.linalg.norm(candidate[:2] - visited[:2])
                            if dist_to_visited < 60:  # Less than 60cm away
                                is_revisit = True
                                break
                        
                        if not is_revisit:
                            goal = candidate
                            self.no_frontier_count = 0
                            break
                    
                    if goal is not None:
                        self.visited_frontiers.append(goal.copy())
                        self.exploration_attempts += 1
                        
                        self.traj = self.planner.plan(self.corrected_pose, goal, mu=1.0)
                        if self.traj.shape[1] > 0:
                           
                            self.frontier_timer = 0
                        else:
                        
                            self.frontier_timer = 25  # Wander more before next attempt
                    else:
                        # No frontier found - but keep wandering to build more map
                        if self.no_frontier_count >= 3 or len(self.visited_frontiers) >= 10:  # Only give up after 3 consecutive failures or 10 visited frontiers
                            print(f"Exploration complete! returning home")
                        
                            self.state = "RETURNING"
                            self.traj = self.planner.plan(
                                self.corrected_pose, np.array([0.0, 0.0, 0.0]), mu=1.0
                            )
                        else:
                            print("No frontier found, wandering to build map…")
                            self.frontier_timer = 25  # Wander more before next attempt

            # Follow trajectory or wander
            if self.traj.shape[1] > 0:
                return self.control_follow_traj()
            
            # Explore mode: wander around to build map
            return reactive_obst_avoid(self.lidar())

        # RETURNING HOME
        if self.traj.shape[1] > 0:
            return self.control_follow_traj()
        
        # Reached home
        if self.counter % 10 == 0:
            print(f"    Home reached. Explored {self.exploration_attempts} frontiers.")
        return {"forward": 0.0, "rotation": 0.0}