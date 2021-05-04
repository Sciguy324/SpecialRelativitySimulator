import pygame
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Callable, Tuple


# Initialize pygame
pygame.init()


class Simulation:
    """Class for running simulations in special relativity"""

    def __init__(self, base_dt: float, t_max: float, forces_func: Callable[..., Tuple[np.ndarray, np.ndarray]] = None, c: float = 10.0):
        """physics_func: additional function to apply forces
        base_dt: time-step relative to the user
        t_max: final time, in the origin's frame, to simulate to"""
        # Declare pygame window variables
        self.win_size = np.array([1024, 576])
        self.window = pygame.Surface(self.win_size)
        self.visible = False

        # Store function to use in forces calculations
        if forces_func is None:
            self.forces_func = self._default_force_func
        else:
            self.forces_func = forces_func

        # Constants
        self.c = c

        # Initialize clock
        self.clock = pygame.time.Clock()
        self.base_dt = base_dt
        self.t_max = t_max
        self.time = []

        # Initialize variable to store the index of the reference frame
        # -1 is interpreted as the origin
        self.reference_index = -1

        # Initialize position arrays
        self.x = np.array([])
        self.y = np.array([])
        # Initialize velocity arrays
        self.vx = np.array([])
        self.vy = np.array([])
        # Initialize force arrays
        self.Fx = np.array([])
        self.Fy = np.array([])
        # Initialize particle property arrays
        self.mass = np.array([])
        self.charge = np.array([])

        # Declare historical arrays
        self.x_history = []
        self.y_history = []
        self.vx_history = []
        self.vy_history = []

        # Initialize list to keep track of polygons
        # Polygons are represented as arrays of indices corresponding to the points outlining the polygon
        self.polygons = []

        # Initialize font for on-screen info
        self.font = pygame.font.SysFont('arial', 24)

    @property
    def object_count(self):
        return self.x_history[0].shape[0]

    def _default_force_func(self, *args):
        """Calculates the forces acting on particles in the absence of forces.  This is the default behavior."""
        return np.zeros(self.object_count), np.zeros(self.object_count)

    def basic_random_start(self, x_range: Tuple[float, float], y_range: Tuple[float, float],
                           vx_range: Tuple[float, float], vy_range: Tuple[float, float], q_range: Tuple[float, float],
                           max_m: float, object_count: int):
        """Randomly populates the simulation with a collection of points."""
        # Randomly generate positions
        self.x = np.append(self.x, (x_range[1]-x_range[0]) * np.random.rand(object_count) + x_range[0])
        self.y = np.append(self.y, (y_range[1]-y_range[0]) * np.random.rand(object_count) + y_range[0])

        # Randomly generate velocities
        self.vx = np.append(self.vx, (vx_range[1]-vx_range[0]) * np.random.rand(object_count) + vx_range[0])
        self.vy = np.append(self.vy, (vy_range[1]-vy_range[0]) * np.random.rand(object_count) + vy_range[0])

        # Randomly generate particle attributes
        self.mass = np.append(self.mass, max_m * np.random.rand(object_count))
        self.charge = np.append(self.charge, (q_range[1]-q_range[0]) * np.random.rand(object_count) + q_range[0])

        # Build force arrays
        self.Fx = np.append(self.Fx, np.zeros(object_count))
        self.Fy = np.append(self.Fy, np.zeros(object_count))

    def add_point(self, x: float, y: float, vx: float, vy: float, mass: float, charge: float):
        """Adds a point-mass to the system"""
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.vx = np.append(self.vx, vx)
        self.vy = np.append(self.vy, vy)
        self.mass = np.append(self.mass, mass)
        self.charge = np.append(self.charge, charge)

    def add_polygon(self, x_array: list, y_array: list, x0: float, y0: float, vx: float, vy: float, mass_array: list = None, charge_array: list = None):
        """Adds a polygon to the system.  x_array and y_array denote the positions of the points in the polygon's own
        frame.  The polygon is then transformed to the origin's reference frame and placed relative to (x0, y0)"""
        # Get the number of new points being added
        object_count = len(x_array)

        # Register the indices as part of a polygon
        self.polygons.append(np.arange(self.x.shape[0], self.x.shape[0]+object_count, 1, dtype=int))

        # Apply the inverse Lorentz boost to obtain the location of the points in the origin's reference frame
        x_prime = np.array(x_array)
        y_prime = np.array(y_array)
        gamma = 1 / np.sqrt(1 - (vx**2+vy**2)/self.c**2)

        # Obtain unit vector in the direction of the velocity
        if vx == 0.0 and vy == 0.0:
            n = np.array([0.0, 0.0])
        else:
            n = np.array([vx, vy]) / np.sqrt(vx**2 + vy**2)

        # Apply length contraction
        x = x_prime + (1 / gamma - 1) * (x_prime * n[0] + y_prime * n[1]) * n[0]
        y = y_prime + (1 / gamma - 1) * (x_prime * n[0] + y_prime * n[1]) * n[1]

        # Apply the coordinate shift
        x, y = x + x0, y + y0

        # Add the attributes to the simulation
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.vx = np.append(self.vx, [vx]*object_count)
        self.vy = np.append(self.vy, [vy]*object_count)

        # If no mass array is specified, assume m=1
        if mass_array is None:
            self.mass = np.append(self.mass, np.ones(object_count))
        else:
            self.mass = np.append(self.mass, mass_array)

        # If no charge array is specified, assume q=0
        if charge_array is None:
            self.charge = np.append(self.charge, np.zeros(object_count))
        else:
            self.charge = np.append(self.charge, charge_array)

    def force_to_acceleration(self, vx: np.ndarray, vy: np.ndarray, mass: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> (np.ndarray, np.ndarray):
        """Computes relativistic acceleration on all points given the forces acting on all points"""
        # Compute relativistic acceleration
        velocity_squared = vx ** 2 + vy ** 2
        gamma = 1 / np.sqrt(1 - velocity_squared / self.c ** 2)
        force_dot_vel = fx * vx + fy * vy

        ax = 1 / (mass * gamma) * (fx - force_dot_vel / self.c ** 2 * vx)
        ay = 1 / (mass * gamma) * (fy - force_dot_vel / self.c ** 2 * vy)

        # Clean up NaN values by setting them to zero—NaN generally occurs for particles travelling >= c
        ax = np.nan_to_num(ax, nan=0.0)
        ay = np.nan_to_num(ay, nan=0.0)

        # Return the results
        return ax, ay

    def euler_sim(self):
        """Apply Euler's method for a single iteration"""
        # Calculate the forces on all objects
        fx, fy = self.forces_func(self.time,
                                  self.x_history, self.y_history,
                                  self.vx_history, self.vy_history,
                                  self.mass, self.charge)

        # Compute relativistic acceleration
        ax, ay = self.force_to_acceleration(self.vx, self.vy, self.mass, fx, fy)

        # Apply Euler's method
        self.x = self.x + self.vx * self.base_dt
        self.y = self.y + self.vy * self.base_dt

        self.vx = self.vx + ax * self.base_dt
        self.vy = self.vy + ay * self.base_dt

        # Record to history
        self.time.append(self.time[-1] + self.base_dt)
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.vx_history.append(self.vx)
        self.vy_history.append(self.vy)

    def rk4_sim(self):
        """Apply a 4th order Runge-Kutta method for a single iteration"""
        # Current time for future conciseness
        t = self.time[-1]

        # Calculate the forces on all objects
        fx, fy = self.forces_func(self.time,
                                  self.x_history, self.y_history,
                                  self.vx_history, self.vy_history,
                                  self.mass, self.charge)

        # Compute relativistic acceleration
        ax, ay = self.force_to_acceleration(self.vx, self.vy, self.mass, fx, fy)

        # k1
        k1_vx = self.base_dt * ax
        k1_vy = self.base_dt * ay
        k1_x = self.base_dt * self.vx
        k1_y = self.base_dt * self.vy

        # k2
        fx, fy = self.forces_func(self.time + [t+self.base_dt/2],
                                  self.x_history + [self.x + k1_x/2], self.y_history + [self.y + k1_y/2],
                                  self.vx_history + [self.vx + k1_vx/2], self.vy_history + [self.vy + k1_vy/2],
                                  self.mass, self.charge)
        ax, ay = self.force_to_acceleration(self.vx + k1_vx/2, self.vy + k1_vy/2, self.mass, fx, fy)
        k2_x = self.base_dt * (self.vx + k1_vx/2)
        k2_y = self.base_dt * (self.vy + k1_vy / 2)
        k2_vx = self.base_dt * ax
        k2_vy = self.base_dt * ay

        # k3
        fx, fy = self.forces_func(self.time + [t + self.base_dt/2],
                                  self.x_history + [self.x + k2_x/2], self.y_history + [self.y + k2_y/2],
                                  self.vx_history + [self.vx + k2_vx/2], self.vy_history + [self.vy + k2_vy/2],
                                  self.mass, self.charge)
        # fx, fy = self.forces_func(self.x + k2_x/2, self.y + k2_y/2,
        #                          self.vx + k2_vx/2, self.vy + k2_vy/2,
        #                          self.mass, self.charge)
        ax, ay = self.force_to_acceleration(self.vx + k2_vx/2, self.vy + k2_vy/2, self.mass, fx, fy)
        k3_x = self.base_dt * (self.vx + k2_vx / 2)
        k3_y = self.base_dt * (self.vy + k2_vy / 2)
        k3_vx = self.base_dt * ax
        k3_vy = self.base_dt * ay

        # k4
        fx, fy = self.forces_func(self.time + [t + self.base_dt],
                                  self.x_history + [self.x + k3_x], self.y_history + [self.y + k3_y],
                                  self.vx_history + [self.vx + k3_vx], self.vy_history + [self.vy + k3_vy],
                                  self.mass, self.charge)
        # fx, fy = self.forces_func(self.x + k3_x, self.y + k3_y,
        #                          self.vx + k3_vx, self.vy + k3_vy,
        #                          self.mass, self.charge)
        ax, ay = self.force_to_acceleration(self.vx + k3_x, self.vy + k3_vy, self.mass, fx, fy)
        k4_x = self.base_dt * (self.vx + k3_vx)
        k4_y = self.base_dt * (self.vy + k3_vy)
        k4_vx = self.base_dt * ax
        k4_vy = self.base_dt * ay

        # Final update
        self.x = self.x + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        self.y = self.y + (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
        self.vx = self.vx + (k1_vx + 2*k2_vx + 2*k3_vx + k4_vx) / 6
        self.vy = self.vy + (k1_vy + 2*k2_vy + 2*k3_vy + k4_vy) / 6

        # Record to history
        self.time.append(self.time[-1] + self.base_dt)
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.vx_history.append(self.vx)
        self.vy_history.append(self.vy)

    def lorentz_boost(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """Calculated the Lorentz boosted position of all objects"""
        # Obtain the velocity and space-time position relative to the origin
        if self.reference_index == -1:
            # Currently in the origin's frame
            ref_vx_history = np.zeros(len(self.vx_history))
            ref_vy_history = np.zeros(len(self.vy_history))
        else:
            # Currently in some other frame
            ref_vx_history = np.array(self.vx_history)[:, self.reference_index]
            ref_vy_history = np.array(self.vy_history)[:, self.reference_index]

        # Compute the lorentz factor of the frame
        gamma_history = 1 / np.sqrt(1 - (ref_vx_history ** 2 + ref_vy_history ** 2) / self.c ** 2)

        # Compute the unit velocity vector (relative to the origin), removing any NaNs that might appear
        n_x_history = ref_vx_history / np.sqrt(ref_vx_history**2 + ref_vy_history**2)
        n_y_history = ref_vy_history / np.sqrt(ref_vx_history**2 + ref_vy_history**2)
        n_x_history = np.nan_to_num(n_x_history, nan=0.0)
        n_y_history = np.nan_to_num(n_y_history, nan=0.0)

        # Convert history to numpy arrays
        # Format:
        # [Time 1: [object 1, object 2, object 3... ]
        #  Time 2: [object 1, object 2,...]
        #  Time 3: [...]...]
        x_history = np.array(self.x_history)
        y_history = np.array(self.y_history)

        # Generate t-history using the origin's definition of simultaneity
        t_history = np.tile(np.array(self.time), (self.object_count, 1)).transpose()

        # Time coordinate transform:
        # Use the trapezoidal method to compute the time integral of the Lorentz factor of the current frame's history
        delta_t = t_history[1:] - t_history[:-1]
        gamma_integral = np.cumsum(((gamma_history[1:] + gamma_history[:-1])/2)[:, None] * delta_t, axis=0)
        # Add a row of zeros for t=0
        gamma_integral = np.insert(gamma_integral, 0, np.zeros(gamma_integral.shape[1])).reshape(t_history.shape)

        # Compute the time-shift due to velocity and position
        time_shift = gamma_history[:, None] * (ref_vx_history[:, None] * x_history + ref_vy_history[:, None] * y_history) / self.c**2

        # Compute t-prime
        t_history_prime = gamma_integral - time_shift

        # SPATIAL COORDINATE TRANSFORM:
        # Compute the integral of the Lorentz factor times the velocity from t=0 to t=now
        vx_gamma_integral = np.cumsum(((ref_vx_history[1:] * gamma_history[1:] + ref_vx_history[:-1] * gamma_history[:-1])/2)[:, None] * delta_t[:], axis=0)
        vy_gamma_integral = np.cumsum(((ref_vy_history[1:] * gamma_history[1:] + ref_vy_history[:-1] * gamma_history[:-1])/2)[:, None] * delta_t[:], axis=0)
        # Add a row of zeros for t=0
        vx_gamma_integral = np.insert(vx_gamma_integral, 0, np.zeros(vx_gamma_integral.shape[1])).reshape(t_history.shape)
        vy_gamma_integral = np.insert(vy_gamma_integral, 0, np.zeros(vy_gamma_integral.shape[1])).reshape(t_history.shape)

        x_pos_shift = (gamma_history - 1)[:, None] * (x_history*n_x_history[:, None] + y_history*n_y_history[:, None]) * n_x_history[:, None]
        y_pos_shift = (gamma_history - 1)[:, None] * (x_history*n_x_history[:, None] + y_history*n_y_history[:, None]) * n_y_history[:, None]

        x_history_prime = x_history + x_pos_shift - vx_gamma_integral
        y_history_prime = y_history + y_pos_shift - vy_gamma_integral

        # Return the results
        return t_history_prime, x_history_prime, y_history_prime

    def obtain_present(self, time_index: int, t_history_prime: np.ndarray, x_history_prime: np.ndarray, y_history_prime: np.ndarray):
        """Obtain the present view of the reference object at the given time_index"""
        # Obtain the current (t, x, y) position of the selected reference frame relative to the origin's frame
        if self.reference_index == -1:
            frame_x_prime = 0.0
            frame_y_prime = 0.0
            frame_t_prime = self.time[time_index]
        else:
            frame_x_prime = x_history_prime[time_index][self.reference_index]
            frame_y_prime = y_history_prime[time_index][self.reference_index]
            frame_t_prime = t_history_prime[time_index][self.reference_index]

        # Shift the primed frame to place the target object at the center
        x_offset = frame_x_prime
        y_offset = frame_y_prime
        x_history_prime = x_history_prime - x_offset
        y_history_prime = y_history_prime - y_offset

        # Find the positions of everything in the object's present
        present_time = frame_t_prime
        time_diff = np.abs(t_history_prime - present_time)
        other_present_index = np.nanargmin(time_diff, axis=0)

        current_x = x_history_prime[other_present_index, np.arange(x_history_prime.shape[1])]
        current_y = y_history_prime[other_present_index, np.arange(y_history_prime.shape[1])]

        # Return the results
        return current_x, current_y, frame_t_prime

    def draw_screen(self, x: np.ndarray, y: np.ndarray):
        """Draws all objects on the screen"""
        # Get the center of the window—the reference frame should be drawn at the center.
        # Coordinates must be integers to appease pygame
        win_center = (self.win_size / 2).astype(int)

        # Clear the screen
        self.window.fill((0, 0, 0))

        # Obtain all point indices that are part of a polygon
        if len(self.polygons) > 0:
            polygon_points = np.concatenate(self.polygons)
        else:
            polygon_points = []

        mouse_x, mouse_y = pygame.mouse.get_pos()

        # If the mouse is hovering over a point, mark that point as having a special color
        dist = (x - mouse_x + win_center[0])**2 + (y - mouse_y + win_center[1])**2
        nearest_index = np.argmin(dist)
        if dist[nearest_index] > 10**2:
            nearest_index = -1

        # Draw particles
        for i, (px, py) in enumerate(zip(x, y)):
            # Don't draw polygon points as a particle
            if i not in polygon_points:
                # Default color: red
                if self.charge[i] > 0.0:
                    color = np.array((128, 255, 128))
                else:
                    color = np.array((255, 0, 0))

                # Mouse hovering over particle: aqua
                if i == nearest_index:
                    color = np.array((0, 255, 255))

                # Followed particle: green
                if i == self.reference_index:
                    color = np.array((0, 255, 0))

                # Draw circle to represent the particle
                pygame.draw.circle(self.window, color, win_center + (int(px), int(py)), 10)

                # Draw charge indicator
                if self.charge[i] != 0.0:
                    # Draw minus sign if the charge is non-zero
                    pygame.draw.line(self.window, (0, 0, 0),
                                     (win_center+(int(px)-5, int(py))), (win_center+(int(px)+5, int(py))))

                if self.charge[i] > 0.0:
                    # Add another line to turn the minus into a plus, if applicable
                    pygame.draw.line(self.window, (0, 0, 0),
                                     (win_center + (int(px), int(py)-5)), (win_center + (int(px), int(py)+5)))

        # Draw polygons
        for polygon in self.polygons:
            # Obtain coordinates of the points in the polygon
            points = np.array(list(zip(x[polygon], y[polygon])))

            # Draw the polygon
            pygame.draw.polygon(self.window, (255, 0, 0), win_center+points.astype(int))

            # Draw a smaller point to indicate that this is something you can click
            for i, j in points:
                pygame.draw.circle(self.window, (255, 0, 0), win_center+(int(i), int(j)), 5)

    def draw_info(self, paused: bool, present_time: float, primed_time: float, mouse_pos: np.ndarray, ruler_start: np.ndarray):
        """Adds informational items/text"""
        # Obtain the center of the window
        win_center = (self.win_size / 2).astype(int)

        # Obtain mouse position relative to the center of the window
        shifted_mouse_pos = mouse_pos - win_center

        # Draw the crosshair
        pygame.draw.line(self.window, (0, 128, 128), win_center - (10, 0), win_center + (10, 0), 2)
        pygame.draw.line(self.window, (0, 128, 128), win_center - (0, 10), win_center + (0, 10), 2)

        # Draw info
        proper_time_text = self.font.render(f"Primed Time: {primed_time:.2f}", 12, (255, 255, 255))
        origin_time_text = self.font.render(f"Origin Time:  {present_time:.2f}", 12, (255, 255, 255))
        position_text = self.font.render(f"Mouse Pos: {shifted_mouse_pos[0]}, {shifted_mouse_pos[1]}", 12, (255, 255, 255))
        self.window.blit(proper_time_text, (45, 0))
        self.window.blit(origin_time_text, (45, 30))
        self.window.blit(position_text, (10, 60))

        # Pause indicator
        if paused:
            pygame.draw.rect(self.window, (128, 128, 128), (10, 5, 10, 20))
            pygame.draw.rect(self.window, (128, 128, 128), (25, 5, 10, 20))
        else:
            pygame.draw.polygon(self.window, (128, 128, 128), ([10, 5], [35, 15], [10, 25]))

        # If the user is holding right-click: draw the ruler and distance indicator
        if pygame.mouse.get_pressed()[2]:
            # Draw distance text
            ruler_distance = np.linalg.norm(mouse_pos - ruler_start)
            ruler_text = self.font.render(f"Ruler: {ruler_distance:.2f}", 12, (255, 255, 255))

            # Draw ruler
            pygame.draw.line(self.window, (0, 0, 255), ruler_start, mouse_pos.astype(int), 5)
        else:
            # Ruler is inactive
            ruler_text = self.font.render(f"Ruler: -", 12, (255, 255, 255))
        self.window.blit(ruler_text, (10, 90))

    def run(self, method: str = 'rk4', print_progress: bool = False):
        """Runs the simulation"""
        # Record initial positions to history
        self.x_history.append(self.x)
        self.y_history.append(self.y)
        self.vx_history.append(self.vx)
        self.vy_history.append(self.vy)
        self.time.append(0.0)

        # Select the requested ODE solver method
        if method == 'euler':
            sim_func = self.euler_sim
        elif method == 'rk4':
            sim_func = self.rk4_sim
        else:
            raise ValueError(f'{method} is not a recognized ODE solving method!')

        # Main loop
        while self.time[-1] < self.t_max:
            # Run the physics calculations
            sim_func()

            # Print the current progress, if applicable
            if print_progress:
                if not len(self.time) % 10:
                    print(f"Time: {self.time[-1]} / {self.t_max}")

    def show(self, t_start: float = 0.0, t_stop: float = None, max_fps: int = 90):
        """Shows an interactive window of the simulation's results"""
        # If user entered a start time, find the nearest time-index to start at
        indexed_time = float(np.abs(np.array(self.time) - t_start).argmin())

        # If user entered a stop time, find nearest time-index to stop at.  Otherwise, assume the end
        if t_stop is None:
            stop_index = float(len(self.time) - 1)
        else:
            stop_index = float(np.abs(np.array(self.time) - t_stop).argmin())

        # Open the window
        pygame.display.init()
        self.window = pygame.display.set_mode(self.win_size, flags=pygame.RESIZABLE)
        pygame.display.set_caption("Special Relativity Simulation")

        # Perform initial calculation of the primed world-lines
        t_history_prime, x_history_prime, y_history_prime = self.lorentz_boost()

        # Main display loop
        done = False
        paused = False
        ruler_start = np.array([0.0, 0.0], dtype=float)
        while not done:
            # Obtain mouse position relative to the center
            mouse_pos = np.array(pygame.mouse.get_pos())

            # Obtain the present positions of everything
            x_prime, y_prime, primed_time = self.obtain_present(int(indexed_time),
                                                                t_history_prime, x_history_prime, y_history_prime)

            # Handle pygame events
            for event in pygame.event.get():
                # User quit the window
                if event.type == pygame.QUIT:
                    done = True

                # User resized the window
                if event.type == pygame.VIDEORESIZE:
                    self.win_size = np.array(event.size)
                    self.window = pygame.display.set_mode(self.win_size, flags=pygame.RESIZABLE)

                    # User pressed a key
                if event.type == pygame.KEYDOWN:
                    # ESC: Return to origin frame
                    if event.key == pygame.K_ESCAPE:
                        self.reference_index = -1

                        # Recalculate Lorentz-boosted world-lines
                        t_history_prime, x_history_prime, y_history_prime = self.lorentz_boost()

                    # Space: Pause
                    if event.key == pygame.K_SPACE:
                        # Toggle pause state
                        paused = not paused

                        # Update window title accordingly
                        pygame.display.set_caption(f"Special Relativity Simulation{' - Paused' if paused else ''}")

                # User clicked on window
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Left mouse button: Attempt to select a reference frame
                    if event.button == pygame.BUTTON_LEFT:
                        # Find distance between the mouse and visual objects
                        shifted_mouse = mouse_pos - self.win_size/2
                        distance_squared = (x_prime - shifted_mouse[0])**2 + (y_prime - shifted_mouse[1])**2

                        # Select the nearest point
                        nearest_index = np.argmin(distance_squared)
                        # Only switch frames if the nearest point is near enough (within 1 visual radius)
                        if distance_squared[nearest_index] < 10**2:
                            # Check to make sure the target object is not at (or exceeding?) the speed of light
                            if self.vx_history[int(indexed_time)][nearest_index]**2 + self.vy_history[int(indexed_time)][nearest_index]**2 >= self.c**2:
                                print("Target object is superluminal!?")
                                nearest_index = -1

                            # Confirm the new reference index
                            self.reference_index = nearest_index

                            # Recalculate the primed world-lines.  This only needs to be run once upon changing frames
                            # since it calculates across the entire history of the world-lines.
                            t_history_prime, x_history_prime, y_history_prime = self.lorentz_boost()

                    # Right mouse button: start ruler
                    if event.button == pygame.BUTTON_RIGHT:
                        # Save the starting position of the ruler
                        ruler_start = np.array(pygame.mouse.get_pos())

            # Draw the screen
            self.draw_screen(x_prime, y_prime)

            # Draw info
            self.draw_info(paused, self.time[int(indexed_time)], primed_time, mouse_pos, ruler_start)

            # Flip the display buffer with the window
            pygame.display.flip()

            # Call the clock to limit the FPS
            self.clock.tick(max_fps)

            # Unless the simulation has reached it's end-point, increment the time
            if indexed_time < stop_index and not paused:
                indexed_time += 1.0

        # Close the display module
        pygame.display.quit()

    def plot(self, reference_frame: int = -1, x_lim: tuple = None, y_lim: tuple = None, t_lim: tuple = None):
        """Plot the world-lines from the simulation"""
        # Set up the plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Switch to the provided reference frame, if applicable
        self.reference_index = reference_frame
        t_history, x_history, y_history = self.lorentz_boost()

        # Plot the world lines
        # Reshape the x-history and y-history such that the first index denotes an object's worldline
        x_positions = np.transpose(np.array(x_history))
        y_positions = np.transpose(np.array(y_history))
        t_positions = np.transpose(np.array(t_history))

        for worldline_x, worldline_y, worldline_t in zip(x_positions, y_positions, t_positions):
            ax.plot(worldline_x, worldline_y, worldline_t)

        # Mark the origin
        ax.scatter((0.0,), (0.0,), color="red")

        # Plot the light cone
        x_range, y_range = np.meshgrid(np.linspace(np.min(x_positions), np.max(x_positions), 10),
                                       np.linspace(np.min(y_positions), np.max(y_positions), 10))
        z_cone = np.sqrt(x_range**2 + y_range**2) / self.c
        #ax.plot_wireframe(x_range, y_range, z_cone, edgecolor='green')

        # Configure the plot
        ax.set_xlabel('X-Position')
        ax.set_ylabel('Y-Position')
        ax.set_zlabel('Time')
        plt.title("World-Lines From Simulation")
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(t_lim)

        # Show the plot
        plt.show()

    def save(self, file_name: str, compressed: bool = True):
        """Saves simulation results to a .csv file"""
        # Put the data into a pandas data-frame
        # Convert histories into a 1D array for the purposes of saving
        array_length = len(self.time) * self.object_count
        x_history = np.array(self.x_history).reshape(array_length)
        y_history = np.array(self.y_history).reshape(array_length)
        vx_history = np.array(self.vx_history).reshape(array_length)
        vy_history = np.array(self.vy_history).reshape(array_length)

        # Load all data into a data-frame
        data = pd.DataFrame(data=[x_history, y_history, vx_history, vy_history, np.array(self.time), self.mass,
                                  self.charge, np.array([self.c, len(self.time), self.object_count])],
                            index=["X-History", "Y-History", "VX-History", "VY-History", "Time", "Mass", "Charge",
                                   "Parameters"])

        # Save to file
        if compressed:
            data.to_csv(file_name, compression="gzip")
        else:
            data.to_csv(file_name)

    def load(self, file_name: str, compressed: bool = True):
        """Loads simulation data from a .csv file"""
        # Load data-frame from file
        if compressed:
            data = pd.read_csv(file_name, index_col=0, compression="gzip")
        else:
            data = pd.read_csv(file_name, index_col=0)

        # Obtain size to reshape to history arrays to
        time_length, object_count = int(data.loc["Parameters"][1]), int(data.loc["Parameters"][2])

        # Load data from data-frame into simulation data
        self.x_history = list(data.loc["X-History"].to_numpy().reshape(time_length, object_count))
        self.y_history = list(data.loc["Y-History"].to_numpy().reshape(time_length, object_count))
        self.vx_history = list(data.loc["VX-History"].to_numpy().reshape(time_length, object_count))
        self.vy_history = list(data.loc["VY-History"].to_numpy().reshape(time_length, object_count))
        self.time = list(data.loc["Time"].to_numpy()[:time_length])
        self.mass = data.loc["Mass"].to_numpy()[:object_count]
        self.charge = data.loc["Charge"].to_numpy()[:object_count]
        self.c = data.loc["Parameters"][0]
