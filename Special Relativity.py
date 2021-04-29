import pygame
import numpy as np
import matplotlib.pyplot as plt
#from scipy.constants import epsilon_0
epsilon_0 = 1.0e-6


# Initialize pygame
pygame.init()


class Simulation:
    """Class for running simulations in special relativity"""

    def __init__(self, base_dt, t_max, forces_func=None, c=10.0):
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
        return self.x.shape[0]

    def _default_force_func(self, *args):
        """Calculates the forces acting on particles in the absence of forces.  The default behavior."""
        return np.zeros(self.object_count), np.zeros(self.object_count)

    def basic_random_start(self, x_range: tuple, y_range: tuple, vx_range: tuple, vy_range: tuple, q_range: tuple, max_m: float, object_count: int):
        """Randomly generates starting conditions.  Particles are uncharged
        Note: this will erase any currently stored points"""
        # Randomly generate positions
        self.x = (x_range[1]-x_range[0]) * np.random.rand(object_count) + x_range[0]
        self.y = (y_range[1]-y_range[0]) * np.random.rand(object_count) + y_range[0]

        # Randomly generate velocities
        self.vx = (vx_range[1]-vx_range[0]) * np.random.rand(object_count) + vx_range[0]
        self.vy = (vy_range[1]-vy_range[0]) * np.random.rand(object_count) + vy_range[0]

        # Randomly generate particle attributes
        self.mass = max_m * np.random.rand(object_count)
        self.charge = (q_range[1]-q_range[0]) * np.random.rand(object_count) + q_range[0]

        # Build force arrays
        self.Fx = np.zeros(object_count)
        self.Fy = np.zeros(object_count)

    def add_point(self, x, y, vx, vy, mass, charge):
        """Adds a point-mass to the system"""
        self.x = np.append(self.x, x)
        self.y = np.append(self.y, y)
        self.vx = np.append(self.vx, vx)
        self.vy = np.append(self.vy, vy)
        self.mass = np.append(self.mass, mass)
        self.charge = np.append(self.charge, charge)

    def add_polygon(self, x_array, y_array, x0, y0, vx, vy, mass_array=None, charge_array=None):
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

    def force_to_acceleration(self, vx, vy, mass, fx, fy):
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

    def lorentz_boost(self, time_index, x, y, vx, vy):
        """Calculated the Lorentz boosted position of all objects"""
        # Obtain the velocity relative to the origin
        if self.reference_index == -1:
            # Currently in the origin's frame
            vel = np.array([0.0, 0.0])
            frame_x, frame_y, frame_t = 0.0, 0.0, self.time[time_index]
            n = np.array([0.0, 0.0])
            gamma = 1.0
        else:
            # Currently in some other frame
            vel = np.array([vx[self.reference_index], vy[self.reference_index]])

            # Get position of other frame according to origin
            frame_x, frame_y, frame_t = x[self.reference_index], y[self.reference_index], self.time[time_index]

            # Compute the lorentz factor of the frame
            gamma = 1 / np.sqrt(1 - np.linalg.norm(vel)**2 / self.c**2)

            # Compute the unit velocity vector (relative to the origin), removing any NaNs that might appear
            n = vel / np.linalg.norm(vel)
            n = np.nan_to_num(n, nan=0.0)

        # Obtain unit vector in the direction of the velocity
        #n_x_history = vel_x_history / np.sqrt(vel_x_history**2 + vel_y_history**2)
        #n_y_history = vel_y_history / np.sqrt(vel_y_history**2 + vel_y_history**2)
        #n_x_history = np.nan_to_num(n_x_history, nan=0.0)
        #n_y_history = np.nan_to_num(n_y_history, nan=0.0)

        # Convert history to numpy arrays
        # Format [time 1: [object1:[property], object2,...], time 2:..., time 3:...]
        # [Time 1: [object 1, object 2, object 3... ]
        #  Time 2: [object 1, object 2,...]
        #  Time 3: [...]...]
        x_history = np.array(self.x_history)
        y_history = np.array(self.y_history)
        vx_history = np.array(self.vx_history)
        vy_history = np.array(self.vy_history)

        # Generate t-history using the origin's definition of simultaneity
        t_history = np.tile(np.array(self.time), (self.object_count, 1)).transpose()

        # TODO: Implement generalized reference frame shift
        # Compute gamma history
        #gamma_history = 1/np.sqrt((vel_x_history**2 + vel_y_history**2)/self.c)

        # Apply the Lorentz-boost to convert to a frame moving at the same velocity as the reference index's
        # TIME COORDINATE TRANSFORM:
        # Compute the integral of the Lorentz factor from t=0 to t=now using the trapezoidal method
        #delta_t = t_history[1:] - t_history[:-1]
        #gamma_integral = np.cumsum((gamma_history[1:] + gamma_history[:-1])/2 * delta_t, axis=0)


        #gamma_integral = np.insert(gamma_integral, 0, np.zeros(gamma_integral.shape[1]), axis=0)

        # SPATIAL COORDINATE TRANSFORM:
        t_history_prime = gamma * (t_history - (vel[0] * x_history + vel[1] * y_history) / self.c**2)
        #t_history_prime = gamma_integral - gamma_history*(vel_x_history*x_history + vel_y_history) / self.c**2

        # Compute the integral of the Lorentz factor times the velocity from t=0 to t=now
        #vel_x_gamma_integral = np.cumsum((vel_x_history[1:] * gamma_history[1:] + vel_x_history[:-1] * gamma_history[:-1])/2 * delta_t, axis=0)
        #vel_y_gamma_integral = np.cumsum((vel_y_history[1:] * gamma_history[1:] + vel_y_history[:-1] * gamma_history[:-1])/2 * delta_t, axis=0)

        #x_history_prime = x_history + (gamma_history - 1) * (x_history * n_x_history + y_history * n_y_history) * n_x_history - vel_x_gamma_integral
        #y_history_prime = y_history + (gamma_history - 1) * (x_history * n_x_history + y_history * n_y_history) * n_y_history - vel_x_gamma_integral

        x_history_prime = x_history + (gamma-1)*(x_history*n[0] + y_history*n[1]) * n[0] - gamma*t_history*vel[0]
        y_history_prime = y_history + (gamma-1)*(x_history*n[0] + y_history*n[1]) * n[1] - gamma*t_history*vel[1]

        frame_t_prime = gamma * (frame_t - (vel[0] * frame_x + vel[1] * frame_y) / self.c**2)
        frame_x_prime = frame_x + (gamma-1)*(frame_x*n[0] + frame_y*n[1]) * n[0] - gamma*frame_t*vel[0]
        frame_y_prime = frame_y + (gamma-1)*(frame_x*n[0] + frame_y*n[1]) * n[1] - gamma*frame_t*vel[1]

        #if self.reference_index == -1:
        #    frame_x_prime = 0.0
        #    frame_y_prime = 0.0
        #    frame_t_prime = frame_t
        #else:
        #    frame_x_prime = x_history_prime[-1][self.reference_index]
        #    frame_y_prime = y_history_prime[-1][self.reference_index]
        #    frame_t_prime = t_history_prime[-1][self.reference_index]

        # Shift the primed frame to place the target object at the center
        x_offset = frame_x_prime
        y_offset = frame_y_prime
        x_history_prime = x_history_prime - x_offset
        y_history_prime = y_history_prime - y_offset

        # Find the positions of everything in the object's present
        present_time = frame_t_prime
        time_diff = np.abs(t_history_prime - present_time)
        other_present_index = time_diff.argmin(axis=0)

        current_x = x_history_prime[other_present_index, np.arange(x_history_prime.shape[1])]
        current_y = y_history_prime[other_present_index, np.arange(y_history_prime.shape[1])]

        # Return the results
        return current_x, current_y, frame_t_prime

    def draw_screen(self, x, y):
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

    def draw_info(self, paused, present_time, proper_time, mouse_pos, ruler_start):
        """Adds informational items/text"""
        # Obtain the center of the window
        win_center = (self.win_size / 2).astype(int)

        # Obtain mouse position relative to the center of the window
        shifted_mouse_pos = mouse_pos - win_center

        # Draw the crosshair
        pygame.draw.line(self.window, (0, 128, 128), win_center - (10, 0), win_center + (10, 0), 2)
        pygame.draw.line(self.window, (0, 128, 128), win_center - (0, 10), win_center + (0, 10), 2)

        # Draw info
        proper_time_text = self.font.render(f"Proper Time: {proper_time:.2f}", 12, (255, 255, 255))
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

    def run(self, method='rk4', print_progress=False):
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

    def show(self, t_start=0.0, t_stop=None, max_fps=90):
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
        self.window = pygame.display.set_mode(self.win_size)
        pygame.display.set_caption("Special Relativity Simulation")

        # Main display loop
        done = False
        paused = False
        ruler_start = np.array([0.0, 0.0], dtype=float)
        while not done:
            # Obtain mouse position relative to the center
            mouse_pos = np.array(pygame.mouse.get_pos())

            # Compute effect of Lorentz boost
            x_prime, y_prime, proper_time = self.lorentz_boost(int(indexed_time),
                                                               self.x_history[int(indexed_time)],
                                                               self.y_history[int(indexed_time)],
                                                               self.vx_history[int(indexed_time)],
                                                               self.vy_history[int(indexed_time)])

            # Handle pygame events
            for event in pygame.event.get():
                # User quit the window
                if event.type == pygame.QUIT:
                    done = True

                # User pressed a key
                if event.type == pygame.KEYDOWN:
                    # ESC: Return to origin frame
                    if event.key == pygame.K_ESCAPE:
                        self.reference_index = -1

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

                            self.reference_index = nearest_index

                    # Right mouse button: start ruler
                    if event.button == pygame.BUTTON_RIGHT:
                        # Save the starting position of the ruler
                        ruler_start = np.array(pygame.mouse.get_pos())

            # Draw the screen
            self.draw_screen(x_prime, y_prime)

            # Draw the light cones of the initial positions
            for i, j in zip(self.x_history[0], self.y_history[0]):
                break
                try:
                    pygame.draw.circle(self.window, (0, 0, 255), (self.win_size/2 + (i, j)).astype(int),
                                       int(self.time[int(indexed_time)] * self.c), 2)
                except ValueError:
                    pass

            # Draw info
            self.draw_info(paused, self.time[int(indexed_time)], proper_time, mouse_pos, ruler_start)

            # Flip the display buffer with the window
            pygame.display.flip()

            # Call the clock to limit the FPS
            self.clock.tick(max_fps)

            # Unless the simulation has reached it's end-point, increment the time
            if indexed_time < stop_index and not paused:
                indexed_time += 1.0

        # Close the display module
        pygame.display.quit()

    def plot(self):
        """Plot the world-lines from the simulation"""
        # Set up the plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Plot the world lines
        # Reshape the x-history and y-history such that the first index denotes a worldline
        x_positions = np.transpose(np.array(self.x_history))
        y_positions = np.transpose(np.array(self.y_history))

        # TODO: Make it so polygons are rendered as polygons rather than ordinary points

        for worldline_x, worldline_y in zip(x_positions, y_positions):
            ax.plot(worldline_x, worldline_y, self.time)

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

        # Show the plot
        plt.show()


# Main program
def main():
    # Test 1: Random particles
    # test1()

    # Test 2: Polygon
    # test2()

    # Test 3: Motionless random particles
    # test3()

    # Test 4: Barn and ladder
    test4()

    # Test 5: Constant force
    # test5()

    # Test 6: Electrostatic force
    # test6()

    # Test 7: Moving line of charge
    # test7()


def test1():
    """Test a random start"""
    # Declare simulation
    sim = Simulation(0.1, 100, c=2.0)

    # Add particles
    sim.basic_random_start((-20.0, 20.0), (-20.0, 20.0),
                           (-1.0, 1.0), (-1.0, 1.0),
                           (0.0, 0.0),
                           1.0,
                           10)

    # Run the simulation
    sim.run()

    # Show the simulation
    sim.show()

    # Plot the simulation
    sim.plot()


def test2():
    """Test a polygon"""
    # Declare simulation
    sim = Simulation(0.01, 10)

    # Add particles
    sim.add_polygon([0, 50, 50, 0], [0, 0, 50, 50], 10.0, 10.0, 7.0, 7.0)

    # Run the simulation
    sim.run()

    # Show the simulation
    sim.show()

    # Plot the simulation
    sim.plot()


def test3():
    """Test a random start, no motion"""
    # Declare simulation
    sim = Simulation(0.1, 100)

    # Add particles
    sim.basic_random_start((-20.0, 20.0), (-20.0, 20.0),
                           (0.0, 0.0), (0.0, 0.0),
                           (0.0, 0.0),
                           1.0,
                           10)

    # Run the simulation
    sim.run()

    # Show the simulation
    sim.show()

    # Plot the simulation
    sim.plot()


def test4():
    """Test the barn-and-ladder paradox"""
    # Declare simulation
    sim = Simulation(0.01, 50)

    # Add observer
    sim.add_point(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    # Add barn
    sim.add_polygon([0, 0, 101, 101, 0, 0, 100, 100], [100, 101, 101, 0, 0, 1, 1, 100], 50, 50, 0.0, 0.0)

    # Add "ladder"
    sim.add_polygon([0, 25, 50, 50, 0], [0, 0, 0, 1, 1], -100, 100, 9.0, 0.0)

    # Run simulation
    sim.run()

    # Show the simulation
    sim.show()

    # Show the simulation up until the pole hits the barn in the furthest reference frame
    sim.show(t_stop=15.24)


def test5():
    """Acceleration of a particle due to a constant force to the right"""
    # Build function for forces
    def forces(t, x, y, vx, vy, mass, charge):
        # Declare force array
        force_x = np.zeros(x[-1].shape)
        force_y = np.zeros(y[-1].shape)

        # Apply a constant force to the third particle
        force_x[2] = 1.0

        # Return the results
        return force_x, force_y

    # Declare simulation
    sim = Simulation(0.01, 50, forces)

    # Add observer
    sim.add_point(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    # Add photon
    sim.add_point(0.0, 10.0, sim.c, 0.0, 0.0, 0.0)

    # Add accelerating particle
    sim.add_point(0.0, -10.0, 0.0, 0.0, 1.0, 0.0)

    # Add a particle with a constant, sub-light velocity
    sim.add_point(0.0, -20.0, sim.c / 2, 0.0, 1.0, 0.0)
    sim.add_point(0.0, -30.0, sim.c / 4, 0.0, 1.0, 0.0)

    # Run the simulation
    sim.run(method='rk4')

    # Show the results
    sim.show()

    # Show a plot of the results
    sim.plot()


def test6():
    """Full, non-instantaneous electrostatic forces"""
    # Declare function for the forces
    def forces(t, x, y, vx, vy, mass, charge):
        """t, x, y, vx and vy are the entire histories of simulation thus far"""
        # Declare force arrays
        fx = np.zeros(x[-1].shape)
        fy = np.zeros(y[-1].shape)

        # Compute distance between objects as they were
        x_histories = np.transpose(np.array(x))
        y_histories = np.transpose(np.array(y))
        t_history = np.array(t)

        present_time = t[-1]

        for i, (xi, yi) in enumerate(zip(x[-1], y[-1])):
            # Get the histories with the exception of this object's
            other_x_histories = np.delete(x_histories, i, axis=0)
            other_y_histories = np.delete(y_histories, i, axis=0)
            other_charges = np.delete(charge, i, axis=0)

            if len(other_x_histories.shape) == 1:
                other_x_histories = np.reshape(other_x_histories, (other_x_histories.shape[0], 1))

            if len(other_y_histories.shape) == 1:
                other_y_histories = np.reshape(other_y_histories, (other_y_histories.shape[0], 1))

            # Compute the space-time interval between this object and all other objects
            s_squared = sim.c**2 * (present_time - t_history)**2 - (xi - other_x_histories)**2 - (yi - other_y_histories)**2

            # Find the index of the interval that's zero
            # s_squared format:
            # [Object 1: [ Time 1, Time 2,...],
            #  Object 2: [ Time 1, Time 2,...],
            #  Object 3: [...],
            #  ...
            zero_index = np.argmin(np.abs(s_squared), axis=1)

            # Check for actual intersection
            check1 = np.abs(s_squared[np.arange(s_squared.shape[0]), zero_index]) > (100 * sim.base_dt * sim.c) ** 2
            # An index of 0 is probably due to the intersection occurring before the start of the simulation
            check2 = zero_index == 0
            valid = np.logical_not(np.logical_or(check1, check2))

            delta_x = other_x_histories[np.arange(other_x_histories.shape[0]), zero_index] - xi
            delta_y = other_y_histories[np.arange(other_x_histories.shape[0]), zero_index] - yi
            distance = np.sqrt(delta_x ** 2 + delta_y ** 2)

            # Compute the magnitude of the force
            f = charge[i] * other_charges[valid] / (4 * np.pi * epsilon_0 * distance[valid] ** 2)

            # Break the force into components and add to net force
            fx[i] = np.sum(f * delta_x[valid] / distance[valid])
            fy[i] = np.sum(f * delta_y[valid] / distance[valid])

        # Return the results
        return fx, fy

    # Declare simulation
    sim = Simulation(0.01, 20, forces, c=10.0)

    # Spawn particles
    v_max = sim.c / np.sqrt(2)
    spread = 50
    sim.basic_random_start((-spread, spread), (-spread, spread),
                           (-0.0, 0.0), (-0.0, 0.0),
                           (-1.0, 1.0),
                           5,
                           10)

    # Run simulation
    sim.run(method='rk4')

    # Show the results
    sim.show()

    sim.plot()


def test7():
    """A moving line of charge with non-instantaneous forces"""
    # Declare function for the forces
    def forces(t, x, y, vx, vy, mass, charge):
        """t, x, y, vx and vy are the entire histories of simulation thus far"""
        # Declare force arrays
        fx = np.zeros(x[-1].shape)
        fy = np.zeros(y[-1].shape)

        # Compute distance between objects as they were
        x_histories = np.transpose(np.array(x))
        y_histories = np.transpose(np.array(y))
        t_history = np.array(t)

        present_time = t[-1]

        for i, (xi, yi) in enumerate(zip(x[-1], y[-1])):
            # Get the histories with the exception of this object's
            other_x_histories = np.delete(x_histories, i, axis=0)
            other_y_histories = np.delete(y_histories, i, axis=0)
            other_charges = np.delete(charge, i, axis=0)

            if len(other_x_histories.shape) == 1:
                other_x_histories = np.reshape(other_x_histories, (other_x_histories.shape[0], 1))

            if len(other_y_histories.shape) == 1:
                other_y_histories = np.reshape(other_y_histories, (other_y_histories.shape[0], 1))

            # Compute the space-time interval between this object and all other objects
            s_squared = sim.c ** 2 * (present_time - t_history) ** 2 - (xi - other_x_histories) ** 2 - (
                        yi - other_y_histories) ** 2

            # Find the index of the interval that's zero
            # s_squared format:
            # [Object 1: [ Time 1, Time 2,...],
            #  Object 2: [ Time 1, Time 2,...],
            #  Object 3: [...],
            #  ...
            zero_index = np.argmin(np.abs(s_squared), axis=1)

            # Check for actual intersection
            check1 = np.abs(s_squared[np.arange(s_squared.shape[0]), zero_index]) > (100 * sim.base_dt * sim.c) ** 2
            # An index of 0 is probably due to the intersection occurring before the start of the simulation
            check2 = zero_index == 0
            valid = np.logical_not(np.logical_or(check1, check2))

            delta_x = other_x_histories[np.arange(other_x_histories.shape[0]), zero_index] - xi
            delta_y = other_y_histories[np.arange(other_x_histories.shape[0]), zero_index] - yi
            distance = np.sqrt(delta_x ** 2 + delta_y ** 2)

            # Compute the magnitude of the force
            f = charge[i] * other_charges[valid] / (4 * np.pi * epsilon_0 * distance[valid] ** 2)

            # Break the force into components and add to net force
            fx[i] = np.sum(f * delta_x[valid] / distance[valid])
            fy[i] = np.sum(f * delta_y[valid] / distance[valid])

        # Return the results
        return fx, fy

    # Declare simulation
    sim = Simulation(0.01, 20, forces, c=20.0)

    # Spawn line of moving charge
    for x_pos in np.linspace(-500.0, 500.0, 15):
        sim.add_point(x_pos, -50.0, 0.9*sim.c, 0.0, 1.0, -1.0)

    # Spawn test charge near the line
    sim.add_point(0.0, 50.0, 0.0, 0.0, 1.0, -1.0)

    # Run the simulation
    sim.run(print_progress=True)

    # Show the results
    sim.show()

    # Show a plot of the results
    sim.plot()


# Run main program
if __name__ == "__main__":
    main()
