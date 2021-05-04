# Imports
from SpecialRelativity import *
import numpy as np


# Main program
def main():
    # Test 1: Random particles
    test1()

    # Test 2: Polygon
    test2()

    # Test 3: Barn and ladder
    test3()

    # Test 4: Constant force
    test4()

    # Test 5: Electrostatic force.  Commented-out due to test 5.1
    # test5()

    # Test 5.1: Electrostatic force loaded from a file
    test5_1()

    # Test 6: Moving particle next to line of charge.  Commented-out due to test 6.1
    # test6()

    # Test 6.1: Moving particle next to line of charge loaded from a file
    test6_1()


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
    """Test the barn-and-ladder paradox"""
    # Declare simulation
    sim = Simulation(0.01, 50)

    # Add observer
    sim.add_point(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    # Add barn
    sim.add_polygon([0, 0, 101, 101, 0, 0, 100, 100], [100, 101, 101, 0, 0, 1, 1, 100], 50, 50, 0.0, 0.0)

    # Add "ladder"
    sim.add_polygon([0, 50, 100, 100, 0], [0, 0, 0, 1, 1], -100, 100, 9.0, 0.0)

    # Run simulation
    sim.run()

    # Show the simulation
    sim.show()


def test4():
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
    sim.run()

    # Show the results
    sim.show()

    # Show a plot of the results
    sim.plot()


def test5():
    """Full, non-instantaneous electrostatic forces"""
    # Work in units where epsilon_0 is 1e-6
    epsilon_0 = 1.0e-6

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
            f = -charge[i] * other_charges[valid] / (4 * np.pi * epsilon_0 * distance[valid] ** 2)

            # Break the force into components and add to net force
            fx[i] = np.sum(f * delta_x[valid] / distance[valid])
            fy[i] = np.sum(f * delta_y[valid] / distance[valid])

        # Return the results
        return fx, fy

    # Declare simulation
    sim = Simulation(0.01, 10, forces, c=10.0)

    # Spawn particles
    spread = 50
    sim.basic_random_start((-spread, spread), (-spread, spread),
                           (-0.0, 0.0), (-0.0, 0.0),
                           (-1.0, 1.0),
                           5,
                           10)

    # Run simulation
    sim.run(method='rk4', print_progress=True)

    # Show the results
    sim.show()

    sim.plot()

    # Save the results to a file
    sim.save("Electrostatic Simulation.gz")


def test5_1():
    """Full, non-instantaneous electrostatic forces"""
    # Declare the simulation
    sim = Simulation(1.0, 1.0)

    # Load simulation data from a file
    sim.load("Electrostatic Simulation.gz")

    # Show the simulation
    sim.show()


def test6():
    """A moving line of charge with non-instantaneous forces.  All charges in the wire are idealized so that they feel
    no forces."""
    # Work in units where epsilon_0 is 1e-6
    epsilon_0 = 1.0e-6

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

        # Wait for a bit before calling in forces
        if present_time < 5.0:
            return fx, fy

        for i, (xi, yi) in enumerate(zip(x[-1], y[-1])):
            # All charges in the wire are idealized to feel no force.  So, skip calculating any forces for them
            if i in range(100):
                continue

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
            f = -charge[i] * other_charges[valid] / (4 * np.pi * epsilon_0 * distance[valid] ** 2)

            # Break the force into components and add to net force
            fx[i] = np.sum(f * delta_x[valid] / distance[valid])
            fy[i] = np.sum(f * delta_y[valid] / distance[valid])

        # Return the results
        return fx, fy

    # Declare simulation
    sim = Simulation(0.01, 20, forces, c=20.0)

    # Take the frame of the particle that is moving near the wire.
    # This will require calculating the length contraction of the particles
    drift_velocity = 0.5 * sim.c
    drift_gamma = 1 / np.sqrt(1-drift_velocity**2 / sim.c**2)
    primed_span = 700
    positive_charge_span = primed_span / drift_gamma
    negative_charge_span = primed_span * drift_gamma

    # Spawn line of stationary negative charges
    for x_pos in np.linspace(-negative_charge_span, negative_charge_span, 50):
        sim.add_point(x_pos, -20.0, 0.0, 0.0, 1.0, -1.0)

    # Spawn line of moving positive charges
    for x_pos in np.linspace(-positive_charge_span, positive_charge_span, 50):
        sim.add_point(x_pos, -20.0, -drift_velocity, 0.0, 1.0, 1.0)

    # Spawn observational point
    sim.add_point(-20.0, 50.0, -drift_velocity, 0.0, 1.0, 0.0)

    # Spawn test charge near the line
    sim.add_point(10.0, 50.0, 0.0, 0.0, 0.1, 1.0)

    # Run the simulation
    sim.run(print_progress=True)

    # Save this simulation to a file
    sim.save("Line of Charge Simulation.gz")

    # Show the results
    sim.show()

    # Show a plot of the results
    sim.plot()


def test6_1():
    """Same as test6, except loaded from a file to save time"""
    # Declare simulation
    sim = Simulation(0.01, 30)

    # Load simulation data from file
    sim.load("Line of Charge Simulation.gz")

    # Show the results of the simulation
    sim.show()

    # Show a plot of the results
    sim.plot()


# Run main program
if __name__ == "__main__":
    main()
