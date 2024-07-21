import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# Constants
ROOM_DIMENSION = [2.5, 9, 2.5]  # 3D space dimensions (width, depth, height) in meters
NUM_PARTICLES = 1000
SIMULATION_DURATION = 300  # seconds
UPDATE_INTERVAL = 0.01  # seconds for particle updates
TAG_STEP_TIME = 0.01  # seconds for tag movements
EXPECTED_RSSI = -50
RSSI_STDDEV = 5
MIN_ACCELERATION = -3  # m/s^2
MAX_ACCELERATION = 3  # m/s^2
DISTANCE_NOISE = 0.2  # meters, +/- 20 cm of noise

# Derived constants
UPDATES_PER_SECOND = int(1 / UPDATE_INTERVAL)
TAG_STEP_UPDATES = int(TAG_STEP_TIME / UPDATE_INTERVAL)
TOTAL_UPDATES = SIMULATION_DURATION * UPDATES_PER_SECOND

class Anchor:
    def __init__(self, x, y, z, name):
        self.position = np.array([x, y, z])
        self.name = name

class Tag:
    def __init__(self, num_particles, room_dimension, name):
        self.num_particles = num_particles
        self.room_dimension = room_dimension
        self.name = name
        self.actual_position = np.random.rand(3) * self.room_dimension
        self.direction = self._random_unit_vector()
        self.velocity = self._initial_random_velocity()
        self.acceleration = self._random_acceleration()

    def _random_unit_vector(self):
        vector = np.random.randn(3)
        return vector / np.linalg.norm(vector)

    def _random_acceleration(self):
        return np.random.uniform(MIN_ACCELERATION, MAX_ACCELERATION, size=3)

    def _initial_random_velocity(self):
        speed = np.random.uniform(0.1, 0.5)
        return self.direction * speed

    def update_acceleration(self):
        if np.random.random() < 0.1:  # 10% chance to change acceleration each update
            self.acceleration = self._random_acceleration()

    def move(self):
        self.update_acceleration()

        # Update the velocity based on acceleration
        self.velocity += self.acceleration * TAG_STEP_TIME

        # Update the position based on velocity
        new_position = self.actual_position + self.velocity * TAG_STEP_TIME

        # Check if the new position is within the room boundaries
        for i in range(3):
            if new_position[i] < 0 or new_position[i] > self.room_dimension[i]:
                # If we hit a wall, reverse the direction only in the affected dimension
                self.direction[i] *= -1
                self.velocity[i] *= -1
                new_position[i] = np.clip(new_position[i], 0, self.room_dimension[i])

        # Update the position
        self.actual_position = new_position

        return True

class Simulation:
    def __init__(self, room_dimension, num_particles, anchors, tag_names, expected_rssi, rssi_stddev):
        self.room_dimension = room_dimension
        self.num_particles = num_particles
        self.anchors = anchors  # List of Anchor objects
        self.tags = [Tag(num_particles, room_dimension, name) for name in tag_names]
        self.distances_rssi_dict = {anchor.name: {tag.name: (0, 0) for tag in self.tags} for anchor in self.anchors}

       # Define consistent colors for each tag
        self.colors = ['tab:blue', 'tab:orange', 'tab:green']

        # Initialize the plot
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(0, self.room_dimension[0])
        self.ax.set_ylim(0, self.room_dimension[1])
        self.ax.set_zlim(0, self.room_dimension[2])
        self.ax.set_xlabel('Width (meters)')
        self.ax.set_ylabel('Depth (meters)')
        self.ax.set_zlabel('Height (meters)')
        self.ax.set_title('Indoor Positioning Simulation')
        self.anchors_scatter = self.ax.scatter([anchor.position[0] for anchor in anchors], [anchor.position[1] for anchor in anchors], [anchor.position[2] for anchor in anchors], marker='x', color='blue', label='Anchors')
        self.estimated_position_scatters = [self.ax.scatter([], [], [], s=100, marker='^', color=self.colors[i], label=f'Estimated {tag.name}') for i, tag in enumerate(self.tags)]
        self.actual_position_scatters = [self.ax.scatter([], [], [], s=100, marker='o', color=self.colors[i], label=f'Actual {tag.name}') for i, tag in enumerate(self.tags)]
        self.ax.legend()
        self.ax.grid(True)
        self.first_run = True
        self.distance_accumulations = {tag.name: [] for tag in self.tags}
        self.rssi_accumulations = {tag.name: [] for tag in self.tags}

    def update_distances_rssi(self):
        for anchor in self.anchors:
            for tag in self.tags:
                distance = np.linalg.norm(anchor.position - tag.actual_position)
                distance += np.random.uniform(-DISTANCE_NOISE, DISTANCE_NOISE)  # Adding random noise
                rssi = np.random.normal(-50, 5)  # Example RSSI value with some noise
                self.distances_rssi_dict[anchor.name][tag.name] = (distance, rssi)

    def multilateration(self, tag_name):
        tag_distances = np.array([self.distances_rssi_dict[anchor.name][tag_name][0] for anchor in self.anchors])

        def objective(x):
            return np.sum((np.linalg.norm(np.array([anchor.position for anchor in self.anchors]) - x, axis=1) - tag_distances)**2)

        result = minimize(objective, np.mean([anchor.position for anchor in self.anchors], axis=0), method='Nelder-Mead')
        return result.x

    def update(self, frame):
        updated_artists = []

        self.update_distances_rssi()

        for tag in self.tags:
            tag_name = tag.name
            self.distance_accumulations[tag_name].append([self.distances_rssi_dict[anchor.name][tag_name][0] for anchor in self.anchors])
            self.rssi_accumulations[tag_name].append([self.distances_rssi_dict[anchor.name][tag_name][1] for anchor in self.anchors])

        if (frame + 1) % TAG_STEP_UPDATES == 0:
            for tag in self.tags:
                tag_name = tag.name
                estimated_position = self.multilateration(tag_name)

                scatter_idx = self.tags.index(tag)
                est_scatter, act_scatter = self.estimated_position_scatters[scatter_idx], self.actual_position_scatters[scatter_idx]
                est_scatter._offsets3d = ([estimated_position[0]], [estimated_position[1]], [estimated_position[2]])
                act_scatter._offsets3d = ([tag.actual_position[0]], [tag.actual_position[1]], [tag.actual_position[2]])

                # Move the actual tag
                tag.move()
                
                updated_artists.extend([est_scatter, act_scatter])

                # Clear accumulations for the next TAG_STEP_TIME interval
                self.distance_accumulations[tag_name] = []
                self.rssi_accumulations[tag_name] = []

        return updated_artists

    def run(self):
        ani = FuncAnimation(self.fig, self.update, frames=TOTAL_UPDATES, interval=UPDATE_INTERVAL * 1000, blit=False)
        plt.show()

# Anchors in 3D positions (in meters)
anchors = [
    Anchor(0, 0, 0, 'Anchor 1'), Anchor(ROOM_DIMENSION[0], 0, 0, 'Anchor 2'),
    Anchor(0, ROOM_DIMENSION[1], 0, 'Anchor 3'), Anchor(ROOM_DIMENSION[0], ROOM_DIMENSION[1], 0, 'Anchor 4'),
    Anchor(0, 0, ROOM_DIMENSION[2], 'Anchor 5'), Anchor(ROOM_DIMENSION[0], 0, ROOM_DIMENSION[2], 'Anchor 6'),
    Anchor(0, ROOM_DIMENSION[1], ROOM_DIMENSION[2], 'Anchor 7'), Anchor(ROOM_DIMENSION[0], ROOM_DIMENSION[1], ROOM_DIMENSION[2], 'Anchor 8')
]

# Tags
tag_names = ['Tag 1', 'Tag 2', 'Tag 3']

# Initial simulated distances and RSSI values from anchors to tags
distances_rssi_dict = {}

# Run the simulation
simulation = Simulation(ROOM_DIMENSION, NUM_PARTICLES, anchors, tag_names, EXPECTED_RSSI, RSSI_STDDEV)
simulation.run()