# pollution_sim.py
import os
import json
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
@dataclass
class SimConfig:
    """
    Data class for storing simulation hyperparameters and environment settings.
    """
    grid_x: int = 100
    grid_y: int = 100
    cell_size_m: float = 50.0
    burn_in_steps: int = 24
    sampling_steps: int = 48
    wind_vector: Tuple[float, float] = (1.5, -0.8)
    diffusion_sigma: float = 1.2
    decay_rate: float = 0.05
    bg_pollution: float = 10.0
    bg_noise_std: float = 2.0
    num_main_routes: int = 3
    num_minor_routes: int = 5
    num_mobile_sources: int = 50
    num_static_sources: int = 10
    num_sensors: int = 200 
    num_layouts: int = 5

# ==========================================
# 2. PHYSICAL ENVIRONMENT
# ==========================================
class Environment:
    """
    Simulates the physical dynamics of pollution dispersion including wind advection, 
    diffusion, and atmospheric decay.
    """
    def __init__(self, config: SimConfig):
        self.config = config
        self.grid = np.full((config.grid_x, config.grid_y), config.bg_pollution)
        self.current_wind = list(config.wind_vector)
        
    def step(self, emissions: np.ndarray):
        """
        Executes a single simulation step to update the pollution grid state.
        """
        # Update wind vector using a random walk process
        self.current_wind[0] += np.random.normal(0, 0.1)
        self.current_wind[1] += np.random.normal(0, 0.1)
        self.current_wind = [np.clip(w, -3.0, 3.0) for w in self.current_wind]
        
        # Apply emissions and calculate advection (wind shift)
        self.grid += emissions
        self.grid = ndimage.shift(self.grid, self.current_wind, mode='nearest')
        
        # Apply Gaussian diffusion and atmospheric decay
        self.grid = ndimage.gaussian_filter(self.grid, sigma=self.config.diffusion_sigma)
        self.grid *= (1.0 - self.config.decay_rate)
        
        # Add background white noise and ensure non-negative concentration
        noise = np.random.normal(0, self.config.bg_noise_std, self.grid.shape)
        self.grid = np.clip(self.grid + noise, 0, None)

# ==========================================
# 3. EMISSION SOURCES
# ==========================================
class SourceManager:
    """
    Manages static pollution sources and mobile agents moving along predefined routes.
    """
    def __init__(self, config: SimConfig):
        self.config = config
        self.routes = self._generate_routes()
        self.static_sources = self._generate_static()
        self.mobile_sources = self._init_mobile()

    def _generate_routes(self) -> List[np.ndarray]:
        """
        Generates linear paths representing transportation infrastructure.
        """
        routes = []
        for _ in range(self.config.num_main_routes + self.config.num_minor_routes):
            p1 = (np.random.randint(0, self.config.grid_x), 0)
            p2 = (np.random.randint(0, self.config.grid_x), self.config.grid_y - 1)
            length = max(self.config.grid_x, self.config.grid_y) * 2
            x = np.linspace(p1[0], p2[0], length).astype(int)
            y = np.linspace(p1[1], p2[1], length).astype(int)
            routes.append(np.vstack((x, y)).T)
        return routes

    def _generate_static(self) -> np.ndarray:
        """
        Generates coordinates and intensities for fixed industrial emission points.
        """
        x = np.random.randint(5, self.config.grid_x - 5, self.config.num_static_sources)
        y = np.random.randint(5, self.config.grid_y - 5, self.config.num_static_sources)
        intensity = np.random.uniform(50, 150, self.config.num_static_sources)
        return np.column_stack((x, y, intensity))

    def _init_mobile(self) -> List[Dict]:
        """
        Initializes state parameters for mobile pollution sources.
        """
        mobile = []
        for _ in range(self.config.num_mobile_sources):
            r_idx = np.random.randint(0, len(self.routes))
            mobile.append({
                'route_idx': r_idx,
                'pos_idx': np.random.randint(0, len(self.routes[r_idx])),
                'speed': np.random.randint(1, 4),
                'intensity': np.random.uniform(5, 15)
            })
        return mobile

    def get_emissions_grid(self) -> np.ndarray:
        """
        Computes the spatial distribution of emissions for the current time step.
        """
        emissions = np.zeros((self.config.grid_x, self.config.grid_y))
        for sx, sy, intensity in self.static_sources:
            emissions[int(sx), int(sy)] += intensity
        for mob in self.mobile_sources:
            route = self.routes[mob['route_idx']]
            mob['pos_idx'] = (mob['pos_idx'] + mob['speed']) % len(route)
            x, y = route[mob['pos_idx']]
            emissions[x, y] += mob['intensity']
        return emissions

# ==========================================
# 4. SENSORS
# ==========================================
class SensorManager:
    """
    Simulates the deployment of sensor networks and data acquisition processes.
    """
    def __init__(self, config: SimConfig):
        self.config = config
        self.layouts = self._generate_layouts()
        self.sensor_drift = [np.random.normal(0, 2.0, config.num_sensors) for _ in range(config.num_layouts)]

    def _generate_layouts(self) -> List[np.ndarray]:
        """
        Generates stochastic spatial configurations for sensor placements.
        """
        layouts = []
        for _ in range(self.config.num_layouts):
            x = np.random.randint(0, self.config.grid_x, self.config.num_sensors)
            y = np.random.randint(0, self.config.grid_y, self.config.num_sensors)
            layouts.append(np.column_stack((x, y)))
        return layouts

    def sample(self, grid: np.ndarray, packet_loss_prob=0.1) -> List[np.ndarray]:
        """
        Extracts noisy sensor observations from the ground truth pollution grid.
        """
        readings = []
        for i, layout in enumerate(self.layouts):
            vals = grid[layout[:, 0], layout[:, 1]]
            # Apply measurement noise and systematic sensor drift
            vals_noisy = vals * np.random.normal(1.0, 0.05, len(vals)) + self.sensor_drift[i]
            vals_noisy = np.clip(vals_noisy, 0, None)
            # Simulate data transmission failure (packet loss)
            mask = np.random.rand(len(vals)) > packet_loss_prob
            readings.append(vals_noisy * mask)
        return readings

# ==========================================
# 5. SIMULATOR AND DATASET GENERATOR
# ==========================================
class Simulator:
    """
    Orchestrates the environment and agent interactions to perform a full simulation run.
    """
    def __init__(self, config: SimConfig):
        self.config = config
        self.env = Environment(config)
        self.sources = SourceManager(config)
        self.sensors = SensorManager(config)
        
    def run(self) -> Tuple[np.ndarray, List[np.ndarray], List[float]]:
        """
        Executes the simulation for the specified number of steps, including a burn-in period.
        """
        total_steps = self.config.burn_in_steps + self.config.sampling_steps
        gt_history = []
        sensor_history = [[] for _ in range(self.config.num_layouts)]
        actual_winds = []
        
        for step in range(total_steps):
            self.env.step(self.sources.get_emissions_grid())
            actual_winds.append(list(self.env.current_wind))
            
            # Record data only after the initial burn-in phase
            if step >= self.config.burn_in_steps:
                gt_history.append(self.env.grid.copy())
                readings = self.sensors.sample(self.env.grid)
                for i, r in enumerate(readings):
                    sensor_history[i].append(r)
                    
        # Calculate the mean wind vector for metadata reporting
        avg_wind = np.mean(actual_winds[self.config.burn_in_steps:], axis=0).tolist()
        return np.array(gt_history), [np.array(sh) for sh in sensor_history], avg_wind

def generate_dataset(base_config: SimConfig, num_scenarios: int, output_dir: str):
    """
    Generates a comprehensive dataset consisting of multiple simulation scenarios.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Commencing generation of {num_scenarios} scenarios with {base_config.num_sensors} sensors.")
    
    for i in tqdm(range(num_scenarios)):
        cfg = SimConfig(
            grid_x=base_config.grid_x, grid_y=base_config.grid_y,
            sampling_steps=base_config.sampling_steps,
            burn_in_steps=base_config.burn_in_steps,
            num_sensors=base_config.num_sensors,
            num_layouts=base_config.num_layouts,
            wind_vector=(np.random.uniform(-2, 2), np.random.uniform(-2, 2)),
            decay_rate=np.random.uniform(0.02, 0.08)
        )
        sim = Simulator(cfg)
        gt, sensors, avg_w = sim.run()
        
        # Save compressed ground truth and sensor observation data
        s_path = os.path.join(output_dir, f"scenario_{i:04d}")
        os.makedirs(s_path, exist_ok=True)
        np.savez_compressed(os.path.join(s_path, "ground_truth.npz"), data=gt)
        for j, data in enumerate(sensors):
            np.savez_compressed(os.path.join(s_path, f"sensor_layout_{j}.npz"), 
                                readings=data, coordinates=sim.sensors.layouts[j])
        
        # Persist simulation metadata and calculated average wind vector
        meta = asdict(cfg)
        meta['avg_wind'] = avg_w 
        with open(os.path.join(s_path, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)

# ==========================================
# 6. VISUALIZATION
# ==========================================
class Visualizer:
    """
    Provides static methods for visualizing spatial infrastructure and pollution maps.
    """
    @staticmethod
    def plot_infrastructure(sim: Simulator):
        """
        Renders the map of routes, static sources, and sensor deployments.
        """
        plt.figure(figsize=(5, 5))
        for r in sim.sources.routes: plt.plot(r[:, 1], r[:, 0], color='gray', alpha=0.3)
        st = sim.sources.static_sources
        plt.scatter(st[:, 1], st[:, 0], c='red', marker='^', s=100)
        lay = sim.sensors.layouts[0]
        plt.scatter(lay[:, 1], lay[:, 0], c='blue', marker='.', alpha=0.5)
        plt.title("Infrastructure Map")
        plt.show()

    @staticmethod
    def plot_timestep(gt, step=-1):
        """
        Displays the spatial distribution of pollution for a specific time step.
        """
        plt.imshow(gt[step], cmap='inferno', origin='lower')
        plt.colorbar()
        plt.title(f"Pollution Map (Step {step})")
        plt.show()