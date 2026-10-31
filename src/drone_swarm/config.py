"""Versioned simulation/deployment contract. No mutable settings inside JITs."""
from dataclasses import dataclass, field, asdict

CONTRACT_VERSION = 'swarm-v3-inertial-tof'
OBSERVATION_VERSION = 3
ACTION_VERSION = 3
REWARD_VERSION = 3

@dataclass(frozen=True)
class SwarmConfig:
    es_population: int = 512
    swarm_size: int = 4
    num_generations: int = 10000
    training_step_budget: int = 0  # 0 = generation-limited; summed active-agent physics steps
    hidden_layers: tuple = (128, 128, 128)
    num_frames: int = 3
    num_lidars: int = 16
    env_freq: int = 50
    physics_freq: int = 500
    device: str = 'auto'
    mutation_rank: int = 1
    alpha_start: float = .001
    alpha_end: float = .0002
    sigma_start: float = .02
    sigma_end: float = .005
    beta1: float = .9
    beta2: float = .999
    eps_adam: float = 1e-8
    grad_clip: float = 10.
    weight_decay: float = .005
    drone_radius: float = .046
    obstacle_radius: float = .10
    nominal_mass: float = .030
    mass_variation: float = .20
    inertia_variation: float = .15
    max_thrust_per_motor: float = .14834
    max_speed: float = .5
    max_vertical_speed: float = .2
    max_yaw_rate: float = .5
    max_tilt_deg: float = 15.
    velocity_kp: float = 4.0
    velocity_ki: float = .5
    vertical_kp: float = 6.0
    vertical_ki: float = 3.0
    actuator_tau: float = .06
    wind_force: float = .003
    lidar_max_range: float = 3.
    lidar_min_range: float = .04
    lidar_fov_deg: float = 27.
    sensor_period_steps: int = 2  # 25 Hz, policy at 50 Hz
    sensor_latency_steps: int = 1
    sensor_noise: float = .01
    sensor_dropout: float = .02
    sensor_bias: float = .01
    # Calibrated IMU measurement model. No camera, optical flow or ground-truth position fixes.
    accel_noise: float = .015  # m/s^2 per 50 Hz sample
    accel_bias: float = .001  # constant per episode, after launch calibration
    gyro_noise: float = .002
    gyro_bias: float = .0001
    attitude_noise: float = .0005  # roll/pitch AHRS error, radians
    launch_position_noise: float = .005
    launch_velocity_noise: float = .002
    odometry_warn: float = .35  # predicted horizontal position uncertainty, metres
    odometry_failure: float = .7
    altitude_timeout: float = 2.
    range_localization: bool = True
    enclosed_room: bool = True
    room_margin: float = 1.5
    range_innovation_gate: float = .18
    control_profile: str = 'guarded'
    avoidance_cones: bool = False
    avoidance_margin: float = .12
    avoidance_gain: float = 2.0
    hold_damping: float = .2
    dynamic_scale: float = 1.0
    preflight_seconds: float = 0.  # measured closed-loop stabilization before mission release
    level_flight: bool = False  # track mission altitude; learned residual remains horizontal
    mission_mode: str = 'slots'  # historical default; individual-transfer recipe uses region
    scenario_family: str = 'legacy'
    region_radius: float = .75
    region_clearance: float = .12  # minimum physical surface clearance for team holding
    effort_weight: float = .05  # command-effort proxy, not electrical energy
    dynamic_speed_scale: float = 1.0
    guidance_kp: float = 1.2
    residual_speed: float = .15  # maximum learned velocity correction (m/s)
    residual_vertical_speed: float = .15  # separately configurable for guarded-profile diagnostics
    residual_yaw_rate: float = .1
    reward_distance_scale: float = .4
    reward_speed_scale: float = .25
    success_time_weight: float = 1.0  # Only after full route + continuous hold succeeds
    training_noise_start: float = .25
    training_noise_stage_increment: float = .75  # nominal noise from route stage
    slot_tolerance: float = .20
    hold_speed: float = .10
    hold_seconds: float = 3.
    waypoint_tolerance: float = .15
    separation: float = .5
    safe_clearance: float = .15
    eval_interval: int = 25
    validation_episodes: int = 40
    promotion_success: float = .90
    promotion_collision: float = .02
    promotion_streak: int = 3
    checkpoint_safety_first: bool = True
    stall_generations: int = 200
    replay_fraction: float = .20
    training_seed: int = 1337
    validation_seed: int = 100000
    test_seed: int = 200000

    def __post_init__(self):
        if not 0 <= self.preflight_seconds <= 30:
            raise ValueError('preflight_seconds must be in [0,30]')
        if self.mission_mode not in ('slots','region') or self.scenario_family not in ('legacy','procedural'):
            raise ValueError('Unknown mission mode or scenario family')
        if not self.slot_tolerance < self.region_radius < .95 or self.region_clearance < 0:
            raise ValueError('Invalid arrival region or separation')
        if self.effort_weight < 0 or self.dynamic_speed_scale <= 0:
            raise ValueError('Invalid effort weight or obstacle speed')
        if self.mission_mode=='region' and self.control_profile!='guarded':
            raise ValueError('Region missions require guarded control')
        if self.control_profile not in ('legacy','guarded'):
            raise ValueError('Unknown control profile')
        if not 0 <= self.dynamic_scale <= 1:
            raise ValueError('dynamic_scale must be in [0,1]')
        if min(self.avoidance_margin,self.avoidance_gain) <= 0 or self.hold_damping < 0:
            raise ValueError('Invalid guarded control settings')
        if self.device not in ('auto', 'cpu', 'gpu'):
            raise ValueError('device must be auto, cpu or gpu')
        if self.es_population < 2 or self.es_population % 2:
            raise ValueError('es_population must be positive and even')
        if self.swarm_size < 1 or self.num_lidars not in self.supported_sensor_counts:
            raise ValueError(f'swarm_size >= 1; supported sensor profiles: {self.supported_sensor_counts}')
        if self.physics_freq % self.env_freq:
            raise ValueError('physics_freq must be divisible by env_freq')
        if min(self.sensor_latency_steps,self.sensor_period_steps) < 1:
            raise ValueError('sensor latency/period must be >= 1')

        if self.velocity_kp <= 0 or self.residual_vertical_speed < 0:
            raise ValueError('Invalid controller gain or residual bound')
        if self.hold_seconds <= 0:
            raise ValueError('hold_seconds must be positive')
        if not 0 < self.odometry_warn < self.odometry_failure:
            raise ValueError('Require 0 < odometry_warn < odometry_failure')
        if self.num_generations < 1 or self.validation_episodes < 1 or self.eval_interval < 1:
            raise ValueError('Generation/evaluation counts must be positive')

    @property
    def supported_sensor_counts(self): return (10, 16)
    @property
    def dt(self): return 1. / self.env_freq
    @property
    def arrival_tolerance(self): return self.region_radius if self.mission_mode=='region' else self.slot_tolerance
    @property
    def num_drones(self): return self.es_population * self.swarm_size
    @property
    def base_state_size(self): return 24
    @property
    def frame_size(self): return self.base_state_size + self.num_lidars * 3
    @property
    def input_size(self): return self.frame_size * self.num_frames
    @property
    def layers(self): return (self.input_size, *self.hidden_layers, 4)
    def to_dict(self): return asdict(self)

config = SwarmConfig()
