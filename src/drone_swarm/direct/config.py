"""Separate observation/action contract; never reinterpret residual checkpoints."""
from dataclasses import dataclass
from drone_swarm.config import SwarmConfig

CONTRACT = 'droneling-v4-direct-motors-sim-navigation'

@dataclass(frozen=True)
class DirectConfig(SwarmConfig):
    es_population: int = 8192
    swarm_size: int = 1
    hidden_layers: tuple = (64,64)
    num_lidars: int = 26
    num_frames: int = 4
    env_freq: int = 100
    sensor_period_steps: int = 4
    sensor_latency_steps: int = 2
    range_localization: bool = False
    mission_mode: str = 'region'
    alpha_start: float = .0003
    alpha_end: float = .00005
    sigma_start: float = .01
    sigma_end: float = .005
    max_speed: float = 2.  # observation scale only; no velocity command limiter
    batch_worlds: int = 256
    episode_seconds: float = 12.
    motor_tau: float = .025
    navigation_position_noise: float = .01
    navigation_velocity_noise: float = .02
    initial_tilt_degrees: float = 5.
    direct_curriculum_version: int = 1  # preserve historical checkpoints; fresh CLI runs use v2
    direct_reward_version: int = 2
    direct_stall_generations: int = 200
    direct_observation_version: int = 1
    direct_curriculum_generations: int = 0
    direct_schedule_generations: int = 0  # 0 preserves historical total-budget schedule
    direct_validation_groups: int = 1
    direct_encounter_generations: int = 0  # v4 training ramp; evaluation always full difficulty

    def __post_init__(self):
        super().__post_init__()
        if self.direct_curriculum_version not in (1,2,3,4):
            raise ValueError("Unsupported direct curriculum version")
        if self.direct_reward_version not in (1,2,3,4,5) or self.direct_stall_generations<0:
            raise ValueError('Unsupported direct reward version or stall budget')
        if self.direct_encounter_generations<0 or (self.direct_encounter_generations and self.direct_curriculum_version!=4):
            raise ValueError('Encounter ramp requires curriculum v4 and a nonnegative duration')
        if self.direct_schedule_generations<0 or self.direct_validation_groups<1:
            raise ValueError('Invalid learning schedule or validation group count')
        if self.direct_observation_version not in (1,2) or self.direct_curriculum_generations<0:
            raise ValueError('Unsupported direct observation or curriculum')
        if self.batch_worlds<1 or self.episode_seconds<=0 or self.motor_tau<=0:
            raise ValueError('Positive batch size, horizon and motor response time required')
        if min(self.navigation_position_noise,self.navigation_velocity_noise,self.initial_tilt_degrees)<0:
            raise ValueError('Noise and initial tilt must be nonnegative')
        if self.preflight_seconds or self.level_flight or self.range_localization:
            raise ValueError('Direct motors do not use preflight, level-flight or mapped localization')

    @property
    def supported_sensor_counts(self): return (10,16,26)
    @property
    def base_state_size(self): return 16
    @property
    def frame_size(self):
        return self.base_state_size+self.num_lidars*(3 if self.direct_observation_version==1 else 1)
