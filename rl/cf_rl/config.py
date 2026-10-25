"""Crazyflie 2.1 Brushless parameters and the approach-and-hover task.

gym-pybullet-drones 2.2.0 ships a Crazyflie 2.x X-configuration URDF
(``cf2x.urdf``), not a brushless airframe. ``DroneConfig`` is the override
applied on top of that URDF. Geometry (arm length, mesh, inertia tensor
shape) stays the library model; mass and thrust do not.
"""

from dataclasses import dataclass, field


# Values read from gym-pybullet-drones 2.2.0 assets/cf2x.urdf.
# Commit f83fbdd4ee1c07e1afa52694963f3a9a889a0f56.
@dataclass(frozen=True)
class LibraryCF2X:
    """Default model this project starts from, before the brushless override."""

    mass_kg: float = 0.027
    arm_m: float = 0.0397
    kf: float = 3.16e-10
    km: float = 7.94e-12
    thrust2weight: float = 2.25
    prop_radius_m: float = 2.31348e-2
    g: float = 9.8  # BaseAviary uses 9.8, not 9.81

    @property
    def max_thrust_n(self) -> float:
        return self.thrust2weight * self.mass_kg * self.g

    @property
    def max_thrust_g(self) -> float:
        return self.max_thrust_n / self.g * 1000.0


LIBRARY_CF2X = LibraryCF2X()


@dataclass(frozen=True)
class DroneConfig:
    """Physical parameters of the vehicle the policy is meant to fly.

    Bitcraze Crazyflie 2.1 Brushless datasheet (rev 3):

    * takeoff weight 34 g with legs, 37 g with guards
    * 4 x 08028-10000KV motors, up to 30 g thrust each (120 g total)
    * motor-to-motor diagonal 100 mm, so the arm is 50 mm

    The default mass is 35 g, between the legs and guards figures.
    ``with_legs`` and ``with_guards`` are the datasheet endpoints.

    Thrust is applied by scaling the URDF motor constants ``kf`` and ``km``
    so the library's maximum RPM (just inside the DSL PID PWM map) produces
    ``max_thrust_g`` instead of the Crazyflie 2.x figure of about 61 g.
    The URDF arm stays 39.7 mm. This is not an identified brushless model;
    crazyflow's ``cf21B_500`` is the one that is. See ``ROADMAP.md``.
    """

    mass_kg: float = 0.035
    max_thrust_g: float = 120.0
    g: float = 9.8

    def __post_init__(self) -> None:
        if self.mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        if self.max_thrust_g <= 0:
            raise ValueError("max_thrust_g must be positive")
        if self.max_thrust_kg <= self.mass_kg:
            raise ValueError(
                f"max thrust {self.max_thrust_g:.0f} g does not exceed "
                f"mass {self.mass_kg * 1000:.0f} g, so the drone cannot hover"
            )

    @property
    def max_thrust_kg(self) -> float:
        return self.max_thrust_g / 1000.0

    @property
    def max_thrust_n(self) -> float:
        return self.max_thrust_kg * self.g

    @property
    def thrust_to_weight(self) -> float:
        return self.max_thrust_kg / self.mass_kg

    @property
    def kf_scale(self) -> float:
        """Multiply the URDF ``kf`` and ``km`` by this to hit ``max_thrust_g``."""
        return self.max_thrust_n / LIBRARY_CF2X.max_thrust_n

    def with_legs(self) -> "DroneConfig":
        return DroneConfig(mass_kg=0.034, max_thrust_g=self.max_thrust_g, g=self.g)

    def with_guards(self) -> "DroneConfig":
        return DroneConfig(mass_kg=0.037, max_thrust_g=self.max_thrust_g, g=self.g)

    def summary(self) -> str:
        lib_g = LIBRARY_CF2X.max_thrust_g
        return (
            f"CF2.1 Brushless override: mass {self.mass_kg * 1000:.1f} g "
            f"(library CF2X {LIBRARY_CF2X.mass_kg * 1000:.0f} g), "
            f"max thrust {self.max_thrust_g:.0f} g (library {lib_g:.0f} g), "
            f"thrust/weight {self.thrust_to_weight:.2f} "
            f"(library {LIBRARY_CF2X.thrust2weight:.2f}), "
            f"KF scale x{self.kf_scale:.3f}. "
            f"Arm and mesh stay the CF2X URDF "
            f"({LIBRARY_CF2X.arm_m * 1000:.1f} mm, not the brushless 50 mm)."
        )


@dataclass(frozen=True)
class HoverLimits:
    """Clip and slew applied to every setpoint, in the sim and intended on the radio.

    Rates are the maximum change of the command, not of the vehicle.
    ``max_dv`` is horizontal acceleration of the *command* in m/s^2.
    """

    vx: float = 0.40          # m/s, heading frame
    vy: float = 0.40          # m/s, heading frame
    yaw_rate: float = 1.0     # rad/s
    z_min: float = 0.15       # m
    z_max: float = 1.20       # m
    max_dv: float = 2.0       # m/s^2
    max_dyaw: float = 4.0     # rad/s^2
    max_dz: float = 0.50      # m/s, height-command slew


@dataclass(frozen=True)
class TaskConfig:
    """Approach a colored target and hover at a standoff point in front of it."""

    standoff_m: float = 0.45
    target_radius_m: float = 0.09
    episode_seconds: float = 8.0
    arena_xy: float = 1.60
    ceiling_m: float = 1.30
    floor_m: float = 0.05
    max_tilt_rad: float = 1.0
    hold_radius_m: float = 0.15
    hold_speed_m_s: float = 0.20
    hold_bonus: float = 1.0
    crash_penalty: float = 5.0
    action_penalty: float = 0.05
    tilt_penalty: float = 0.10
    image_size: int = 64
    # Stand-in for the AI-deck camera. Not a calibrated Himax model.
    camera_fov_deg: float = 90.0
    limits: HoverLimits = field(default_factory=HoverLimits)
    target_x: tuple[float, float] = (0.95, 1.40)
    target_y: tuple[float, float] = (-0.45, 0.45)
    target_z: tuple[float, float] = (0.40, 0.85)
    start_x: tuple[float, float] = (-0.15, 0.25)
    start_y: tuple[float, float] = (-0.30, 0.30)
    start_z: tuple[float, float] = (0.35, 0.70)
    start_yaw: tuple[float, float] = (-0.50, 0.50)
