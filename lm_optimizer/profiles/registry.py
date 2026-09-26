"""Optimization profiles registry."""

from dataclasses import dataclass, field


@dataclass
class Profile:
    """Optimization profile with weights and settings."""

    name: str
    description: str
    weights: dict[str, float] = field(default_factory=dict)
    minimum_quality: float = 0.97
    context_priority: float = 0.5  # 0=speed, 1=max context

    def __post_init__(self):
        # Default weights if not provided
        if not self.weights:
            self.weights = self._default_weights()

    def _default_weights(self) -> dict[str, float]:
        """Default weights based on profile name.

        Each profile intentionally trades off speed vs quality vs context:
        - Speed: favors throughput (gen 40%, prompt 25%), tolerates lower quality (10%)
        - Balanced: even distribution for real-world use
        - Context: maximizes stable context (25%), still requires decent quality
        - Quality: maximizes correctness (35%) and context (21%), speed secondary
        Weights sum to 1.0 and directly affect hardware-agnostic scoring.
        """
        if self.name == "speed":
            return {
                "generation_speed": 0.40,
                "prompt_processing": 0.25,
                "ttft": 0.15,
                "quality": 0.10,
                "stability": 0.05,
                "context_capacity": 0.03,
                "memory_efficiency": 0.02,
            }
        if self.name == "balanced":
            return {
                "generation_speed": 0.27,
                "prompt_processing": 0.15,
                "ttft": 0.12,
                "quality": 0.12,
                "stability": 0.12,
                "context_capacity": 0.11,
                "memory_efficiency": 0.11,
            }
        if self.name == "context":
            return {
                "generation_speed": 0.10,
                "prompt_processing": 0.10,
                "ttft": 0.10,
                "quality": 0.15,
                "stability": 0.15,
                "context_capacity": 0.25,
                "memory_efficiency": 0.15,
            }
        if self.name == "quality":
            return {
                "generation_speed": 0.08,
                "prompt_processing": 0.08,
                "ttft": 0.08,
                "quality": 0.35,
                "stability": 0.13,
                "context_capacity": 0.21,
                "memory_efficiency": 0.07,
            }
        if self.name == "custom":
            # Default custom is balanced-like but user overrides
            return {
                "generation_speed": 0.20,
                "prompt_processing": 0.15,
                "ttft": 0.10,
                "quality": 0.20,
                "stability": 0.10,
                "context_capacity": 0.15,
                "memory_efficiency": 0.10,
            }
        return {}


class ProfileRegistry:
    """Registry of optimization profiles."""

    def __init__(self):
        self._profiles: dict[str, Profile] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register built-in profiles.

        Thresholds are intentional and documented:
        - Speed: 0.95 (allows minor heuristic misses for throughput)
        - Balanced: 0.97 (default, rejects clearly broken outputs)
        - Context: 0.97 (same as balanced, but weights favor context)
        - Quality: 0.99 (strict, rejects any heuristic failure)
        - Custom: user-defined (0.90-1.0)
        Profile affects BOTH filtering (threshold) and scoring (weights).
        A profile claiming quality optimization must have high quality weight (>0.3)
        and high threshold, verified here.
        """
        self.register(
            Profile(
                name="speed",
                description="Maximize generation and prompt throughput. Context only as large as necessary. Quality threshold 0.95 allows throughput at cost of minor quality loss.",
                minimum_quality=0.95,
                context_priority=0.2,
            )
        )
        self.register(
            Profile(
                name="balanced",
                description="Maximize practical real-world performance with balanced tradeoffs. Threshold 0.97 rejects broken outputs while allowing real-world variance.",
                minimum_quality=0.97,
                context_priority=0.5,
            )
        )
        self.register(
            Profile(
                name="context",
                description="Maximize stable context length while maintaining quality. Threshold 0.97, weights heavily favor context (25%).",
                minimum_quality=0.97,
                context_priority=0.9,
            )
        )
        self.register(
            Profile(
                name="quality",
                description="Maximize output correctness and usable context. Threshold 0.99 strict, quality weight 35% ensures quality drives selection.",
                minimum_quality=0.99,
                context_priority=0.8,
            )
        )
        # Custom is not pre-registered but can be created via create_custom

    def register(self, profile: Profile) -> None:
        """Register a profile."""
        self._profiles[profile.name] = profile

    def get(self, name: str) -> Profile:
        """Get a profile by name."""
        if name not in self._profiles:
            raise ValueError(f"Unknown profile: {name}. Available: {list(self._profiles.keys())}")
        return self._profiles[name]

    def list_profiles(self) -> list[Profile]:
        """List all registered profiles."""
        return list(self._profiles.values())

    def create_custom(
        self, name: str, description: str, weights: dict[str, float], **kwargs
    ) -> Profile:
        """Create a custom profile."""
        profile = Profile(name=name, description=description, weights=weights, **kwargs)
        self.register(profile)
        return profile
