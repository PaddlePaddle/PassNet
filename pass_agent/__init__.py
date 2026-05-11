"""
PassAgent - R2E-Gym extension for PassNet compiler optimization tasks.

This package provides the PassNet-specific runtime, tools, and configurations
for training agents on compiler optimization tasks using R2E-Gym.
"""

__version__ = "0.1.0"

from pass_agent.runtime.passnet_docker import PassNetDocker

__all__ = ["PassNetDocker"]
