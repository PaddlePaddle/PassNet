"""
AI4C Agent - R2E-Gym extension for AI4C compiler optimization tasks.

This package provides the AI4C-specific runtime, tools, and configurations
for training agents on compiler optimization tasks using R2E-Gym.
"""

__version__ = "0.1.0"

from pass_agent.runtime.ai4c_docker import AI4CDocker

__all__ = ["AI4CDocker"]
