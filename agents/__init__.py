"""
Stream Agents Package

This package provides AI agent functionality for Stream video calls.
It uses the stream-py package for STT and TTS services.
"""

from .agents import Agent, Tool, PreProcessor, STT, TTS, STS, TurnDetection

# Import Model from the models package
try:
    from models.model import Model
except ImportError:
    # Fallback to local Model protocol if models package not available
    from .agents import Model

__all__ = [
    "Agent",
    "Tool",
    "PreProcessor",
    "Model",
    "STT",
    "TTS",
    "STS",
    "TurnDetection",
]

__version__ = "0.1.0"
