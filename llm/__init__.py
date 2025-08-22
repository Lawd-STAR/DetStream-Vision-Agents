"""
Models Package

This package provides AI model interfaces and implementations for Stream Agents.
"""

from .model import Model
from .openai import OpenAILLM

__all__ = ["Model", "OpenAILLM"]

__version__ = "0.1.0"
