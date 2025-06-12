"""
STAR Body System - Multi-layer hierarchical body representation
"""

from .core.star_interface import STARInterface
from .core.body_definitions import BodyDefinitions
from .layers.sphere_layer import SphereLayer
from .layers.capsule_layer import CapsuleLayer
from .layers.simple_capsule_layer import SimpleCapsuleLayer

__version__ = "0.1.0"
__all__ = [
    "STARInterface",
    "BodyDefinitions", 
    "SphereLayer",
    "CapsuleLayer",
    "SimpleCapsuleLayer"
]