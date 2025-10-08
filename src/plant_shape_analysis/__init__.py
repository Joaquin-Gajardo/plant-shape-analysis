"""
plant-shape-analysis: Plant shape analysis and reconstruction using implicit neural representations
"""

__version__ = "0.1.0"

# Expose main dataloaders at package level
from plant_shape_analysis.dataloaders.trackplant3D import (
    PlantSequencesDataset,
    LeafSequencesDataset,
)

__all__ = [
    "PlantSequencesDataset",
    "LeafSequencesDataset",
]
