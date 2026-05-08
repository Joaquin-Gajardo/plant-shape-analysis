"""
Tracking subpackage. Each method lives in its own subdirectory.

Currently available:
  trackplant3d — TrackPlant3D (Magistri et al. 2023)
"""

from plant_shape_analysis.tracking.trackplant3d.pipeline import run_tracking_pipeline

__all__ = ["run_tracking_pipeline"]
