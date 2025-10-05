#!/usr/bin/env python3
"""
Test script for plant sequence alignment and visualization
"""

from pathlib import Path

from plant_shape_analysis.dataloaders.trackplant3D import PlantSequencesDataset
from plant_shape_analysis.vis.plot_functions import visualize_plant_sequence


def main():
    dataset_path = Path("data/TrackPlant3D/versions/v1")
    sequence_name = "maize_control_plant2"
    offscreen = True  # Set to True to run visualization in offscreen mode

    # Test 1: PlantSequencesDataset without alignment
    print("=" * 60)
    print("Test 1: Loading plant dataset without alignment")
    print("=" * 60)
    plant_dataset_no_align = PlantSequencesDataset(dataset_path, use_ply=True)
    print(f"Number of sequences: {len(plant_dataset_no_align)}")
    print(f"Sequence names: {plant_dataset_no_align.get_sequence_names()[:3]}")

    # Get a sample without alignment
    sample_no_align = plant_dataset_no_align.get_timeseries_by_sequence_name(
        sequence_name
    )
    print(f"Sample structure: {sample_no_align.keys()}")
    print(f"Is aligned: {sample_no_align['is_aligned']}")
    print()

    # Test 2: PlantSequencesDataset with alignment
    print("=" * 60)
    print("Test 2: Loading plant dataset WITH PCA alignment")
    print("=" * 60)
    plant_dataset_aligned = PlantSequencesDataset(
        dataset_path,
        save_transformations=False,
        use_ply=True,
        alignment_method="icp",
    )
    print(f"Number of sequences: {len(plant_dataset_aligned)}")

    # Get a sample with alignment
    sample_aligned = plant_dataset_aligned.get_timeseries_by_sequence_name(
        sequence_name
    )
    print(f"Sample structure: {sample_aligned.keys()}")
    print(f"Is aligned: {sample_aligned['is_aligned']}")
    print(f"Number of transformations: {len(sample_aligned['transformations'])}")
    print()

    # Test 3: Visualize a plant sequence
    print("=" * 60)
    print("Test 3: Visualizing plant sequences")
    print("=" * 60)

    # Visualize unaligned sequence
    visualize_plant_sequence(
        sample_no_align,
        dense_points=False,
        color_by_organ=True,
        show_leaf_tips=True,
        show_connections=False,
        spacing=50.0,
        window_name="Plant Sequence Unaligned",
        offscreen=offscreen,
    )

    # Visualize aligned sequence
    visualize_plant_sequence(
        sample_aligned,
        dense_points=False,
        color_by_organ=True,
        show_leaf_tips=True,
        show_connections=False,
        spacing=50.0,
        window_name="Plant Sequence Aligned",
        offscreen=offscreen,
    )

    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
