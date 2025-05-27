import os
import sys
import numpy as np
import open3d as o3d
import argparse
import pandas as pd
import pickle

class PointCloudLabeler:
    def __init__(self):
        self.point_cloud = None
        self.vis = None
        self.labels = {}
        self.current_label = "leaf_tip"
        self.selected_points = []
        self.sphere_radius = 0.05  # default radius for sphere around selected points
        self.sphere_objects = []
        self.view_control = None
        self.original_colors = None
        self.points_data = None  # Will store additional columns from the input file
        self.file_path = None
        self.file_type = None
        self.temp_sphere = None
        
        # Key mapping definitions
        self.key_to_callback = {
            76: self.label_selected_points_callback,  # L key
            84: self.toggle_label_type_callback,      # T key
            67: self.clear_selection_callback,        # C key
            83: self.save_labels_callback,            # S key
            43: self.increase_radius_callback,        # + key
            45: self.decrease_radius_callback,        # - key
            82: self.add_sphere_callback,             # R key
            68: self.delete_spheres_callback,         # D key
        }
        
        self.picked_points = []

    def load_point_cloud(self, file_path):
        """Load point cloud from PLY or TXT file"""
        self.file_path = file_path
        self.file_type = os.path.splitext(file_path)[1].lower()
        
        if self.file_type == '.ply':
            self.point_cloud = o3d.io.read_point_cloud(file_path)
            if not self.point_cloud.has_colors():
                # Add default colors (grey) if no color information is present
                points = np.asarray(self.point_cloud.points)
                colors = np.ones((len(points), 3)) * 0.7  # Grey color
                self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
            
            self.original_colors = np.asarray(self.point_cloud.colors).copy()
            
            # Try to load additional data if there are scalar fields in the PLY file
            try:
                # Create a numpy array from the point cloud
                points = np.asarray(self.point_cloud.points)
                # For PLY, we might need a custom reader to extract other scalar fields
                # This is a simple approach - you might need to extend this
                self.points_data = pd.DataFrame(points, columns=['x', 'y', 'z'])
            except Exception as e:
                print(f"Warning: Could not extract additional data from PLY file: {e}")
                self.points_data = pd.DataFrame(np.asarray(self.point_cloud.points), columns=['x', 'y', 'z'])
                
        elif self.file_type == '.txt':
            # Try to load TXT file as CSV with space, tab, or comma delimiter
            try:
                # Try different delimiters
                for sep in [None, ',', ' ', '\t']:
                    try:
                        df = pd.read_csv(file_path, sep=sep, engine='python')
                        if len(df.columns) >= 3:  # At least x, y, z needed
                            break
                    except:
                        continue
                
                # If nothing worked, try with fixed whitespace
                if 'df' not in locals() or len(df.columns) < 3:
                    df = pd.read_csv(file_path, delim_whitespace=True, header=None)
                
                # Check if we have at least x, y, z columns
                if len(df.columns) < 3:
                    raise ValueError("File must have at least 3 columns for x, y, z coordinates")
                
                # Store all data
                self.points_data = df
                
                # Extract just the XYZ coordinates for the point cloud
                points = df.iloc[:, 0:3].values
                
                # Create point cloud
                self.point_cloud = o3d.geometry.PointCloud()
                self.point_cloud.points = o3d.utility.Vector3dVector(points)
                
                # Assign default colors (grey)
                colors = np.ones((len(points), 3)) * 0.7  # Grey color
                self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
                self.original_colors = colors.copy()
                
                print(f"Loaded TXT file with {len(points)} points and {len(df.columns)} columns")
                print(f"Column names: {df.columns.tolist()}")
                
            except Exception as e:
                print(f"Error loading TXT file: {e}")
                sys.exit(1)
        else:
            print(f"Unsupported file type: {self.file_type}. Please use .ply or .txt")
            sys.exit(1)
            
        # If successful, print some info
        print(f"Loaded point cloud with {len(self.point_cloud.points)} points")
        
        # If we have previously saved labels for this file, load them
        label_file = file_path + ".labels"
        if os.path.exists(label_file):
            try:
                with open(label_file, 'rb') as f:
                    self.labels = pickle.load(f)
                print(f"Loaded {sum(len(pts) for pts in self.labels.values())} existing labels from {label_file}")
                self.update_point_colors()
            except Exception as e:
                print(f"Error loading existing labels: {e}")

    def update_point_colors(self):
        """Update point colors based on labels"""
        # Reset to original colors
        colors = self.original_colors.copy()
        
        # Color labeled points
        for label_type, point_indices in self.labels.items():
            for idx in point_indices:
                if idx < len(colors):  # Safety check
                    if label_type == "leaf_tip":
                        colors[idx] = [1, 0, 0]  # Red for leaf tips
                    elif label_type == "other":
                        colors[idx] = [0, 1, 0]  # Green for other labels
        
        # Color selected points
        for idx in self.picked_points:
            if idx < len(colors):  # Safety check
                colors[idx] = [0, 0, 1]  # Blue for selected points
            
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
        if self.vis is not None:
            self.vis.update_geometry(self.point_cloud)
            self.vis.poll_events()
            self.vis.update_renderer()

    def start_labeling(self):
        """Start the interactive labeling interface"""
        if self.point_cloud is None:
            print("No point cloud loaded")
            return

        # Create a standard visualizer (not VisualizerWithEditing which lacks key callbacks)
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Point Cloud Labeler", width=1024, height=768)
        self.vis.add_geometry(self.point_cloud)
        
        # Set up view control and options
        self.view_control = self.vis.get_view_control()
        render_option = self.vis.get_render_option()
        render_option.point_size = 3.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
        
        # Add a coordinate system for reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0])
        self.vis.add_geometry(coordinate_frame)
        
        print("\n=== Point Cloud Labeler ===")
        print("Instructions:")
        print("- Pick points by holding Ctrl and left-clicking")
        print("- L: Label selected points as current label type")
        print("- T: Toggle between label types (leaf_tip, other)")
        print("- C: Clear current selection")
        print("- S: Save labels")
        print("- +/-: Increase/decrease sphere radius")
        print("- R: Add sphere around selected point")
        print("- D: Delete spheres")
        print("- Q: Quit")
        
        # Register key callbacks for the visualizer
        for key, callback in self.key_to_callback.items():
            self.vis.register_key_callback(key, callback)
        
        # Register picker
        self.vis.register_key_action_callback(90, self.pick_points_callback)  # Z key to enter pick mode
        
        # Run the visualizer
        self.vis.run()
        self.vis.destroy_window()
        
        # Export labels when done
        self.export_labels()
    
    def pick_points_callback(self, vis, action, mods):
        """Custom point picking functionality"""
        # We'll implement a simple point picker using raycasting
        if action == 1:  # Key press
            print("Entering point picking mode (click on points to select them)")
            
            def pick_point(event):
                if event.type == o3d.visualization.gui.EventType.MOUSE_DOWN and event.is_button_down(o3d.visualization.gui.MouseButton.LEFT):
                    # Cast ray from camera to clicked point
                    x, y = event.x, event.y
                    ray = self.vis.get_view_control().unproject_ray(x, y)
                    points = np.asarray(self.point_cloud.points)
                    
                    # Find closest point to ray
                    # This is a simplified approach - could be improved with more sophisticated picking
                    ray_origin = ray[0]
                    ray_direction = ray[1]
                    
                    # Calculate distances from ray to all points
                    # This is a simple approximation
                    distances = np.abs(np.cross(ray_direction, points - ray_origin))
                    distances = np.linalg.norm(distances, axis=1)
                    
                    # Get closest point
                    closest_point_idx = np.argmin(distances)
                    
                    # Add to picked points
                    self.picked_points.append(closest_point_idx)
                    print(f"Selected point {closest_point_idx}")
                    
                    # Update colors
                    self.update_point_colors()
                    
                    return True
                return False
            
            # Register mouse callback for picking
            # Note: This is a simplified approach and may not work in all Open3D versions
            # For a more robust approach, consider using the picking functionality in draw_geometries_with_editing
            try:
                self.vis.register_mouse_callback(pick_point)
            except:
                print("Point picking mode not supported in this version of Open3D.")
                print("Use draw_geometries_with_editing() instead.")
        return False
        
    def label_selected_points_callback(self, vis):
        """Label the currently selected points with the current label type"""
        if not self.picked_points:
            print("No points selected")
            return False
            
        # Initialize label type in dictionary if it doesn't exist
        if self.current_label not in self.labels:
            self.labels[self.current_label] = []
            
        # Add selected points to labels
        for idx in self.picked_points:
            if idx not in self.labels[self.current_label]:
                self.labels[self.current_label].append(idx)
                
        print(f"Labeled {len(self.picked_points)} points as '{self.current_label}'")
        
        # Clear selection after labeling
        self.picked_points = []
        self.update_point_colors()
        return True
        
    def toggle_label_type_callback(self, vis):
        """Toggle between different label types"""
        if self.current_label == "leaf_tip":
            self.current_label = "other"
        else:
            self.current_label = "leaf_tip"
        print(f"Current label type: {self.current_label}")
        return False
        
    def clear_selection_callback(self, vis):
        """Clear the current selection"""
        self.picked_points = []
        self.update_point_colors()
        print("Selection cleared")
        return False
        
    def save_labels_callback(self, vis):
        """Save the current labels to a file"""
        if not self.labels:
            print("No labels to save")
            return False
            
        label_file = self.file_path + ".labels"
        with open(label_file, 'wb') as f:
            pickle.dump(self.labels, f)
            
        # Also save a readable text version
        text_label_file = self.file_path + ".labels.txt"
        with open(text_label_file, 'w') as f:
            f.write("# Point Cloud Labels\n")
            f.write(f"# File: {self.file_path}\n")
            f.write(f"# Total points: {len(self.point_cloud.points)}\n\n")
            
            for label_type, point_indices in self.labels.items():
                f.write(f"## {label_type}: {len(point_indices)} points\n")
                for idx in point_indices:
                    if idx < len(self.point_cloud.points):  # Safety check
                        point = np.asarray(self.point_cloud.points)[idx]
                        f.write(f"{idx}: {point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}")
                        
                        # Add any additional data columns if available
                        if self.points_data is not None and len(self.points_data.columns) > 3:
                            try:
                                extra_data = self.points_data.iloc[idx, 3:].values
                                f.write(f" | {', '.join(map(str, extra_data))}")
                            except:
                                # Handle any issues with accessing additional data
                                pass
                        f.write("\n")
                f.write("\n")
                
        print(f"Labels saved to {label_file} and {text_label_file}")
        return False
        
    def increase_radius_callback(self, vis):
        """Increase the sphere radius"""
        self.sphere_radius += 0.01
        print(f"Sphere radius: {self.sphere_radius:.2f}")
        return False
        
    def decrease_radius_callback(self, vis):
        """Decrease the sphere radius"""
        self.sphere_radius = max(0.01, self.sphere_radius - 0.01)
        print(f"Sphere radius: {self.sphere_radius:.2f}")
        return False
        
    def add_sphere_callback(self, vis):
        """Add a sphere around the selected point"""
        if not self.picked_points:
            print("No points selected")
            return False
            
        for idx in self.picked_points:
            if idx < 0 or idx >= len(self.point_cloud.points):
                continue
                
            point = np.asarray(self.point_cloud.points)[idx]
            sphere = o3d.geometry.TriangleMesh.create_sphere(
                radius=self.sphere_radius)
            sphere.translate(point)
            
            # Give each sphere a unique color based on the current label
            if self.current_label == "leaf_tip":
                sphere.paint_uniform_color([1, 0.7, 0])  # Orange for leaf tips
            else:
                sphere.paint_uniform_color([0, 0.7, 0.7])  # Cyan for other
                
            self.sphere_objects.append(sphere)
            self.vis.add_geometry(sphere)
            
        print(f"Added {len(self.picked_points)} spheres")
        return True
        
    def delete_spheres_callback(self, vis):
        """Delete all sphere objects"""
        for sphere in self.sphere_objects:
            self.vis.remove_geometry(sphere)
        self.sphere_objects = []
            
        print("All spheres deleted")
        return True

    def export_labels(self, output_file=None):
        """Export the labeled points to a new point cloud file"""
        if not self.labels:
            print("No labels to export")
            return
            
        if output_file is None:
            output_file = self.file_path + ".labeled" + self.file_type
            
        # Create a copy of the original data with an additional label column
        if self.points_data is not None:
            labeled_data = self.points_data.copy()
            labeled_data['label'] = 'none'  # Default label
            
            # Set labels for each point
            for label_type, point_indices in self.labels.items():
                for idx in point_indices:
                    if idx < len(labeled_data):  # Safety check
                        labeled_data.loc[idx, 'label'] = label_type
                    
            # Save as CSV or similar format
            if output_file.endswith('.txt') or output_file.endswith('.csv'):
                labeled_data.to_csv(output_file, index=False)
            elif output_file.endswith('.ply'):
                # For PLY, we need to save with custom attributes
                # Update the colors based on labels for better visualization
                colors = self.original_colors.copy()
                for label_type, point_indices in self.labels.items():
                    for idx in point_indices:
                        if idx < len(colors):  # Safety check
                            if label_type == "leaf_tip":
                                colors[idx] = [1, 0, 0]  # Red for leaf tips
                            elif label_type == "other":
                                colors[idx] = [0, 1, 0]  # Green for other labels
                
                labeled_cloud = o3d.geometry.PointCloud()
                labeled_cloud.points = o3d.utility.Vector3dVector(np.asarray(self.point_cloud.points))
                labeled_cloud.colors = o3d.utility.Vector3dVector(colors)
                
                o3d.io.write_point_cloud(output_file, labeled_cloud)
                
            print(f"Labeled point cloud exported to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Point Cloud Labeling Tool")
    parser.add_argument("file", help="Path to point cloud file (PLY or TXT)")
    args = parser.parse_args()
    
    labeler = PointCloudLabeler()
    labeler.load_point_cloud(args.file)
    labeler.start_labeling()

if __name__ == "__main__":
    main()