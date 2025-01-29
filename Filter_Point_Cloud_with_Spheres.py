import open3d as o3d
import numpy as np
import laspy

def load_las_file(file_path):
    """Load a LAS file and create a point cloud."""
    
    # Print a message indicating that the LAS file is being loaded.
    print(f"Loading LAS file: {file_path}")
    
    # Read the LAS file and extract x, y, z coordinates.
    las_data = laspy.read(file_path)
    points = np.vstack((las_data.x, las_data.y, las_data.z)).T  
    
    # Create an Open3D point cloud object and assign points to it.
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def downsample_point_cloud(point_cloud, voxel_size=0.05):
    """If chosen by the user, downsample the point cloud using voxel grid downsampling for faster processing."""
    
    # Print a message that the point cloud is being downsampled with the chosen voxel size.
    print(f"Downsampling point cloud with voxel size: {voxel_size}")
    
    # Downsample the point cloud and return the downsampled point cloud.
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)
    return downsampled_point_cloud

def create_octree(point_cloud, max_depth=4, size_expand=0.01):
    """Create an octree from the point cloud."""
    
    # Print a message that the Octree is being created at the chosen depth.
    print(f"Creating Octree with max depth: {max_depth}")
    
    # Generate the octree from the point cloud.
    octree = o3d.geometry.Octree(max_depth=max_depth)
    octree.convert_from_point_cloud(point_cloud, size_expand=size_expand)
    return octree

def is_point_in_sphere(point, center, radius):
    """Check if a point is within a sphere."""
    
    # Return True if point is within the sphere, False otherwise.
    return np.linalg.norm(point - center) <= radius

def filter_points_in_sphere(node, node_info, point_cloud):
    """Filter points within a sphere for a given octree node."""
    
    # Check whether a node is internal or a leaf node.
    if isinstance(node, (o3d.geometry.OctreeInternalNode, o3d.geometry.OctreeLeafNode)):
        
        # Calculate the center position and radius of a sphere.
        center = node_info.origin + node_info.size / 2
        radius = node_info.size / 2
        
        # Filter points that lie inside the sphere.
        if hasattr(node, "indices"):
            indices = node.indices
            points = np.asarray(point_cloud.points)[indices]
            filtered_indices = [i for i, point in zip(indices, points) if is_point_in_sphere(point, center, radius)]
            node.indices = filtered_indices
        
        # Recursively process child nodes.
        if hasattr(node, "children"):
            for child_node in node.children:
                if child_node is not None:
                    filter_points_in_sphere(child_node, node_info, point_cloud)

def traverse_and_filter_octree(octree, point_cloud, filter_initial_cube):
    """Traverse and filter the octree."""
    
    # Callback function to traverse the octree.
    def traverse_callback(node, node_info):
        
        # Filter the initial octree cube if required.
        if filter_initial_cube or node_info.depth > 0:
            filter_points_in_sphere(node, node_info, point_cloud)
        return False  # Continue traversal

    # Start traversing the octree.
    octree.traverse(traverse_callback)

def get_filtered_points(octree, point_cloud):
    """Collect the filtered points from the octree."""
    
    # Initialize a list to store the filtered points.
    filtered_points = []

    # Callback function to collect the points from the leaf nodes.
    def collect_points(node, node_info):
        if isinstance(node, o3d.geometry.OctreeLeafNode):
            if hasattr(node, "indices"):
                indices = node.indices
                points = np.asarray(point_cloud.points)[indices]
                filtered_points.extend(points)
        return False  # Continue traversal

    # Start collecting the points from the octree.
    octree.traverse(collect_points)
    return np.array(filtered_points)

def visualize_point_cloud(point_cloud, title="Point Cloud"):
    """Visualize the point cloud."""
    
    # Print a message indicating that the object is being visualized.
    print(f"Visualizing {title}...")
    
    # Create a visualizer window and add the point cloud.
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=800, height=600, left=50, top=50, visible=True)
    vis.add_geometry(point_cloud)
    vis.get_render_option().background_color = np.asarray([1, 1, 1])  # Set background to white
    
    # Start the visualization and destroy the window after use.
    vis.run()
    vis.destroy_window()

def process_point_cloud(las_file_path, max_depth=2, downsampling_voxel_size=0, filter_initial_cube=False):
    """Main processing function to load, filter, and visualize a point cloud."""
    
    # Step 1: Load the LAS file and create a point cloud.
    point_cloud = load_las_file(las_file_path)

    # Step 2: Optionally downsample the point cloud for faster processing.
    if downsampling_voxel_size is not None and downsampling_voxel_size > 0:
        point_cloud = downsample_point_cloud(point_cloud, voxel_size=downsampling_voxel_size)
    else:
        print("No downsampling applied. Keeping original point cloud density.")

    # Step 3: Create an octree from the point cloud.
    octree = create_octree(point_cloud, max_depth=max_depth)

    # Step 4: Traverse and filter the octree based on criteria.
    traverse_and_filter_octree(octree, point_cloud, filter_initial_cube)

    # Step 5: Retrieve the filtered points after traversal.
    filtered_points = get_filtered_points(octree, point_cloud)

    # Step 6: Create a filtered point cloud for visualization.
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # Step 7: Visualize the filtered point cloud.
    visualize_point_cloud(filtered_pcd, title="Filtered Point Cloud")

# Example usage
if __name__ == "__main__":
    # Main parameters for processing
    las_file_path = "C:/Users/karol/Desktop/Coding/Matom AI task/2743_1234.las"  # Replace with the path to your .las file.
    max_depth = 2  # Choose the maximum depth for the octree.

    # Additional parameters
    downsampling_voxel_size = 0  # Set the voxel size for downsampling.
    # 0 or negative values will result in no downsampling and will display the original point density.
    # Values larger than 0 will reduce the point cloud density which is useful for testing and faster processing.
    filter_initial_cube = False  # Choose whether to create a sphere and filter the points in the initial single cube of the point cloud (at depth 0).

    # Process the point cloud from the LAS file.
    process_point_cloud(las_file_path, max_depth=max_depth, downsampling_voxel_size=downsampling_voxel_size, filter_initial_cube=filter_initial_cube)
