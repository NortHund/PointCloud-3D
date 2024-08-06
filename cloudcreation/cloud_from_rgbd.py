def pointcloud_from_rgbd(color_dir, depth_dir, intrinsic_json, poses_json):
    # Load camera intrinsics
    intrinsic = o3d.io.read_pinhole_camera_intrinsic(intrinsic_json)
    
    # Load poses
    with open(poses_json, 'r') as f:
        poses = json.load(f)
    
    print(f"Creating a point cloud from {len(poses)} frames")
    
    # Initialize an empty point cloud to store the combined point cloud
    combined_pcd = o3d.geometry.PointCloud()
    
    for filename, pose in poses.items():
        # Read color image
        color_path = os.path.join(color_dir, f"{filename}.jpg")
        color_image = cv2.imread(color_path)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        # Read depth image
        depth_path = os.path.join(depth_dir, f"{filename}.png")
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Depth image should be loaded as uint16
        
        # Convert to Open3D image format
        color_o3d = o3d.geometry.Image(color_image)
        depth_o3d = o3d.geometry.Image(depth_image)
        
        # Create RGBD image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, convert_rgb_to_intensity=False, depth_trunc=8.0)
        
        # Create point cloud from RGBD image
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic)
            
        #extrinsic = extrinsics[filename]
        #pcd.transform(extrinsic)
        
        # Apply the camera pose transformation
        pose_matrix = np.array(pose)
        pcd.transform(pose_matrix)
        
        # Add the transformed point cloud to the combined point cloud
        combined_pcd += pcd
    
    combined_pcd.voxel_down_sample(voxel_size=0.02)
    
    #cloud_path = "cloud.ply"
    #o3d.io.write_point_cloud(cloud_path, combined_pcd)
    
    return combined_pcd