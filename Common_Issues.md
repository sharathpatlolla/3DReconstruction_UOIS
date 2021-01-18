### How to Free Cuda Memory

- Check current GPU usage from terminal,` nvidia-smi`
- Check and Delete the process that is using a lot of memory, `sudo kill -9 PROCESS_PID`

### How to collect point cloud data using Gazebo: (Manually)
- Source the setup file first, \
`source catkin_ws/devel/setup.sh`

- Create a gazebo world with the scene and the depth camera. I created this manually and saved the world file in `/home/ll4ma/camera_cluttered_scene_fruits_and_box.world`
-- For the starter environment, use the command Martin mentioned, (This needs a lot of installing and dependencies, so check lab website for that \
`roslaunch ll4ma_robots_gazebo lbr4.launch end_effector:=allegro robot_table:=true world_launch_file:=grasp_scene.launch`

- Open the gazebo world in gazebo simulator, \
`rosrun gazebo_ros gazebo /home/ll4ma/camera_cluttered_scene_fruits_and_box.world`

-  Now open `rviz` in another terminal without exiting the gazebo, to record and visualize the data and follow the last section of this tutorial \
`rosrun rviz rviz`

- After following the tutorial, make sure that you can see the point cloud in `rviz`. For the raw cloud, you have to subscribe to the topic /camera/depth/points if you have camera running in gazebo already you can visualize it by rviz->Add->By topic-> and choosing camera/depth/points from the hierarchy.

- Use `rosbag` to save the point cloud to a file (.bag), rosbag reference \
`rosbag record -O ~/cluttered_pcd.bag /camera/depth/points`

- Now use `bag_to_pcd` to create point clouds from the ‘.bag’ output file in the previous step \
`rosrun pcl_ros bag_to_pcd <input_file.bag> <topic> <output_directory>` \
Example: ` rosrun pcl_ros bag_to_pcd /home/ll4ma/cluttered_pcd.bag /camera/depth/points /home/ll4ma/cluttered_pcds` \

NOTE: The point cloud you get from this is flipped and is raw. To transform the point cloud, I did this, (I was using open3d library, so the command was), \
`temp_pcd = o3d.io.read_point_cloud(pcd_file_path)` -> Read the point cloud file of type, .pcd \
`temp_pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])` -> Transform (Do this only for visualization purposes, it’s not required for the segmentation code because this command is already performed) \
`o3d.visualization.draw_geometries([temp_pcd])` -> Visualize
