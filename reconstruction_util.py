"""
This api is written by Martin Matak, one of the authors of the PointSDF reconstruction paper and our fellow researcher. Thanks!
"""

import sys
import numpy as np

# Change this to include the path of PointConv
sys.path.append("/home/ll4ma/Downloads/ResearchStuff/uois_implementation/uois-new2/dependencies/PointConv")
from sdf_pointconv_model import get_sdf_prediction, get_pointconv_model
from sdf_dataset import get_processed_pcd
from mise import mise_voxel
from pypcd import pypcd
from pypcd import *

class MeshReconstructor():

    def __init__(self, model_path):
        print("Loading model from: " + model_path)
        self.get_sdf, self.get_embedding, _ = get_sdf_prediction(get_pointconv_model, model_path)

    def reconstruct_from_msg(self, msg):
        return self.reconstruct(PointCloud.from_msg(msg))


    def reconstruct(self, point_cloud, output_obj_path):
        print("reconstructing...")
        # pointcloud should be of type pypcd.PointCloud

        # Bounds of 3D space to evaluate in: [-bound, bound] in each dim.
        bound = 0.8
        # Starting voxel resolution.
        initial_voxel_resolution = 32
        # Final voxel resolution.
        final_voxel_resolution = 512
    
        # Point cloud for this view.
        pc_, length, scale, centroid_diff = get_processed_pcd(point_cloud, object_frame=False, verbose=False)
        
        voxel_size = (2.*bound * length) / float(final_voxel_resolution)    
        point_clouds_ = np.reshape(pc_, (1,1000,3))

        # Make view specific sdf func.
        def get_sdf_query(query_points):
            return self.get_sdf(point_clouds_, query_points)

        recon_voxel_pts = mise_voxel(get_sdf_query, bound, initial_voxel_resolution, final_voxel_resolution, voxel_size, centroid_diff, output_obj_path, verbose=False)

        print("reconstruction finished!")


