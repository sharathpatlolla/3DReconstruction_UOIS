import glob
import imageio
import numpy as np
import open3d as o3d

class DataLoaderOCID:
    def __init__(self):
        # Add dataset name, image shape and other specifics here
        self.img_dim = (480,640)
        
    def get_rgb_from_pcd(self, point_cloud):
        """
        Returns rgb array from point cloud by filling in null values with 0 

        Args:
            point_cloud: input point cloud in .pcd format
        Returns:
            rgb_img: rgb image corresponding to the point cloud
        """

        num_missing = self.img_dim[0]*self.img_dim[1] - np.asarray(point_cloud.colors).shape[0]
        filled_in_rgb_img = np.concatenate([np.asarray(point_cloud.colors), np.zeros((num_missing,3))])
        rgb_img = np.round(255 * filled_in_rgb_img.reshape(self.img_dim[0],self.img_dim[1],3)).astype(np.uint8)

        return rgb_img

    def get_xyz_from_pcd(self, point_cloud):
        """
        Returns xyz array from point cloud by filling in null values with 0 

        Args:
            point_cloud: input point cloud in .pcd format
        Returns:
            xyz_img: xyz image corresponding to the point cloud
        """

        num_missing = self.img_dim[0]*self.img_dim[1] - np.asarray(point_cloud.points).shape[0]
        filled_in_points = np.concatenate([np.asarray(point_cloud.points), np.zeros((num_missing,3))])

        xyz_img = np.asarray(filled_in_points).reshape(self.img_dim[0],self.img_dim[1],3)
        xyz_img[np.isnan(xyz_img)] = 0
        xyz_img[...,1] *= -1
        return xyz_img

    def get_dict_of_scenes(self, dir_path):
        """
        Returns dictionary of scenes where each scene is one image and each image has its rgb, xyz, depth, label values

        Args:
            dir_path: path where the individual directories [rgb, label, depth] are present
        Returns:
            dict_of_np_array_dict: dictionary (dict_of_np_array_dict) whose key and values are defined below
                key : directory_path + file_name
                         it should be noted that the file_name is same in each of the sub directories (rgb, label and depth)
                val : dictionary(np_array_dict) of images with 4 key-value pairs; key-val pairs of np_array_dict are below
                         key: image_type -> {'rgb', 'xyz', 'label', 'depth'} 
                         val: corresponding image_array for each type
        """
        label_dir_path = dir_path + '/label'
        label_image_files_path = sorted(glob.glob(label_dir_path + '/*.png'))
        label_image_files_name = {x[len(label_dir_path)+1:-4] for x in label_image_files_path}

        pcd_dir_path = dir_path + '/pcd'
        pcd_image_files_path = sorted(glob.glob(pcd_dir_path + '/*.pcd'))
        pcd_image_files_name = {x[len(pcd_dir_path)+1:-4] for x in pcd_image_files_path}

        dict_of_np_array_dict = {}
        for cur_file_name in label_image_files_name:
            if cur_file_name in pcd_image_files_name:
                np_array_dict = {}
                np_array_dict['label'] = imageio.imread(label_dir_path+'/'+cur_file_name+'.png')

                temp_pcd = o3d.io.read_point_cloud(pcd_dir_path+'/'+cur_file_name+'.pcd', remove_nan_points=False)
                np_array_dict['rgb'] = self.get_rgb_from_pcd(temp_pcd)
                np_array_dict['xyz'] = self.get_xyz_from_pcd(temp_pcd)
                np_array_dict['depth'] = np_array_dict['xyz'][:,:,2]

                dict_of_np_array_dict[dir_path+'/'+cur_file_name] = np_array_dict
        return dict_of_np_array_dict
    
    def sample_pcd(self, point_cloud):
        """
        Returns uniform sample of the given point cloud. This can be refactored later to include more specific downsampling based on the camera view etc

        Args:
            point_cloud: input point cloud
        """
        return point_cloud.uniform_down_sample(5)
    