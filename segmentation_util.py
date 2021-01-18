"""
    @inproceedings{xie2020uois3d,
    author    = {Christopher Xie and Yu Xiang and Arsalan Mousavian and Dieter Fox},
    title     = {Unseen Object Instance Segmentation for Robotic Environments},
    booktitle = {arXiv:2007.08073},
    year      = {2020}
    }
"""

import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os

import sklearn
import sklearn.linear_model
import torch

import dependencies.uois3d.src.data_augmentation as data_augmentation
import dependencies.uois3d.src.segmentation as segmentation
import dependencies.uois3d.src.evaluation as evaluation
import dependencies.uois3d.src.util.utilities as util_

from data_util import DataLoaderOCID 
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = "0" # If you have more than one GPU, Change this


def get_segmented_masks_from_rgbd(dict_of_scenes, training_weights_checkpoint_dir,
                                  dsn_config = None, rrn_config = None, uois3d_config = None):
    """
    Returns segmented masks (seg_masks) of the given scenes
    
    Args:
        dict_of_scenes: dictionary of scenes where each scene consists of one image. Input to this function is the output of data_util.DataLoaderOCID.get_dict_of_scenes() function
        training_weights_checkpoint_dir: path to the trained weights
        dsn_config: dsn hyperparameters in dictionary format
        rrn_config: rrn hyperparameters in dictionary format
        uois3d_config: combined net hyperparameters in dictionary format
        
    Returns:
        rgb_imgs: rgb for each img, np array shape = (N, 480, 640, 3) where N is the number of imgs
        xyz_imgs: xyz coordinates for each img, np array shape = (N, 480, 640, 3) where N is the number of imgs 
        seg_masks: predicted object labels for each img, np array shape = (N, 480, 640) where N is the number of imgs 
        label_imgs: true object labels for each img, np array shape = (N, 480, 640) where N is the number of imgs 
        fg_masks: foreground mask for each img, np array shape = (N, 480, 640) where N is the number of imgs 
        file_names: list of length N, where N is the number of imgs
    
    This code is modified from "https://github.com/chrisdxie/uois/blob/master/uois_3D_example.ipynb". It is also cited above and the repo is included in the dependencies
    """
    if dsn_config is None:
        dsn_config = {

            # Sizes
            'feature_dim' : 64, # 32 would be normal

            # Mean Shift parameters (for 3D voting)
            'max_GMS_iters' : 10, 
            'epsilon' : 0.05, # Connected Components parameter
            'sigma' : 0.02, # Gaussian bandwidth parameter
            'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
            'subsample_factor' : 5,

            # Misc
            'min_pixels_thresh' : 500,
            'tau' : 15.,

        }
    
    if rrn_config is None:
        rrn_config = {

        # Sizes
        'feature_dim' : 64, # 32 would be normal
        'img_H' : 224,
        'img_W' : 224,

        # architecture parameters
        'use_coordconv' : False,

        }
    
    if uois3d_config is None:
        uois3d_config = {

        # Padding for RGB Refinement Network
        'padding_percentage' : 0.25,

        # Open/Close Morphology for IMP (Initial Mask Processing) module
        'use_open_close_morphology' : True,
        'open_close_morphology_ksize' : 9,

        # Largest Connected Component for IMP module
        'use_largest_connected_component' : True,

        }
    
    dsn_filename = training_weights_checkpoint_dir + 'DepthSeedingNetwork_3D_TOD_checkpoint.pth'
    rrn_filename = training_weights_checkpoint_dir + 'RRN_OID_checkpoint.pth'
    uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_filename
    uois_net_3d = segmentation.UOISNet3D(uois3d_config, 
                                         dsn_filename,
                                         dsn_config,
                                         rrn_filename,
                                         rrn_config
                                        )
    

    N = len(dict_of_scenes)
    if N == 0:
        print("Number of scenes in the dictonary are 0")
        return None, None, None, None, None, None
    rgb_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
    xyz_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
    label_imgs = np.zeros((N, 480, 640), dtype=np.uint8)
    file_names = [None]*N
    
    i = 0
    for key, val in dict_of_scenes.items():
        # file names
        file_names[i] = key
        
        # RGB
        rgb_img = val['rgb']
        rgb_imgs[i] = data_augmentation.standardize_image(rgb_img)

        # XYZ
        xyz_imgs[i] = val['xyz']

        # Label
        label_imgs[i] = val['label']
        i += 1

    batch = {
        'rgb' : data_augmentation.array_to_tensor(rgb_imgs),
        'xyz' : data_augmentation.array_to_tensor(xyz_imgs),
    }
    
    print("Number of images: {0}".format(N))
    ### Compute segmentation masks ###
    fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)
    st_time = time()
    total_time = time() - st_time
    print('Total time taken for Segmentation: {0} seconds'.format(round(total_time, 3)))

    print("fg_masks.shape = ", fg_masks.shape)
    print("seg_masks.shape = ", seg_masks.shape)
    # Get results in numpy
    seg_masks = seg_masks.cpu().numpy()
    if np.array_equal(seg_masks, np.zeros(seg_masks.shape)):
        print("Seg_mask is zero")
        sys.exit()
    fg_masks = fg_masks.cpu().numpy()
    center_offsets = center_offsets.cpu().numpy().transpose(0,2,3,1)
    initial_masks = initial_masks.cpu().numpy()
    rgb_imgs = util_.torch_to_numpy(batch['rgb'].cpu(), is_standardized_image=True)

    return rgb_imgs, xyz_imgs, seg_masks, label_imgs, fg_masks, file_names

def get_segmented_point_clouds(seg_masks, depth):
    """
    Returns point clouds of all objects in the image, given segmented masks
    
    Args:
        seg_masks: predicted object labels for one img. np array shape = (480, 640)
        depth: depth image which is essentially the z component of xyz image. np array shape = (480, 640)
        
    Returns:
        segmented_pcds: list of point clouds where each point cloud is in '.pcd' format and length = number of predicted objects
    """    
    obj_labels = np.unique(seg_masks)
    num_objs = obj_labels.shape[0]+1
    rows, cols = seg_masks.shape
    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/num_objs) for i in range(num_objs)]
        
    object_dict = {}
    # key - object label; val - depth array of that object
    for i in obj_labels:
        object_dict[i] = np.zeros((rows,cols), dtype = np.float32)

    for i in range(rows):
        for j in range(cols):
            if seg_masks[i][j] != 0 and seg_masks[i][j] != -1:
                object_dict[seg_masks[i][j]][i][j] = depth[i][j]
    
    segmented_pcds = []
    for key, val in object_dict.items():
        if key == -1 or key == 0:
            continue
        img = o3d.geometry.Image(val)
        pcd_from_depth = o3d.geometry.PointCloud.create_from_depth_image(
            img,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

        # Multiply with Transformation matrix to get correct view of the PCD
        pcd_from_depth.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pcd_from_depth.paint_uniform_color(np.array(colors[key][:3], dtype = np.uint8) * 255)
        segmented_pcds.append(pcd_from_depth)
    return segmented_pcds

def visualize_predicted_and_true_segment_masks(rgb_imgs, xyz_imgs, seg_masks, label_imgs):
    """
    Plots a graph with multiple sub graphs comparing the rgb, depth, predicted segmentation masks and true segmentation masks. Grid size is N x 4 where N is the number of images
    
    Args:
        rgb_imgs: rgb for each img, np array shape = (N, 480, 640, 3) where N is the number of imgs 
        xyz_imgs: xyz coordinates for each img, np array shape = (N, 480, 640, 3) where N is the number of imgs 
        seg_masks: predicted object labels for each img, np array shape = (N, 480, 640) where N is the number of imgs 
        label_imgs: true object labels for each img, np array shape = (N, 480, 640) where N is the number of imgs 
        
    Returns: None
    
    This code is modified from "https://github.com/chrisdxie/uois/blob/master/uois_3D_example.ipynb". It is also cited above and the repo is included in the dependencies
    """
    N = rgb_imgs.shape[0]
    fig_index = 1
    for i in range(N):
        num_objs = max(np.unique(seg_masks[i,...]).max(), np.unique(label_imgs[i,...]).max()) + 1

        rgb = rgb_imgs[i].astype(np.uint8)
        
        depth = xyz_imgs[i,...,2]

        seg_mask_plot = util_.get_color_mask(seg_masks[i,...], nc=num_objs)
        gt_masks = util_.get_color_mask(label_imgs[i,...], nc=num_objs)

        images = [rgb, depth, seg_mask_plot, gt_masks]
        titles = [f'Image {i+1}', 'Depth',
                  f"Refined Masks. #objects: {np.unique(seg_masks[i,...]).shape[0]-1}",
                  f"Ground Truth. #objects: {np.unique(label_imgs[i,...]).shape[0]-1}"
                 ]
        util_.subplotter(images, titles, fig_num=i+1)
    
def get_segmentation_metrics_for_dataset(big_dir, training_weights_checkpoint_dir):
    """
    Returns segmentation metrics i.e. Overlap and Inbound P/R/F mentioned in the UOIS paper
    
    Args:
        big_dir: path where the individual directories [rgb, label, depth] are present
        training_weights_checkpoint_dir: path to the trained weights
        
    Returns:
        av_eval_metrics: dictionary whose key-value pairs are below,
            key: name of the the evaluation metric i.e. one of ('Objects F-measure', 'Objects Precision', 'Objects Recall', 'Boundary F-measure', 'Boundary Precision', 'Boundary Recall')
            val: average score of the curresponding metric for all images
    """
    label_image_files_path = sorted(glob.glob(big_dir, recursive = True))
    keys = ['Objects F-measure', 'Objects Precision', 'Objects Recall', 'Boundary F-measure', 'Boundary Precision', 'Boundary Recall']
    av_eval_metrics = dict.fromkeys(keys, 0)
    n = 0
    dataloader_obj = DataLoaderOCID()
    for dir_path in label_image_files_path:
        dict_of_scenes = dataloader_obj.get_dict_of_scenes(dir_path)
        if dict_of_scenes is not None and len(dict_of_scenes) > 0:
            rgb_imgs, xyz_imgs, seg_masks, label_imgs, fg_masks, file_names = get_segmented_masks_from_rgbd(dict_of_scenes, training_weights_checkpoint_dir)          
            N = rgb_imgs.shape[0]
            n += N
            eval_metrics = [None]*N
            for i in range(N):
                eval_metrics[i] = evaluation.multilabel_metrics(seg_masks[i,...], label_imgs[i])
            for key, _ in av_eval_metrics.items():
                for cur_img_metric in eval_metrics:
                    av_eval_metrics[key] += cur_img_metric[key]
                    
            # free memory
            del rgb_imgs, xyz_imgs, seg_masks, label_imgs, fg_masks, file_names
        torch.cuda.empty_cache()
    
    return av_eval_metrics, n

def get_segmented_pcds_from_image(pcd_file_path, training_weights_checkpoint_dir):
    """
    Returns segmented object point clouds of the given point cloud image path
    
    Args:
        pcd_file_path: input cluttered scene point cloud path in .pcd format
        training_weights_checkpoint_dir:  path to the trained weights
        
    Returns:
        segmented_pcds: list of point clouds where each point cloud is in '.pcd' format and its length = number of predicted objects
    """    
    pcd_file_name = pcd_file_path[pcd_file_path.rfind('/')+1:-4]
    dict_of_scenes1 = {}
    temp_np_array_dict = {}

    dataloader_obj = DataLoaderOCID()

    # dictionary creation step
    temp_pcd = o3d.io.read_point_cloud(pcd_file_path, remove_nan_points=False)
    temp_np_array_dict['rgb'] = dataloader_obj.get_rgb_from_pcd(temp_pcd)
    temp_xyz_array = np.asarray(temp_pcd.points)
    temp_xyz_array = temp_xyz_array.reshape((temp_np_array_dict['rgb'].shape[0], temp_np_array_dict['rgb'].shape[1], 3))
    temp_np_array_dict['xyz'] = dataloader_obj.get_xyz_from_pcd(temp_pcd)
    temp_np_array_dict['depth'] = temp_np_array_dict['xyz'][:,:,2]
    dict_of_scenes1[pcd_file_name] = temp_np_array_dict
    temp_np_array_dict['label'] = np.zeros_like(temp_np_array_dict['xyz'][:,:,0])

    rgb_imgs1, xyz_imgs1, seg_masks1, label_imgs1, fg_masks1, file_names1 = get_segmented_masks_from_rgbd(dict_of_scenes1, training_weights_checkpoint_dir)
    visualize_predicted_and_true_segment_masks(rgb_imgs1, xyz_imgs1, seg_masks1, label_imgs1)

    segmented_pcds = get_segmented_point_clouds(seg_masks1[0], temp_np_array_dict['depth'])
    return segmented_pcds