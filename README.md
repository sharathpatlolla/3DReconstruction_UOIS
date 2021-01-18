# 3D Reconstruction and UOIS

### Required:
1. GPU access 

### Overview:
This code can be used to obtain segmented point clouds of individual objects, given a cluttered scene. The code is ready to use if the cluttered image data is in point cloud (.pcd) format else some preprocessing might be required. It uses UOIS implementation ([link](https://github.com/chrisdxie/uois)) for getting segmented masks and PointSDF reconstruction ([link](https://github.com/mvandermerwe/PointSDF)) for surface reconstruction of partial object point clouds. 

The idea is to segment a cluttered scene and do data processing on segmentation masks to obtain individual objects in point cloud format (`.pcd`). The outputs are partial object point cloud and PointSDF reconstruction is used on each partial object point cloud to get reconstructed objects(`.obj`). Project report can be found [here]( https://drive.google.com/file/d/19dobBmPl16D2LgDXhVS3aF1PgzkCV4Xw/view?usp=sharing)

### Environment:                                          
Create two separate conda environments using the files below \
Use `uois_env_ubuntu.yml` file to create the anaconda environment for segmentation \
Use `recon_env_ubuntu.yml` file to create the anacoda environment for reconstruction

### Dataset:
The dataset used is [OCID dataset](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/) that can be downloaded from [here](https://data.acin.tuwien.ac.at/index.php/s/g3EkcgcPioolQmJ).

Download this dataset, create a `datasets` directory in the working directory and put the downloaded dataset in `datasets` directory

### Dependencies:
[UOIS implementation](https://github.com/chrisdxie/uois) is used for segmentation and [pointSDF reconstruction](https://github.com/mvandermerwe/PointSDF) is used for 3D surface reconstruction

Run `setup.sh` to clone those repositories and put them in `dependencies` directory. Go through both of the repository `README.md` files to include all the internal dependencies. 

### Models:
The trained weights are taken from the original UOIS implementation paper which can download from [here](https://drive.google.com/uc?export=download&id=1CZgHk5VfhfvUosb8xlgzg7aKhprLpGC3). 

Download the weights, create a `checkpoints` directory in the working directory and put the downloaded weights in `checkpoints` directory

### Usage:
Activate the created conda environment `uois-recon` before using the segmentation code. Segmented object point clouds can be obtained using the functions in `segmentation_util.py`. Refer to `segmentation_sample_code.ipynb` for examples on using those functions 

Activate the created conda environment 'recon_new` before using the segmentation code. Surface reconstruction API I is in `reconstruction_util.py`. Refer to `reconstruction_sample_code.ipynb` for examples on obtaining a reconstructed surface from a given point cloud.

Try changing relative paths to absolute path if there is an issue with the paths

### Future Work/To-Do:
There are numerous to-do tasks (because of the time constraints :P), I've listed some of them below
1. Rewrite reconstruction in PyTorch or segmentation in TensorFlow and refactor. 
2. Include sample code for segmentation training on a sample OCID dataset. (This is a small task, refer to [link](https://github.com/chrisdxie/uois/blob/master/train_DSN.ipynb))
3. The segmented point clouds can preprocessed to remove the outlier points which are a hindrance to surface reconstruction. (refer to the third reconstructed image in Figure 5 of the project [report](https://drive.google.com/file/d/19dobBmPl16D2LgDXhVS3aF1PgzkCV4Xw/view?usp=sharing))

### References
 1. Xie, Christopher, et al. "Unseen object instance segmentation for robotic environments." arXiv preprint arXiv:2007.08073 (2020).
 2. Van der Merwe, Mark, et al. "Learning Continuous 3D Reconstructions for Geometrically Aware Grasping." arXiv preprint arXiv:1910.00983 (2019).
 3. Suchi, Markus, et al. "EasyLabel: A semi-automatic pixel-wise object annotation tool for creating robotic RGB-D datasets." 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019.
