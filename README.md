# CNN-SVO
(Top) camera frames (Bottom) depth maps of the keyframes from Monodepth
<p align="center">
 <img src="https://github.com/yan99033/CNN-SVO/blob/master/gif/kitti_preview.gif" width="723" height="224">
 <img src="https://github.com/yan99033/CNN-SVO/blob/master/gif/robotcar_preview.gif" width="723" height="224">
</p>

Demo: https://www.youtube.com/watch?v=4oTzwoby3jw

Paper: [CNN-SVO: Improving the Mapping in Semi-Direct Visual Odometry Using Single-Image Depth Prediction](https://arxiv.org/pdf/1810.01011.pdf)

This is an extension of the SVO with major improvements for the forward facing camera with the help of monocular depth estimation from CNN.

We tested this code with ROS Kinetic on Ubuntu 16.04.

There are two ways to make it work
1. [Online mode](#online-mode)
2. [Offline mode](#offline-mode)

#### Dataset
We use [KITTI Odometry sequences](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) and [Oxford Robotcar dataset](http://robotcar-dataset.robots.ox.ac.uk/datasets/) to evaluate the robustness of our method.

We provide camera parameters for both datasets. Running a sequence requires setting the directory that keeps the images (also directory to their corresponding depth maps if you are using [Offline mode](#offline-mode)).

Unfortunately, getting Oxford Robotcar dataset to work is not straightforward. Here is the rough idea of what do you need to do to preprocess Oxford Robotcar dataset images:
1. Pick a sequence to download
2. Use the SDK to get the RGB images
3. Crop the images to match the aspect ratio (1248x376) of images in KITTI dataset
4. (For running [Offline mode](#offline-mode)) Save the depth maps of those images

#### Online mode
1. Clone the [monodepth-cpp repo](https://github.com/yan99033/monodepth-cpp) and follow the instructions
2. Make sure the library is successfully built by running the *inference_monodepth* executable
3. Set the image directory in `rpg_svo/svo_ros/param/vo_kitti.yaml`
4. Make sure the images are colour (not greyscale) images

**NOTE:** If you are having difficulty compiling the library, you can still visualize the results using the [Offline mode](#offline-mode)

#### Offline mode
1. Clone the [Monodepth repo](https://github.com/mrharicot/monodepth) and follow the *Testing* instructions
2. Modify the `monodepth_main.py` so that it saves the disparity maps as numpy (.npy) files in one folder (per sequence)
3. Set the image and depth directories in `rpg_svo/svo_ros/param/vo_kitti.yaml`

**NOTE:** For saving disparity map, the naming convention is `depth_x.npy`, where x is 0-based with no leading zero.

#### Instructions
1. Clone this repo
```
cd ~/catkin_ws/src
git clone https://github.com/yan99033/CNN-VO
```

2. (OPTIONAL) Clone [monodepth-cpp repo](https://github.com/yan99033/monodepth-cpp)
```
cd ~/catkin_ws/src/CNN-VO
git clone https://github.com/yan99033/monodepth-cpp
```

2. Enable/disable [Online mode](#online-mode) by toggling **TRUE/FALSE** in `svo_ros/CMakeLists.txt` and `svo/CMakeLists.txt`. Also make sure that Monodepth library and header file are linked properly, if [Online mode](#online-mode) is used. Note that [Online mode](#online-mode) is disabled by default

3. Compile the project
```
cd ~/catkin_ws
catkin_make
```

4. Launch the project
```
roscore
rosrun rviz rviz -d ~/catkin_ws/src/CNN-VO/rpg_svo/svo_ros/rviz_kitti.rviz
roslaunch svo_ros kittiOffline00-02.launch
```

5. Try another KITTI sequence
  * Set the folder image and depth directories in `rpg_svo/svo_ros/param/vo_kitti.yaml`
  * Use the corresponding roslaunch file

#### Disclaimer

The authors take no credit from [SVO](https://github.com/uzh-rpg/rpg_svo) and [Monodepth](https://github.com/mrharicot/monodepth), therefore the licenses should remain intact. Please cite their work if you find them helpful.
