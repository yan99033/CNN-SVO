// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
#include <ros/package.h>
#include <string>
#include <svo/frame_handler_online.h>
#include <svo/map.h>
#include <svo/config.h>
#include <svo_ros/visualizer_online.h>
#include <svo_ros/read_npy.h>
#include <vikit/params_helper.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/String.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <boost/thread.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>
#include <vikit/abstract_camera.h>
#include <vikit/camera_loader.h>
#include <vikit/user_input_thread.h>
#include <stdlib.h>

#include <signal.h>
#include <vector>
#include <iostream>
#include <sophus/se3.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>

void myExitHandler(int s)
{
	std::cout << "Exiting" << std::endl;
	exit(1);
}

void exitThread()
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = myExitHandler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	// firstRosSpin=true;
	while(true) pause();
}

namespace svo {
/// SVO Interface
class VoNode
{
public:
  svo::FrameHandlerOnline* vo_;       	 //!< CNN-VO system
  svo::Visualizer visualizer_;           //!< visualize map, pose and images in rviz
  bool publish_markers_;                 //!< publish only the minimal amount of info (choice for embedded devices)
  bool publish_dense_input_;             //!< false to show less information
  vk::AbstractCamera* cam_;              //!< Camera model
  std::string sequence_;                 //!< Sequence number
  std::string image_folder_;             //!< Folder that stores the images
  cv::Size input_size_;                  //!< Image height and width
  int skipped_frames_;                   //!< Number of skipped frames in the beginning of the sequence
  cv::Mat image_;                        //!< input image to CNN-VO
  std::vector<double> timestamps_;       //!< preloaded timestamps and attach the timestamps to the poses
  std::string save_dir_;                 //!< Directory to save the results
	std::ofstream trace_est_pose_;				 //!< save XYZ and rotation matrix of keyframe poses
  std::ofstream trace_est_pose_ATE;			 //!< save XYZ and quaternion of keyframe poses

  VoNode();
  ~VoNode();
  int firstFrame();
  void imgCb(const int i);
  bool readImage(const int& index);
  void loadTimestamps();
  void adaptShape(cv::Mat& data);
	void tracePoses(const SE3& T_w_f, const double timestamp);
};

VoNode::VoNode() :
  vo_(NULL),
  publish_markers_(vk::getParam<bool>("svo/publish_markers", true)),
  publish_dense_input_(vk::getParam<bool>("svo/publish_dense_input", false)),
  cam_(NULL),
  sequence_(vk::getParam<std::string>("svo/sequence_path", "/default/path/to/sequence")),
  image_folder_(vk::getParam<std::string>("svo/image_folder", "default_image_folder")),
  skipped_frames_(vk::getParam<int>("svo/skipped_frames", 0)),
  save_dir_(std::getenv("HOME")+std::string("/results/CNN_VO"))
{
  // Create Camera
  if(!vk::camera_loader::loadFromRosNs("svo", cam_))
    throw std::runtime_error("Camera model not correctly specified.");

  // Get initial position and orientation
  visualizer_.T_world_from_vision_ = Sophus::SE3(
      vk::rpy2dcm(Vector3d(vk::getParam<double>("svo/init_rx", 0.0),
                           vk::getParam<double>("svo/init_ry", 0.0),
                           vk::getParam<double>("svo/init_rz", 0.0))),
      Eigen::Vector3d(vk::getParam<double>("svo/init_tx", 0.0),
                      vk::getParam<double>("svo/init_ty", 0.0),
                      vk::getParam<double>("svo/init_tz", 0.0)));

  // Create files to save keyframe poses
  int tmp =system(("mkdir " + save_dir_).c_str());
  std::string trace_est_name(save_dir_+"/traj_estimate_benchmark.txt");
  trace_est_pose_.open(trace_est_name.c_str());
  if(trace_est_pose_.fail())
    throw std::runtime_error("Could not create tracefile. Does folder exist?");

  std::string trace_est_nameATE(save_dir_+"/traj_estimate_ATE_benchmark.txt");
  trace_est_pose_ATE.open(trace_est_nameATE.c_str());
  if(trace_est_pose_ATE.fail())
    throw std::runtime_error("Could not create tracefile. Does folder exist?");

  // Get image size from the camera model
  input_size_ = cv::Size(cam_->width(), cam_->height());

  // Init VO and start
  vo_ = new svo::FrameHandlerOnline(cam_);
  vo_->start();
}

VoNode::~VoNode()
{
  delete vo_;
  delete cam_;
}

int VoNode::firstFrame()
{
  return skipped_frames_;
}

void VoNode::tracePoses(const SE3& T_w_f, const double timestamp)
{
  Quaterniond q(T_w_f.unit_quaternion());
  Vector3d p(T_w_f.translation());
  //trace_est_pose_.precision(15);
  trace_est_pose_.setf(std::ios::scientific, std::ios::floatfield );
  trace_est_pose_ << timestamp << " ";

  trace_est_pose_ATE.setf(std::ios::scientific, std::ios::floatfield );
  trace_est_pose_ATE << timestamp << " ";

  Matrix3d Rot_Matrix(T_w_f.rotation_matrix());
  trace_est_pose_ << Rot_Matrix(0,0) << " " << Rot_Matrix(0,1) << " " << Rot_Matrix(0,2) << " " << p.x()<<" "
                  << Rot_Matrix(1,0) << " " << Rot_Matrix(1,1) << " " << Rot_Matrix(1,2) << " " << p.y()<<" "
                  << Rot_Matrix(2,0) << " " << Rot_Matrix(2,1) << " " << Rot_Matrix(2,2) << " " << p.z()<< std::endl;


  trace_est_pose_ATE << p.x() << " " << p.y() << " " << p.z() << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
}

void VoNode::adaptShape(cv::Mat& data)
{
  cv::Size data_size = data.size();
  if (data_size.width == input_size_.width)
    return;
  else
  {
    if (data_size.width < input_size_.width) // need to pad zeros
    {
      int pad_size = input_size_.width - data_size.width;
      cv::copyMakeBorder(data, data, 0, 0, 0, pad_size, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    }
    else // need to slice the data
      data = data(cv::Rect(0,0,input_size_.width,input_size_.height));
  }
}

bool VoNode::readImage(const int& index)
{
  char index_buf[7]; sprintf(index_buf, "%06d", index); std::string index_s(index_buf);
  std::string img_path(sequence_ + "/"+image_folder_ +"/");
  std::string img_name(img_path + index_s + ".png");
  cv::Mat img = cv::imread(img_name);

	if(img.empty())
  {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
  }

  // process image and depth
  adaptShape(img);

	image_ = img;

  // If image has been loaded successfully
  return true;
}

void VoNode::loadTimestamps()
{
  std::ifstream tr;
  std::string timesFile = sequence_ + "/times.txt";
  tr.open(timesFile.c_str());
  while(!tr.eof() && tr.good())
  {
    std::string line;
    char buf[1000];
    tr.getline(buf, 1000);

    int id;
    double stamp;
    float exposure = 0;

    if(1==sscanf(buf, "%lf", &stamp)){
      timestamps_.push_back(stamp);
    }
  }
  tr.close();

  printf("got %d timestamps!\n", (int)timestamps_.size());
}


void VoNode::imgCb(const int i)
{
	// Read image
  readImage(i);

  vo_->addImage(image_, timestamps_[i]);

  // Visualize the output in RViz
  visualizer_.publishMinimal(image_, vo_->getKFDepth(), vo_->lastFrame(), *vo_, timestamps_[i]);

  if(publish_markers_ && vo_->stage() != FrameHandlerBase::STAGE_PAUSED)
    visualizer_.visualizeMarkers(vo_->lastFrame(), vo_->coreKeyframes(), vo_->map());

  if(publish_dense_input_)
    visualizer_.exportToDense(vo_->lastFrame());

  if(vo_->stage() == FrameHandlerOnline::STAGE_PAUSED)
    usleep(100000);
}


} // namespace svo

int main(int argc, char **argv)
{
  // hook crtl+C. TODO: Sometimes it couldn't stop the program immediately.
	boost::thread exit_thread = boost::thread(exitThread);

  ros::init(argc, argv, "svo");
  ros::NodeHandle nh;
  svo::VoNode vo_node;
  int first = vo_node.firstFrame();
  vo_node.loadTimestamps();
  for (size_t i=first;i<vo_node.timestamps_.size();i++){
    vo_node.imgCb(i);

    usleep(50000); // Increase this number to introduce delay, for slower machine
  }

	// Save all the keyframe poses after finished processing a sequence
	for(auto it=vo_node.vo_->map().keyframes_.begin(), ite=vo_node.vo_->map().keyframes_.end();it!=ite; ++it)
  {
      vo_node.tracePoses((*it)->T_f_w_.inverse(), (*it)->timestamp_);
  }

  printf("Result saved in file...");
  printf("SVO terminated.\n");
  ros::shutdown();
  return 0;
}
