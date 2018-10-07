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

#ifndef SVO_INITIALIZATION_H
#define SVO_INITIALIZATION_H

#include <svo/global.h>
#include <vikit/abstract_camera.h>

namespace svo {

class FrameHandlerMonodepth;
class FrameHandlerOnline;

/// Bootstrapping the map from the first two views.
namespace initialization {

enum InitResult { FAILURE, NO_KEYFRAME, SUCCESS };

/// Tracks features using Lucas-Kanade tracker and then estimates a homography.
class CNNVOInit {
  friend class svo::FrameHandlerMonodepth;
  friend class svo::FrameHandlerOnline;
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FramePtr frame_ref_;

  CNNVOInit() {};
  ~CNNVOInit() {};
  InitResult addFirstFrame(FramePtr frame_ref, const cv::Mat& depth);
  InitResult addSecondFrame(FramePtr frame_ref, const cv::Mat& depth);
  void reset();
  void setCamModel(vk::AbstractCamera* cam);
  void setImgSize(const int& width, const int& height);

  vector<cv::Point2f> getPxRef() const { return px_ref_; };
  vector<cv::Point2f> getPxCur() const { return px_cur_; };

protected:
  vk::AbstractCamera* cam_;         //!< Camera model, can be ATAN, Pinhole or Ocam (see vikit).
  vector<cv::Point2f> px_ref_;      //!< keypoints to be tracked in reference frame.
  vector<cv::Point2f> px_cur_;      //!< tracked keypoints in current frame.
  vector<Vector3d> f_ref_;          //!< bearing vectors corresponding to the keypoints in the reference image.
  vector<Vector3d> f_cur_;          //!< bearing vectors corresponding to the keypoints in the current image.
  vector<double> disparities_;      //!< disparity between first and second frame.
  vector<int> inliers_;             //!< inliers after the geometric check (e.g., Homography).
  vector<Vector3d> xyz_in_cur_;     //!< 3D points computed during the geometric check.
  SE3 T_cur_from_ref_;              //!< computed transformation between the first two frames.
  cv::Mat depth_kf_;                //!< Store the keyframe depthmap (from CNN)
  int img_width_;                   //!< Used to compute the pointer to the depthmap
  int img_height_;                  //!< Used to compute the pointer to the depthmap
};

/// Detect Fast corners in the image.
void detectFeatures2(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec,
    const cv::Mat& depthmap,
    vector<Vector3d>& xyz_in_cur,
    const int& img_width,
    const int& img_height);

/// Compute optical flow (Lucas Kanade) for selected keypoints.
void alignFeatures(
    FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities,
    SE3& T_cur_from_ref);

} // namespace initialization
} // namespace svo

#endif // SVO_INITIALIZATION_H
