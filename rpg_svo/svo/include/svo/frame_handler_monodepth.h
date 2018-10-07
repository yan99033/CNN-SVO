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

#ifndef SVO_FRAME_HANDLER_H_
#define SVO_FRAME_HANDLER_H_

#include <set>
#include <deque>
#include <vikit/abstract_camera.h>
#include <svo/frame_handler_base.h>
#include <svo/reprojector.h>
#include <svo/initialization_monodepth.h>

namespace svo {

/// Newly added monodepth mode Visual Odometry Pipeline, modified from the monocular version.
class FrameHandlerMonodepth : public FrameHandlerBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FrameHandlerMonodepth(vk::AbstractCamera* cam);
  virtual ~FrameHandlerMonodepth();

  /// Provide an image and depthmap.
  void addImage(const cv::Mat& img, const cv::Mat& depthmap, double timestamp);

  /// Set the first frame (used for synthetic datasets in benchmark node)
  void setFirstFrame(const FramePtr& first_frame);

  /// Get the last frame that has been processed.
  FramePtr lastFrame() { return last_frame_; }

  /// Get the depthmap of the latest keyframe.
  const cv::Mat& getKFDepth() const { return depthmap_; }

  /// Get the set of spatially closest keyframes of the last frame.
  const set<FramePtr>& coreKeyframes() { return core_kfs_; }

  /// Return the feature track to visualize the KLT tracking during initialization.
  const vector<cv::Point2f>& initFeatureTrackRefPx() const { return cnn_vo_init_.px_ref_; }
  const vector<cv::Point2f>& initFeatureTrackCurPx() const { return cnn_vo_init_.px_cur_; }

  /// Access the depth filter.
  DepthFilter* depthFilter() const { return depth_filter_; }

  /// An external place recognition module may know where to relocalize.
  bool relocalizeFrameAtPose(
      const int keyframe_id,
      const SE3& T_kf_f,
      const cv::Mat& img,
      const double timestamp);

protected:
  vk::AbstractCamera* cam_;                     //!< Camera model, can be ATAN, Pinhole or Ocam (see vikit).
  Reprojector reprojector_;                     //!< Projects points from other keyframes into the current frame
  FramePtr new_frame_;                          //!< Current frame.
  FramePtr last_frame_;                         //!< Last frame, not necessarily a keyframe.
  set<FramePtr> core_kfs_;                      //!< Keyframes in the closer neighbourhood.
  FramePtr buffer_frame_;                       //!< Save the last n keyframes for depth filtering
  deque<FramePtr> core_kfs_df_;                 //!< Fixed number of keyframes (sliding window) for depth filter
  deque<FramePtr> core_kfs_df_ref_;             //!< Reference pointers for previous keyframes (core_kfs_df_), to get the poses of the frame
  vector< pair<FramePtr,size_t> > overlap_kfs_; //!< All keyframes with overlapping field of view. the paired number specifies how many common mappoints are observed TODO: why vector!?
  initialization::CNNVOInit cnn_vo_init_;       //!< Used to estimate pose of the first two keyframes by estimating a homography.
  DepthFilter* depth_filter_;                   //!< Depth estimation algorithm runs in a parallel thread and is used to initialize new 3D points.
  cv::Mat depthmap_;                            //!< Store the incoming depthmap
  cv::Mat last_depthmap_;                       //!< Keep the last depth map in case of tracking lost
  int kfs_processed_;                           //!< Counter of processed keyframes
  int img_width_;                               //!< Used to compute the pointer to the depthmap
  int img_height_;                              //!< Used to compute the pointer to the depthmap
  int min_num_features_kf_;                     //!< if the number of tracked feature dropped under the defined value, create a new keyframe
  int sparse_img_align_iter_;                   //!< Number of iterations for sparse image alignment (may need to lower the number for real time application)
  size_t core_n_kfs_df_;                        //!< Number of previous keyframes used for depth filtering
  bool motion_model_;                           //!< Assume that the camera is having constant velocity
  bool use_sparse_alignment_;                   //!< For some reason the SparseImgAlign is not working in the first set of triangulated 3d points, we need to disable it in the beginning
  SE3 last_T_f_w_;                              //!< Use it in case the tracking is lost
  SE3 velocity_;                                //!< constant velocity model
  SE3 velocity_old_;                            //!< If the velocity has changed dramatically, use the old velocity

  /// Initialize the visual odometry algorithm.
  virtual void initialize();

  /// Processes the first frame and sets it as a keyframe.
  virtual UpdateResult processFirstFrame();

  /// Processes all frames after the first frame until a keyframe is selected.
  virtual UpdateResult processSecondFrame();

  /// Processes all frames after the first two keyframes.
  virtual UpdateResult processFrame();

  /// Try relocalizing the frame at relative position to provided keyframe.
  virtual UpdateResult relocalizeFrame(
      const SE3& T_cur_ref,
      FramePtr ref_keyframe);

  /// Reset the frame handler. Implement in derived class.
  virtual void resetAll();

  /// Keyframe selection criterion.
  virtual bool needNewKf(double scene_depth_mean);

  void setCoreKfs(size_t n_closest);
};

} // namespace svo

#endif // SVO_FRAME_HANDLER_H_
