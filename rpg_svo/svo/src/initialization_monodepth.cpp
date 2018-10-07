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

#include <svo/config.h>
#include <svo/frame.h>
#include <svo/point.h>
#include <svo/feature.h>
#include <svo/sparse_img_align.h>
#include <svo/initialization_monodepth.h>
#include <svo/feature_detection.h>
#include <vikit/math_utils.h>
#include <vikit/homography.h>

namespace svo {
namespace initialization {

InitResult CNNVOInit::addFirstFrame(FramePtr frame_ref, const cv::Mat& depth)
{
  reset();
  // store the depthmap (from CNN)
  depth_kf_ = depth;

  detectFeatures2(frame_ref, px_ref_, f_ref_, depth_kf_, xyz_in_cur_, img_width_, img_height_);

  if(px_ref_.size() < 100)
  {
    SVO_WARN_STREAM_THROTTLE(2.0, "First image has less than 100 features. Retry in more textured environment.");
    return FAILURE;
  }
  frame_ref_ = frame_ref;
  px_cur_.insert(px_cur_.begin(), px_ref_.begin(), px_ref_.end()); // copy the points in reference frame to the tracked point in current frame
  return SUCCESS;
}

InitResult CNNVOInit::addSecondFrame(FramePtr frame_cur, const cv::Mat& depth)
{
  // Store the depth map
  depth_kf_ = depth;

  alignFeatures(frame_ref_, frame_cur, px_ref_, px_cur_, f_ref_, f_cur_, disparities_, T_cur_from_ref_);
  SVO_INFO_STREAM("Init: CNNVO tracked "<< disparities_.size() <<" features");

  if(disparities_.size() < Config::initMinTracked())
  {
    std::cout << "disparities size is smaller than the minimum tracked feature" << std::endl;
    return FAILURE;
  }

  double disparity = vk::getMedian(disparities_);
  SVO_INFO_STREAM("Init: CNN "<<disparity<<"px average disparity.");
  if(disparity < Config::initMinDisparity())
    return NO_KEYFRAME;

  frame_cur->T_f_w_ = T_cur_from_ref_ * frame_ref_->T_f_w_; //incremental pose

  // For each inlier create 3D point and add feature in both frames
  SE3 T_world_cur = frame_cur->T_f_w_.inverse();

  vector<cv::Point2f>::iterator px_ref_it = px_ref_.begin();
  vector<cv::Point2f>::iterator px_cur_it = px_cur_.begin();
  vector<Vector3d>::iterator xyz_in_cur_it = xyz_in_cur_.begin();
  vector<Vector3d>::iterator f_cur_it = f_cur_.begin();
  vector<Vector3d>::iterator f_ref_it = f_ref_.begin();

  for(size_t i=0; px_ref_it != px_ref_.end(); ++i)
  {
    Vector2d px_cur((*px_cur_it).x, (*px_cur_it).y);
    Vector2d px_ref((*px_ref_it).x, (*px_ref_it).y);

    if ((*xyz_in_cur_it).z() > 0) // we only add the points if the points is still visible in the next frame
    {
      Vector3d pos = T_world_cur * (*xyz_in_cur_it); // *xyz_in_cur_it; //
      Point* new_point = new Point(pos);

      Feature* ftr_cur(new Feature(frame_cur.get(), new_point, px_cur, *f_cur_it, 0));
      frame_cur->addFeature(ftr_cur);
      new_point->addFrameRef(ftr_cur);

      Feature* ftr_ref(new Feature(frame_ref_.get(), new_point, px_ref, *f_ref_it, 0));
      frame_ref_->addFeature(ftr_ref);
      new_point->addFrameRef(ftr_ref);
    }

    ++px_ref_it;
    ++px_cur_it;
    ++xyz_in_cur_it;
    ++f_cur_it;
    ++f_ref_it;
  }

  return SUCCESS;
}

void CNNVOInit::reset()
{
  px_cur_.clear();
  frame_ref_.reset();
}

void CNNVOInit::setCamModel(vk::AbstractCamera* cam)
{
  cam_ = cam;
}

void CNNVOInit::setImgSize(const int& width, const int& height)
{
  img_width_ = width;
  img_height_ = height;
}

void detectFeatures2(
    FramePtr frame,
    vector<cv::Point2f>& px_vec,
    vector<Vector3d>& f_vec,
    const cv::Mat& depthmap,
    vector<Vector3d>& xyz_in_cur,
    const int& img_width,
    const int& img_height)
{
  // Features new_features;
  feature_detection::FastDetector detector(
      frame->img().cols, frame->img().rows, Config::gridSize(), Config::nPyrLevels());
  detector.detect(frame.get(), frame->img_pyr_, Config::triangMinCornerScore(), frame->fts_);

  // now for all maximum corners, initialize a new seed
  px_vec.clear(); px_vec.reserve(frame->fts_.size());
  f_vec.clear(); f_vec.reserve(frame->fts_.size());
  xyz_in_cur.clear(); xyz_in_cur.reserve(frame->fts_.size());

  std::for_each(frame->fts_.begin(), frame->fts_.end(), [&](Feature* ftr){
    int offset = (int)ftr->px[1] * img_width + (int)ftr->px[0];

    if (offset < img_width * img_height) // to make sure that the pointer is not pointing outside the depthmap
    {
      float* depthmap_ptr = (float*)depthmap.data + offset;
      float d = *depthmap_ptr;

      Eigen::Vector3d pt_pos_cur;
      px_vec.push_back(cv::Point2f(ftr->px[0], ftr->px[1]));

      pt_pos_cur = ftr->f * d;
      f_vec.push_back(pt_pos_cur);
      xyz_in_cur.push_back(pt_pos_cur);
      Eigen::Vector3d pt_pos_w = frame->T_f_w_.inverse()*pt_pos_cur;
      svo::Point* pt = new svo::Point(pt_pos_w, ftr);
      ftr->point = pt;
    }
  });
}

void alignFeatures(
    FramePtr frame_ref,
    FramePtr frame_cur,
    vector<cv::Point2f>& px_ref,
    vector<cv::Point2f>& px_cur,
    vector<Vector3d>& f_ref,
    vector<Vector3d>& f_cur,
    vector<double>& disparities,
    SE3& T_cur_from_ref)
{
  svo::SparseImgAlign img_align(svo::Config::kltMaxLevel(), svo::Config::kltMinLevel(),
                                  30, svo::SparseImgAlign::GaussNewton, false, false);
  img_align.run(frame_ref, frame_cur); // the transformation is obtained
  T_cur_from_ref = frame_cur->T_f_w_ * frame_ref->T_f_w_.inverse();

  vector<cv::Point2f>::iterator px_ref_it = px_ref.begin();
  vector<Vector3d>::iterator f_ref_it = f_ref.begin();
  px_cur.clear(); px_cur.reserve(px_ref.size());
  f_cur.clear(); f_cur.reserve(px_ref.size());
  disparities.clear(); disparities.reserve(px_cur.size());

  for(size_t i=0; px_ref_it != px_ref.end(); ++i)
  {
    bool valid = false; // change to true if reprojection is valid (Should have used the original implementation of finding valid reprojection)
    Vector3d p_cur = T_cur_from_ref * *f_ref_it;
    cv::Point2f uv;
    Feature f = Feature(frame_cur.get(), p_cur, uv, valid); // both frame_cur and frame_ref can be used, because the intrinsics are the same

    if (!valid)
    {
      px_ref_it = px_ref.erase(px_ref_it);
      f_ref_it = f_ref.erase(f_ref_it);
      continue;
    }
    px_cur.push_back(uv);
    f_cur.push_back(p_cur);
    disparities.push_back(Vector2d(px_ref_it->x - uv.x, px_ref_it->y - uv.y).norm());
    ++px_ref_it;
    ++f_ref_it;
  }
}

} // namespace initialization
} // namespace svo
