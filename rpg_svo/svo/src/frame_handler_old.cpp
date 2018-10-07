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
#include <svo/frame_handler_monodepth.h>
#include <svo/map.h>
#include <svo/frame.h>
#include <svo/feature.h>
#include <svo/point.h>
#include <svo/pose_optimizer.h>
#include <svo/sparse_img_align.h>
#include <vikit/performance_monitor.h>
#include <vikit/params_helper.h>
#include <vikit/timer.h>
#include <svo/depth_filter.h>
#ifdef USE_BUNDLE_ADJUSTMENT
#include <svo/bundle_adjustment.h>
#endif

namespace svo {

FrameHandlerMonodepth::FrameHandlerMonodepth(vk::AbstractCamera* cam) :
  FrameHandlerBase(),
  cam_(cam),
  reprojector_(cam_, map_),
  depth_filter_(NULL),
  min_num_features_kf_(vk::getParam<int>("svo/min_num_features_kf", 100)),
  sparse_img_align_iter_(vk::getParam<int>("svo/sparse_img_align_iter", 30)),
  core_n_kfs_df_(vk::getParam<int>("svo/core_n_kfs_df", 5)),
  cnn_lo_near_fts_(vk::getParam<float>("svo/cnn_lo_near_fts", 1.0f)),
  cnn_hi_near_fts_(vk::getParam<float>("svo/cnn_hi_near_fts", 30.0f)),
  cnn_max_depth_(vk::getParam<float>("svo/cnn_max_depth", 80.0f)),
  motion_model_(vk::getParam<bool>("svo/use_motion_model", false)),
  use_sparse_alignment_(false),
  use_timer_(true), // set true to show the elapsed time
  optical_flow_(vk::getParam<bool>("svo/use_optical_flow", false))
{
  // Set initial velocity
  velocity_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
  last_T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
  img_width_ = cam_->width();
  img_height_ = cam_->height();
  cnn_vo_init_.setCamModel(cam);
  cnn_vo_init_.setCNNMaxDepth(cnn_max_depth_);
  cnn_vo_init_.setImgSize(img_width_, img_height_);

  // Create a timer to get the runtime of a single loop
  t_ = new vk::Timer(true); // true to start and stop the timer manually

  kfs_processed_ = 0;
  initialize();
}

void FrameHandlerMonodepth::initialize()
{
  feature_detection::DetectorPtr feature_detector(
      new feature_detection::FastDetector(
          cam_->width(), cam_->height(), Config::gridSize(), Config::nPyrLevels()));
  DepthFilter::callback_t depth_filter_cb = boost::bind(
      &MapPointCandidates::newCandidatePoint, &map_.point_candidates_, _1, _2);
  depth_filter_ = new DepthFilter(feature_detector, depth_filter_cb,
                                  cnn_max_depth_, cam_->width(), cam_->height(),
                                  core_n_kfs_df_);
  depth_filter_->startThread();
}

FrameHandlerMonodepth::~FrameHandlerMonodepth()
{
  delete depth_filter_;
  delete t_;
}

void FrameHandlerMonodepth::addImage(const cv::Mat& img, const cv::Mat& depthmap, double timestamp)
{
  if(!startFrameProcessingCommon(timestamp))
    return;

  if (use_timer_) t_->restart();

  // SE3 velocity_opticalflow;

  // some cleanup from last iteration, can't do before because of visualization
  core_kfs_.clear();
  overlap_kfs_.clear();

  // create new frame
  SVO_START_TIMER("pyramid_creation");
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  buffer_frame_.reset(new Frame(new_frame_)); // deep copy the frame, just in case we need it for depth filtering (backward)
  depthmap_ = depthmap.clone();
  SVO_STOP_TIMER("pyramid_creation");

  UpdateResult res = RESULT_FAILURE;
  if(stage_ == STAGE_DEFAULT_FRAME)
  {
    std::cout << "stage default frame" << std::endl;
    // Update the velocity with motion prior (optical flow)
    if (motion_model_ && optical_flow_)
    {
      // Store the velocity before any processing
      velocity_old_ = velocity_;

      // Apply motion prior from optical flow (only if there is keyframe)
      FramePtr last_kf = map_.lastKeyframe();
      pose_optimizer::computeMotionPrior(cnn_lo_near_fts_,
                                         cnn_hi_near_fts_,
                                         last_kf,
                                         last_frame_,
                                         new_frame_,
                                         velocity_);
      // temporarily commented
      std::cout << "Velocity (motion prior):" << std::endl;
      std::cout << velocity_.so3() << " " << velocity_.translation().transpose() << std::endl;
    }

    std::cout << "stage default frame" << std::endl;
    // std::cout << "Pose (stage default frame):" << std::endl;
    // std::cout << last_frame_->T_f_w_.so3() << " " << last_frame_->T_f_w_.translation().transpose() << std::endl;
    res = processFrame();

    // Compute the velocity (if set true)
    if (motion_model_ && res != RESULT_FAILURE)
    {
      velocity_ = new_frame_->T_f_w_ * last_frame_->T_f_w_.inverse();
      // temporarily commented
      std::cout << "Velocity (stage default frame):" << std::endl;
      std::cout << velocity_.so3() << " " << velocity_.translation().transpose() << std::endl;

      if (optical_flow_)
      {
        // Check if there is a jump in velocity
        Vector6d vw_old = velocity_old_.log();
        Vector6d vw_new = velocity_.log();
        float diff = (vw_old-vw_new).norm();
        if (diff > 0.3)
          velocity_ = velocity_old_;
      }
    }
  }
  else if(stage_ == STAGE_SECOND_FRAME)
  {
    std::cout << "stage second frame" << std::endl;
    res = processSecondFrame();

    // Reset the last frame to world frame after obtaining the relative transformation
    last_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());

    // Compute the velocity (if set true)
    // if (motion_model_)
    // {
    //   velocity_ = new_frame_->T_f_w_ * last_frame_->T_f_w_.inverse();
    // }
    // new_frame_->T_f_w_ = velocity_*new_frame_->T_f_w_;
  }
  else if(stage_ == STAGE_FIRST_FRAME)
  {
    std::cout << "stage first frame" << std::endl;
    res = processFirstFrame();
  }

  else if(stage_ == STAGE_RELOCALIZING)
  {
    std::cout << "stage relocalizing" << std::endl;
    std::cout << "last_T_f_w_:" << std::endl;
    std::cout << last_T_f_w_.so3() << " " << last_T_f_w_.translation().transpose() << std::endl;

    std::cout << "Velocity(relocalize):" << std::endl;
    std::cout << velocity_.so3() << " " << velocity_.translation().transpose() << std::endl;
    // Set the pose of the last frame as the last tracked pose plus the velocity_
    last_frame_->T_f_w_ = velocity_ * last_T_f_w_;

    // Extract features and add back-project points (for sparse_img_align)
    feature_detection::FastDetector detector(
        last_frame_->img().cols, last_frame_->img().rows, Config::gridSize(), Config::nPyrLevels());
    detector.detect(last_frame_.get(), last_frame_->img_pyr_, Config::triangMinCornerScore(), last_frame_->fts_);
    std::for_each(last_frame_->fts_.begin(), last_frame_->fts_.end(), [&](Feature* ftr){
      int offset = (int)ftr->px[1] * img_width_ + (int)ftr->px[0];

      if (offset < img_width_ * img_height_) // to make sure that the pointer is not pointing outside the depthmap
      {
        float* depthmap_ptr = (float*)last_depthmap_.data + offset;
        float d = *depthmap_ptr;

        Eigen::Vector3d pt_pos_cur;
        Vector2d px(ftr->px[0], ftr->px[1]);
        // d = min(d, cnn_max_depth_);
        pt_pos_cur = ftr->f * d;
        Eigen::Vector3d pt_pos_w = last_frame_->T_f_w_.inverse()*pt_pos_cur;
        svo::Point* pt = new svo::Point(pt_pos_w, ftr);
        ftr->point = pt;
      }
    });

    if (last_frame_->fts_.size() > Config::initMinTracked())
    {
      // Set the last tracked frame as keyframe
      last_frame_->setKeyframe();
      map_.addKeyframe(last_frame_);
      res = relocalizeFrame(SE3(Matrix3d::Identity(), Vector3d::Zero()), last_frame_);
    }
    else
    {
      new_frame_->T_f_w_ = velocity_ * last_T_f_w_; // you need to update the new_frame to observe the frame update in rviz
    }

    // Store the last tracked frame
    last_T_f_w_ = last_frame_->T_f_w_; // new_frame_->T_f_w_;
  }

  // set last frame
  last_frame_ = new_frame_;
  last_depthmap_ = depthmap_.clone(); //store it (will need it if the tracking is lost)
  new_frame_.reset();

  // finish processing
  finishFrameProcessingCommon(last_frame_->id_, res, last_frame_->nObs());
  if (use_timer_)
  {
    double elapsed = t_->stop();
    std::cout << "Elapsed time per loop (ms): " << elapsed*1000 << ", (fps) "
              << 1.0 / elapsed << std::endl;
  }
}

FrameHandlerMonodepth::UpdateResult FrameHandlerMonodepth::processFirstFrame()
{
  new_frame_->T_f_w_ = SE3(Matrix3d::Identity(), Vector3d::Zero());
  if(cnn_vo_init_.addFirstFrame(new_frame_, depthmap_) == initialization::FAILURE)
    return RESULT_NO_KEYFRAME;
  new_frame_->setKeyframe();
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_SECOND_FRAME;
  SVO_INFO_STREAM("Init: Selected first frame.");
  return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMonodepth::processSecondFrame()
{
  initialization::InitResult res = cnn_vo_init_.addSecondFrame(new_frame_, depthmap_);
  if(res == initialization::FAILURE)
    return RESULT_FAILURE;
  else if(res == initialization::NO_KEYFRAME)
    return RESULT_NO_KEYFRAME;

  // two-frame bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  ba::twoViewBA(new_frame_.get(), map_.lastKeyframe().get(), Config::lobaThresh(), &map_);
#endif

  new_frame_->setKeyframe();
  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min); //# look into this
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min, depthmap_); //# look into this

  // add frame to map
  map_.addKeyframe(new_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
  cnn_vo_init_.reset();
  SVO_INFO_STREAM("Init: Selected second frame, triangulated initial map.");

  return RESULT_IS_KEYFRAME;
}

FrameHandlerBase::UpdateResult FrameHandlerMonodepth::processFrame()
{
  // Set initial pose TODO use prior (DONE)
  new_frame_->T_f_w_ = last_frame_->T_f_w_;

  if (motion_model_)
  {
    new_frame_->T_f_w_ = velocity_*new_frame_->T_f_w_;
  }

  if (stage_ != STAGE_RELOCALIZING)
  {
    // Save the new_frame pose as last tracked pose, so that we can recover the tracking from this pose
    last_T_f_w_ = new_frame_->T_f_w_;
  }

  // sparse image align
  if (use_sparse_alignment_)
  {
    SVO_START_TIMER("sparse_img_align");
    SparseImgAlign img_align(Config::kltMaxLevel(), Config::kltMinLevel(),
                             sparse_img_align_iter_, SparseImgAlign::GaussNewton, false, false); //PROBLEM!!!!
    size_t img_align_n_tracked = img_align.run(last_frame_, new_frame_);
    // std::cout << "img_align_n_tracked: " << img_align_n_tracked << std::endl;
    SVO_STOP_TIMER("sparse_img_align");
    SVO_LOG(img_align_n_tracked);
    SVO_DEBUG_STREAM("Img Align:\t Tracked = " << img_align_n_tracked);

    if (img_align_n_tracked < Config::qualityMinFts())
    {
      new_frame_->T_f_w_ = last_T_f_w_; // reset to avoid crazy pose jumps
      tracking_quality_ = TRACKING_INSUFFICIENT;
      // use_sparse_alignment_ = false; //~ disable the SparseImgAlign temporarily to avoid crazy pose jumps
      return RESULT_FAILURE;
    }
  }

  // map reprojection & feature alignment
  SVO_START_TIMER("reproject");
  reprojector_.reprojectMap(new_frame_, overlap_kfs_);
  SVO_STOP_TIMER("reproject");
  const size_t repr_n_new_references = reprojector_.n_matches_;
  const size_t repr_n_mps = reprojector_.n_trials_;
  SVO_LOG2(repr_n_mps, repr_n_new_references);
  SVO_DEBUG_STREAM("Reprojection:\t nPoints = "<<repr_n_mps<<"\t \t nMatches = "<<repr_n_new_references);
  if(repr_n_new_references < Config::qualityMinFts())
  {
    std::cout << "reprojected matches: " << repr_n_new_references << std::endl;
    SVO_WARN_STREAM_THROTTLE(1.0, "Not enough matched features.");
    new_frame_->T_f_w_ = last_T_f_w_; // reset to avoid crazy pose jumps
    tracking_quality_ = TRACKING_INSUFFICIENT;

    // disable the SparseImgAlign temporarily to avoid crazy pose jumps
    use_sparse_alignment_ = false;

    return RESULT_FAILURE;
  }

  // pose optimization
  SVO_START_TIMER("pose_optimizer");
  size_t sfba_n_edges_final;
  double sfba_thresh, sfba_error_init, sfba_error_final;
  pose_optimizer::optimizeGaussNewton(
      Config::poseOptimThresh(), Config::poseOptimNumIter(), false,
      new_frame_, sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_STOP_TIMER("pose_optimizer");
  SVO_LOG4(sfba_thresh, sfba_error_init, sfba_error_final, sfba_n_edges_final);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrInit = "<<sfba_error_init<<"px\t thresh = "<<sfba_thresh);
  SVO_DEBUG_STREAM("PoseOptimizer:\t ErrFin. = "<<sfba_error_final<<"px\t nObsFin. = "<<sfba_n_edges_final);
  if(sfba_n_edges_final < 20)
  {
    // use_sparse_alignment_ = false; //~ disable the SparseImgAlign temporarily to avoid crazy pose jumps
    new_frame_->T_f_w_ = last_T_f_w_;
    return RESULT_FAILURE;
  }

  // structure optimization
  SVO_START_TIMER("point_optimizer");
  optimizeStructure(new_frame_, Config::structureOptimMaxPts(), Config::structureOptimNumIter());
  SVO_STOP_TIMER("point_optimizer");

  // select keyframe
  core_kfs_.insert(new_frame_);
  setTrackingQuality(sfba_n_edges_final);
  if(tracking_quality_ == TRACKING_INSUFFICIENT)
  {
    // use_sparse_alignment_ = false; // ~disable the SparseImgAlign temporarily to avoid crazy pose jumps
    new_frame_->T_f_w_ = last_T_f_w_; // reset to avoid crazy pose jumps (should we still keep this?)
    return RESULT_FAILURE;
  }

  // use SparseImgAlign if everything is alright
  use_sparse_alignment_ = true;

  double depth_mean, depth_min;
  frame_utils::getSceneDepth(*new_frame_, depth_mean, depth_min);
  if((!needNewKf(depth_mean) || tracking_quality_ == TRACKING_BAD) && repr_n_new_references > (unsigned)min_num_features_kf_)
  {
    depth_filter_->addFrame(new_frame_);
    std::cout << "refining depth filter " << std::endl;
    return RESULT_NO_KEYFRAME;
  }
  new_frame_->setKeyframe();

  SVO_DEBUG_STREAM("New keyframe selected.");
  std::cout << "Added new keyframe " << std::endl;

  for(Features::iterator it=new_frame_->fts_.begin(); it!=new_frame_->fts_.end(); ++it)
    if((*it)->point != NULL)
    {
      (*it)->point->addFrameRef(*it);
    }
  map_.point_candidates_.addCandidatePointToFrame(new_frame_);

  // optional bundle adjustment
#ifdef USE_BUNDLE_ADJUSTMENT
  if(Config::lobaNumIter() > 0)
  {
    SVO_START_TIMER("local_ba");
    setCoreKfs(Config::coreNKfs());
    // std::cout << "Number of core keyframes: " << core_kfs_.size() << std::endl;
    size_t loba_n_erredges_init, loba_n_erredges_fin;
    double loba_err_init, loba_err_fin;
    ba::localBA(new_frame_.get(), &core_kfs_, &map_,
                loba_n_erredges_init, loba_n_erredges_fin,
                loba_err_init, loba_err_fin);
    SVO_STOP_TIMER("local_ba");
    SVO_LOG4(loba_n_erredges_init, loba_n_erredges_fin, loba_err_init, loba_err_fin);
    SVO_DEBUG_STREAM("Local BA:\t RemovedEdges {"<<loba_n_erredges_init<<", "<<loba_n_erredges_fin<<"} \t "
                     "Error {"<<loba_err_init<<", "<<loba_err_fin<<"}");
  }
#endif

  // init new depth-filters
  depth_filter_->addKeyframe(new_frame_, depth_mean, 0.5*depth_min, depthmap_);

  // add previous keyframes to depth filter
  if ((unsigned)kfs_processed_ > core_n_kfs_df_)
  {
    // std::cout << "Poses for previous keyframes: ";
    for (size_t i=0; i<core_n_kfs_df_; i++)
    {
      // std::cout << core_kfs_df_ref_[i]->T_f_w_ << " ";
      core_kfs_df_[i]->T_f_w_ = core_kfs_df_ref_[i]->T_f_w_; // get the pose from the stored pointers
      core_kfs_df_[i]->setNotKeyframe(); // need to set as non-keyframe for depth filtering
      depth_filter_->addFrame(core_kfs_df_[i]);
    }
    // std::cout << std::endl;
  }
  map_.point_candidates_.addCandidatePointToFrame(new_frame_);

  // if limited number of keyframes, remove the one furthest apart (has no effect on our system, because we keep all keyframes)
  if(Config::maxNKfs() > 2 && map_.size() >= Config::maxNKfs())
  {
    FramePtr furthest_frame = map_.getFurthestKeyframe(new_frame_->pos());
    depth_filter_->removeKeyframe(furthest_frame); // TODO this interrupts the mapper thread, maybe we can solve this better
    map_.safeDeleteFrame(furthest_frame);
  }

  // add keyframe to map
  map_.addKeyframe(new_frame_);

  // use SparseImgAlign if everything is alright
  // use_sparse_alignment_ = true;

  // add keyframe to the queue, and remove the oldest keyframe from the queue. Fixed size sliding window
  core_kfs_df_.push_back(buffer_frame_);
  if (core_kfs_df_.size() > core_n_kfs_df_)
    core_kfs_df_.pop_front();

  core_kfs_df_ref_.push_back(new_frame_);
  if (core_kfs_df_ref_.size() > core_n_kfs_df_)
    core_kfs_df_ref_.pop_front();

  kfs_processed_ += 1;

  return RESULT_IS_KEYFRAME;
}

FrameHandlerMonodepth::UpdateResult FrameHandlerMonodepth::relocalizeFrame(
    const SE3& T_cur_ref,
    FramePtr ref_frame)
{
  SVO_WARN_STREAM_THROTTLE(1.0, "Relocalizing frame");
  if(ref_frame == nullptr)
  {
    SVO_INFO_STREAM("No reference keyframe for relocalizing.");
    return RESULT_FAILURE;
  }

  // Set pose prior to new frame
  new_frame_->T_f_w_ = ref_frame->T_f_w_;

  FrameHandlerMonodepth::UpdateResult res = processFrame();
  if(res != RESULT_FAILURE)
  {
    stage_ = STAGE_DEFAULT_FRAME;
    SVO_INFO_STREAM("Relocalization successful.");
    return res;
  }
  else
    new_frame_->T_f_w_ = ref_frame->T_f_w_; // move forward until the frame is successfully re-initialized

  return RESULT_FAILURE;
}

bool FrameHandlerMonodepth::relocalizeFrameAtPose(
    const int keyframe_id,
    const SE3& T_f_kf,
    const cv::Mat& img,
    const double timestamp)
{
  FramePtr ref_keyframe;
  if(!map_.getKeyframeById(keyframe_id, ref_keyframe))
    return false;
  new_frame_.reset(new Frame(cam_, img.clone(), timestamp));
  UpdateResult res = relocalizeFrame(T_f_kf, ref_keyframe);
  if(res != RESULT_FAILURE) {
    last_frame_ = new_frame_;
    return true;
  }
  return false;
}

void FrameHandlerMonodepth::resetAll()
{
  resetCommon();
  last_frame_.reset();
  new_frame_.reset();
  core_kfs_.clear();
  // core_kfs_df_.clear();
  overlap_kfs_.clear();
  depth_filter_->reset();
}

void FrameHandlerMonodepth::setFirstFrame(const FramePtr& first_frame)
{
  resetAll();
  last_frame_ = first_frame;
  last_frame_->setKeyframe();
  map_.addKeyframe(last_frame_);
  stage_ = STAGE_DEFAULT_FRAME;
}

bool FrameHandlerMonodepth::needNewKf(double scene_depth_mean)
{
  // // if number of tracked point is much lower than the number of corners can be created, create a new keyframe (happens potentially in large rotation condition)
  // feature_detection::FastDetector detector_1pyr(
  //     frame->img().cols, frame->img().rows, Config::gridSize(), 1);

  for(auto it=overlap_kfs_.begin(), ite=overlap_kfs_.end(); it!=ite; ++it)
  {
    Vector3d relpos = new_frame_->w2f(it->first->pos());
    if(fabs(relpos.x())/scene_depth_mean < Config::kfSelectMinDist() &&
       fabs(relpos.y())/scene_depth_mean < Config::kfSelectMinDist()*0.8 &&
       fabs(relpos.z())/scene_depth_mean < Config::kfSelectMinDist()*1.3)
      return false;
  }
  return true;
}

void FrameHandlerMonodepth::setCoreKfs(size_t n_closest)
{
  size_t n = min(n_closest, overlap_kfs_.size()-1);
  std::partial_sort(overlap_kfs_.begin(), overlap_kfs_.begin()+n, overlap_kfs_.end(),
                    boost::bind(&pair<FramePtr, size_t>::second, _1) >
                    boost::bind(&pair<FramePtr, size_t>::second, _2));
  std::for_each(overlap_kfs_.begin(), overlap_kfs_.end(), [&](pair<FramePtr,size_t>& i){ core_kfs_.insert(i.first); });
}

} // namespace svo
