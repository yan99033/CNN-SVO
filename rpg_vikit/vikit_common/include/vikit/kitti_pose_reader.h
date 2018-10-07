/**
 *  This file is a newly added file to read KITTI odometry ground truth for evaluation
 */

#ifndef VIKIT_KITTI_POSE_READER_H_
#define VIKIT_KITTI_POSE_READER_H_

#include <fstream>
#include <vector>
#include <string>
#include <vikit/math_utils.h>

namespace vk
{

using namespace Eigen;
using namespace Sophus;

class kittiFileReader
{
public:
  kittiFileReader(const std::string& file) :
    file_(file),
    file_stream_(file.c_str())
  {
    std::string line;
    while(std::getline(file_stream_, line))
    {
        std::vector<float> row;
        std::stringstream  lineStream(line);

        float value;
        while(lineStream >> value)
        {
            row.push_back(value);
        }
        poses.push_back(row);
    }

    num_frames = poses.size();
  }

int get_size() { return num_frames; }

SE3 get_T_f_w_(int frame)
{
  std::vector<float> pose = poses[frame];
  Matrix3d R;
  R << pose[0], pose[1], pose[2],
       pose[4], pose[5], pose[6],
       pose[8], pose[9], pose[10];
  Vector3d T(pose[3], pose[7], pose[11]);

  SE3 T_f_w_(Quaterniond(R), T);

  return T_f_w_;
}


private:
  int num_frames;
  std::vector<std::vector<float>> poses; // KITTI provides flattened 3x4 matrix for each pose
  std::string file_;
  std::ifstream file_stream_;
};

} // end namespace vk

#endif // VIKIT_FILE_READER_H_
