#include "global_map.h"

using namespace std;
using namespace tauv_msgs;

Eigen::Vector3d point_to_vec(geometry_msgs::Point point)
{
    Eigen::Vector3d vec;
    vec<<point.x, point.y, point.z;
    return vec;
}

FeatureTracker::FeatureTracker(BucketDetection initial_detection)
{
    FeatureList= {};
    totalDetections=0;

    addFeature(initial_detection);
    tag = initial_detection.tag; //change ownership to Feature?

    //init mahalnobis distance from params
    if(!ros::param::get("/tracker_params/"+tag+"/mahalanobis_threshold", mahalanobisThreshold))
    {
        ros::param::get("/tracker_params/default/mahalanobis_threshold", mahalanobisThreshold);
    }
}

FeatureTracker::~FeatureTracker()
{
    for(Feature *F:FeatureList)
    {
        delete F;
    }
}

void FeatureTracker::addFeature(BucketDetection detection)
{
    Feature *NEW = new Feature(detection);
    FeatureList.push_back(NEW);
    totalDetections+=1;
}

void FeatureTracker::addDetection(BucketDetection detection, int featureIdx)
{
    FeatureList[featureIdx]->addDetection(detection);
    totalDetections+=1;
}

int FeatureTracker::generateSimilarityMatrix(vector<BucketDetection> &detections, vector<vector<double>> &costMatrix, size_t trackerNum, vector<pair<FeatureTracker*, int>> &trackerList)
{

    size_t featureNum = trackerNum;
    for(size_t featureIdx = 0; featureIdx<FeatureList.size(); featureIdx++)
    {
        Feature *Feat = FeatureList[featureIdx];
        vector<double> featureSimMatrix(detections.size());

        for(size_t i=0; i<detections.size(); i++)
        {
            BucketDetection det = detections[i];
            featureSimMatrix[i] = (Feat->getDistance(det));
        }

        costMatrix[featureNum] = featureSimMatrix;
        trackerList[featureNum] = make_pair(this, featureIdx);
        featureNum+=1;
    }

    return featureNum;
}

bool FeatureTracker::validCost(double cost)
{
    return cost<mahalanobisThreshold;
}

Feature::Feature(BucketDetection initial_detection)
{
    tag = initial_detection.tag;

    kPosition = new ConstantKalmanFilter((initial_detection.tag+"/position"), point_to_vec(initial_detection.position));
    kOrientation = new ConstantKalmanFilter((initial_detection.tag+"/orientation"), point_to_vec(initial_detection.orientation));

    numDetections = 1;
}

Feature::~Feature()
{
    delete kPosition;
    delete kOrientation;
}

Eigen::Vector3d Feature::getPosition()
{
    Eigen::Vector3d pose(kPosition->getEstimate());
    return pose;
}

Eigen::Vector3d Feature::getOrientation()
{
    Eigen::Vector3d orientation(kOrientation->getEstimate());
    return orientation;
}

void Feature::addDetection(BucketDetection detection)
{
    numDetections++;

    kPosition->updateEstimate(point_to_vec(detection.position));
    kOrientation->updateEstimate(point_to_vec(detection.orientation));
}

double Feature::getDistance(BucketDetection det)
{
    return (getPosition()-point_to_vec(det.position)).norm();
}