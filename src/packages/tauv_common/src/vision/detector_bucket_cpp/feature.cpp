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
    mahalanobisThreshold = getParam("mahalanobis_threshold");
    tag_weight = getParam("tag_weight");
    distance_weight = getParam("distance_weight");
    orientation_weight = getParam("orientation_weight");
    recency_weight = getParam("recency_weight");
    frequency_weight = getParam("frequency_weight");
    oversaturation_penalty = getParam("oversaturation_penalty");

    predictedFeatureNum = max(size_t(getParam("feature_count")), size_t(1));
}

FeatureTracker::~FeatureTracker()
{
    for(Feature *F:FeatureList)
    {
        delete F;
    }
}

double FeatureTracker::getParam(string property)
{
    double val;

    //try tag-specific
    if(!ros::param::get("/tracker_params/"+tag+"/"+property, val)){
        //try default
        if(!ros::param::get("/tracker_params/default/"+property, val))
        {
            return 0;
        }
    }

    return val;
}

double FeatureTracker::getMahalanobisThreshold(){return mahalanobisThreshold;}

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

//finish similarity cost
vector<double> FeatureTracker::getSimilarityRow(vector<BucketDetection> &detections, size_t featureIdx)
{
    Feature *Feat = FeatureList[featureIdx];

    vector<double> featureSimMatrix(detections.size());

    for(size_t i=0; i<detections.size(); i++)
    {
        BucketDetection det = detections[i];
        featureSimMatrix[i] = distance_weight*(Feat->getDistance(det));
    }

    return featureSimMatrix;
}

int FeatureTracker::generateSimilarityMatrix(vector<BucketDetection> &detections, vector<vector<double>> &costMatrix, size_t trackerNum, vector<pair<FeatureTracker*, int>> &trackerList)
{

    size_t featureNum = trackerNum;
    for(size_t featureIdx = 0; featureIdx<FeatureList.size(); featureIdx++)
    {
        costMatrix[featureNum] = getSimilarityRow(detections, featureIdx);
        trackerList[featureNum] = make_pair(this, featureIdx);
        featureNum+=1;
    }

    return featureNum;
}

bool FeatureTracker::validCost(double cost)
{
    //increase threshold for tracker creation if too many trackers
    int oversaturated = (predictedFeatureNum<FeatureList.size());
    return cost<(mahalanobisThreshold+(oversaturation_penalty*oversaturated));
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

size_t Feature::getNumDetections(){return numDetections;}

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