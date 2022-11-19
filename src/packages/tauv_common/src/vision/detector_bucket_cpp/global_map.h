#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <tauv_msgs/BucketDetection.h>
#include <tauv_msgs/BucketList.h>
#include <tauv_msgs/RegisterObjectDetections.h>
#include <unordered_map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <mutex>
#include "kalman-filter/kalman_filter.hpp"

class FeatureTracker;
class Feature;

using namespace std;
using namespace tauv_msgs;

Eigen::Vector3d point_to_vec(geometry_msgs::Point point);

//Instance manages a single object in the global map
class Feature
{
    public:
        Feature(BucketDetection initial_detection);
        ~Feature();

        string tag;

        double getDistance(BucketDetection det);
        void addDetection(BucketDetection detection);
        
        Eigen::Vector3d getPosition();
        Eigen::Vector3d getOrientation();
        
    private:
        //more computationally efficient to separate pos and orientation
        //for matrix inversion in kalman filtering
        ConstantKalmanFilter *kPosition;
        ConstantKalmanFilter *kOrientation;
        
        size_t numDetections;
};


//Instance manages all features of a particular tag
class FeatureTracker
{
    public:
        FeatureTracker(BucketDetection initial_detection);
        ~FeatureTracker();

        void addDetection(BucketDetection detection, int featureIdx);
        void addFeature(BucketDetection detection);
        bool validCost(double cost);

        //vector<vector<double>> getSimilarityMatrix(vector<BucketDetection> detections);
        int generateSimilarityMatrix(vector<BucketDetection> &detections, vector<vector<double>> &costMatrix, size_t trackerNum, vector<pair<FeatureTracker*, int>> &trackerList);

    private:
        int totalDetections;
        int predictedFeatureNum;
        
        string tag;

        //matching parameters
        double mahalanobisThreshold;

        vector<Feature*> FeatureList;
};

//Manages feature discovery and high-level matching of detections to trackers
//Communicates with rest of system
class GlobalMap
{
    public:
        GlobalMap(ros::NodeHandle& handler);

        void updateTrackers(const RegisterObjectDetections::ConstPtr& detections);
        void addTracker(BucketDetection detection);

        ros::Publisher publisher;
        ros::Subscriber listener;
        mutex mtx;

    private:
        void assignDetections(vector<BucketDetection> detections);

        unordered_map<string, FeatureTracker*> MAP;
        size_t featureCount;
};