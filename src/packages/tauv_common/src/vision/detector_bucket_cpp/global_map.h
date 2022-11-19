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

        double getDistance(BucketDetection det);
        void addDetection(BucketDetection detection);
        size_t getNumDetections();
        
        Eigen::Vector3d getPosition();
        Eigen::Vector3d getOrientation();
        
    private:
        //more computationally efficient to separate pos and orientation
        //for matrix inversion in kalman filtering
        ConstantKalmanFilter *kPosition;
        ConstantKalmanFilter *kOrientation;
        string tag;
        size_t numDetections;
};


//Instance manages all features of a particular tag
//to do: add recency to matching
//add inactive features and feature zombification
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

        //matching parameters
        double getMahalanobisThreshold();

    private:
        double getParam(string property);
        vector<double> getSimilarityRow(vector<BucketDetection> &detections, size_t featureIdx);

        size_t totalDetections;
        size_t predictedFeatureNum;

        //matching weights
        double mahalanobisThreshold;
        int tag_weight;
        int distance_weight;
        int orientation_weight;
        int recency_weight;
        int frequency_weight;
        int oversaturation_penalty;
        
        string tag;

        vector<Feature*> FeatureList;
};

//Manages feature discovery and high-level matching of detections to trackers
//Communicates with rest of system
class GlobalMap
{
    public:
        GlobalMap(ros::NodeHandle& handler);
        ~GlobalMap();

        void updateTrackers(const RegisterObjectDetections::ConstPtr& detections);
        void addTracker(BucketDetection detection);

    private:
        void assignDetections(vector<BucketDetection> detections);

        unordered_map<string, FeatureTracker*> MAP;

        ros::Publisher publisher;
        ros::Subscriber listener;
        mutex mtx;

        size_t featureCount;
        double DUMMY_FILL;
};