#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <tauv_msgs/BucketDetection.h>
#include <tauv_msgs/BucketList.h>
#include <tauv_msgs/RegisterObjectDetections.h>
#include <std_srvs/Trigger.h>
#include <tauv_msgs/MapFind.h>
#include <tauv_msgs/MapFindClosest.h>
#include <unordered_map>
#include <string>
#include <utility>
#include <iostream>
#include <mutex>
#include <queue>
#include "kalman-filter/kalman_filter.hpp"

//to-do: documentation

using namespace std;

class FeatureTracker;
class Feature;

//useful for internal storage, instead of initializing new Eigen vectors
//we reuse the same struct internally, and work with tauv_msgs only for external purposes
struct BucketDetection {
    Eigen::Vector3d position;
    Eigen::Vector3d orientation;
    string tag;
};

Eigen::Vector3d point_to_vec(geometry_msgs::Point point);
geometry_msgs::Point vec_to_point(Eigen::Vector3d vec);

//creds to advaith the zombie naming is pretty sick
enum TrackerState {
    ACTIVE = 0,
    ZOMBIE = 1
};

//Instance manages a single object in the global map
class Feature : public BucketDetection
{
    public:
        Feature(BucketDetection &initial_detection);
        ~Feature();

        double getDistance(BucketDetection &det);
        double getRotation(BucketDetection& detection);
        bool diffTag(BucketDetection &det);

        void addDetection(BucketDetection &detection);
        size_t getNumDetections();

        double getRecency();
        void incrementRecency();

        void reset();
        void reinit(BucketDetection initial_detection);

        TrackerState State;
        
    private:
        //more computationally efficient to separate pos and orientation
        //for matrix inversion in kalman filtering
        ConstantKalmanFilter *kPosition;
        ConstantKalmanFilter *kOrientation;

        void setPosition();
        void setOrientation();

        size_t numDetections;
        double recency;
};


//Instance manages all features of a particular tag
class FeatureTracker
{
    public:
        FeatureTracker(BucketDetection &initial_detection);
        ~FeatureTracker();

        void addDetection(BucketDetection &detection, int featureIdx);
        void addFeature(BucketDetection &detection);
        void deleteFeature(int featureIdx);

        void makeUpdates();

        double decay(int featureIdx, int totalDetections);
        bool validCost(double cost);

        //vector<vector<double>> getSimilarityMatrix(vector<BucketDetection> detections);
        int generateSimilarityMatrix(vector<BucketDetection> &detections, vector<vector<double>> &costMatrix, size_t trackerNum, vector<pair<FeatureTracker*, int>> &trackerList);

        double getMaxThreshold();

        vector<Feature*> getTrackers();

    private:
        double getParam(string property, double def=0);
        double getSimilarityCost(Feature *F, BucketDetection &det);
        vector<double> getSimilarityRow(vector<BucketDetection> &detections, size_t featureIdx);

        double frequencyCalc(double frequency, double totalDet);
        double recencyCalc(double recency);

        int predictedFeatureNum;
        int numDetections;

        //matching weights
        double mahalanobisThreshold;
        double recency_weight;
        double frequency_weight;
        double DECAY_THRESHOLD;

        double tag_weight;
        double distance_weight;
        double orientation_weight;
        double recency_matching_weight;
        double frequency_matching_weight;
        double oversaturation_penalty;
        
        string tag;

        vector<Feature*> FeatureList;
        priority_queue<size_t, vector<size_t>, std::greater<size_t>> Zombies;

        bool needsUpdating;
};

//Manages feature discovery and high-level matching of detections to trackers
//Communicates with rest of system
//to-do: create a reset service
//to-do: create a service position request interface
class GlobalMap
{
    public:
        GlobalMap(ros::NodeHandle& handler);
        ~GlobalMap();

        void updateTrackers(const tauv_msgs::RegisterObjectDetections::ConstPtr& detections);
        void addTracker(BucketDetection &detection);

        bool find(tauv_msgs::MapFind::Request &req, tauv_msgs::MapFind::Response &res);
        bool findClosest(tauv_msgs::MapFindClosest::Request &req, tauv_msgs::MapFindClosest::Response &res);
        bool reset(std_srvs::Trigger::Request &req, 
                                std_srvs::Trigger::Response &res);

        //void publishMap(const ros::TimerEvent& event); //TEMPORARY

    private:
        void assignDetections(vector<BucketDetection> &detections);
        vector<BucketDetection> convertToStruct(vector<tauv_msgs::BucketDetection> &detections);
        vector<pair<FeatureTracker*, int>> generateSimilarityMatrix(vector<BucketDetection> &detections, vector<vector<double>> &costMatrix, size_t costMatSize);

        unordered_map<string, FeatureTracker*> MAP;
        void updateDecay(FeatureTracker *F, int featureIdx);

        //ros::Publisher publisher;
        ros::Subscriber listener;
        ros::ServiceServer resetService;
        ros::ServiceServer findService;
        ros::ServiceServer findClosestService;
        mutex mtx;

        size_t featureCount;
        size_t totalDetections;
        double  DUMMY_FILL;
};