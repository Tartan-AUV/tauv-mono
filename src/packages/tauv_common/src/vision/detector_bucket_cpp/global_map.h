#include <ros/ros.h>
#include <eigen3/Eigen/Dense>
#include <tauv_msgs/FeatureDetection.h>
#include <tauv_msgs/FeatureDetections.h>
#include <std_srvs/Trigger.h>
#include <tauv_msgs/FeatureDetectionsSync.h>
#include <tauv_msgs/MapFind.h>
#include <tauv_msgs/MapFindClosest.h>
#include <tauv_msgs/MapFindOne.h>
#include <unordered_map>
#include <string>
#include <utility>
#include <iostream>
#include <mutex>
#include <queue>
#include <memory>
#include "kalman-filter/kalman_filter.hpp"

/**
 * Mapping system for any feature detections. Uses Kalman filtering and similarity matching to derive better estimates
 * of objects over time. Uses detection count and recency decay to filter sparse false detections.
 * All tunable parameters are in global_map.yaml
**/

using namespace std;

class FeatureTracker;
class Feature;

/**
 * Use an internal C++ definition of detections to avoid reinitializing new Eigen vectors instead of reusing the 
 * same struct internally to do math and working with mathematically inconvenient tauv_msgs only for external purposes
**/
struct FeatureDetection {
    Eigen::Vector3d position;
    Eigen::Vector3d orientation;
    string tag;
    /**
     * Survey detections are detections whose tag is unknown. These will match with any tracker
    **/
    bool survey_point;
};

//struct conversion helper functions
Eigen::Vector3d point_to_vec(geometry_msgs::Point point);
geometry_msgs::Point vec_to_point(Eigen::Vector3d vec);

/**
 * Creds to advaith the zombie naming is pretty sick
 * Used to represent Trackers that have been retired from taking any additional input because of a lack of recency
 * and frequency of detections, which when low enough are assumed to mean the Tracker's detections were false positives
 **/
enum TrackerState {
    ACTIVE = 0,
    ZOMBIE = 1, //retired
    POTENTIAL_ZOMBIE = 2 //tracker has gone over min decay time and no longer is obligated to be present 
};

/**
 * A Feature represents a single detected object that was measured with error over time. This class performs the
 * feature-specific Kalman filtering and is the "bucket", where all the different feature measurements are sent
 * and combined for a better estimate of the object's position and orientation.
 **/
class Feature
{
    public:
        Feature(FeatureDetection &initial_detection, double min_decay_time, ros::NodeHandle& handler);

        //convenience functions, coordinates need retrieval from Kalman filter
        Eigen::Vector3d getPosition();
        Eigen::Vector3d getOrientation();

        //accomodation functions for survey detections
        static bool isSurveyDet(FeatureDetection& detection) {return detection.survey_point;};
        static double getSurveyRotation (FeatureDetection& detection) {return 0;};
        double getSurveyDistance (FeatureDetection& detection);

        double getDistance(FeatureDetection &det);
        double getRotation(FeatureDetection& detection);

        void addDetection(FeatureDetection &detection, double new_decay_time);
        size_t getNumDetections();

        double getRecency();
        void incrementRecency();

        /**
         * reset is used to put a tracker into "ZOMBIE" mode (wipes data and ready to be reinitialized with new detections)
         * reinit is used to revive the tracker with completely new data
         * same as deleting and making a new tracker but slightly more efficient with regards to FeatureTracker that owns the Feature
        **/
        void reset();
        void reinit(FeatureDetection &initial_detection, double min_decay_time);
        void setPotentialZombie(const ros::TimerEvent& event);

        TrackerState State;
        
    private:
        /**
        * more computationally efficient to separate position and orientation predictions
        * for the matrix inversion required in Kalman filtering
        **/
        unique_ptr<ConstantKalmanFilter> kPosition;
        unique_ptr<ConstantKalmanFilter> kOrientation;

        size_t numDetections;
        double recency;

        ros::Timer decayTimer; //tracks elapsed time between detections and creates a potential zombie if overtime
        ros::NodeHandle nodeHandler;
};


/**
 * One FeatureTracker owns all Features of a particular tag.
 * This parent class specializes in setting up matching and retiring unused Features.
 * It generates the similarity matching costs for each of its Feature slaves. 
 * By controlling all Features of a tag from one Tracker, we can perform tag-specific matching protocols by allowing 
 * different tags to have different matching weights (and thus allow different tags to prioritize matching in different ways).
 * We can also use this class to make sure we aren't marking more Features than we expect of a particular tag 
 * (if we know we have two badges, we should be creating a penalty or simply not allowing more than two badges).
 * This class serves as the interface between the GlobalMap and Features, which are focused only on filtering any
 * detections they are given.
**/
class FeatureTracker : public enable_shared_from_this<FeatureTracker>
{
    public:
        FeatureTracker(FeatureDetection &initial_detection, ros::NodeHandle& handler);

        //for interfacing with Features
        void addDetection(FeatureDetection &detection, int featureIdx);
        void addFeature(FeatureDetection &detection);

        /**For Zombifying Features when their decay (some tunable combination of frequency and recency) score is too low.
        * Creating Zombies allows us to avoid interfering with the Feature List while the Global Map is commanding matching,
        * thus preserving safe memory accesses, and creates a way to reuse already setup Features. 
        * Global Map commands updates to Feature List only prior to assignment and matching.
        **/
        void deleteFeature(int featureIdx);
        double getDecay(shared_ptr<Feature> F, int totalDetections);
        double decay(int featureIdx, int totalDetections);
        //for deleting Features when there are too many Zombies
        void makeUpdates();

        //matching information for Global Map
        bool validCost(double cost);
        int generateSimilarityMatrix(vector<FeatureDetection> &detections, vector<vector<double>> &costMatrix, size_t trackerNum, vector<pair<shared_ptr<FeatureTracker>, int>> &trackerList);
        double getMaxThreshold();

        vector<shared_ptr<Feature>> getFeatures();
        size_t getNumFeatures();

        //make private
        string tag;

        /**
         * Survey trackers never match with non-survey points so that corresponding non-survey points generate a new, dominating concrete
         * tracker once the object has been "surveyed" by a concrete detection device. 
         * Survey trackers are given a survey bias that makes matching with potential survey points more difficult than for concrete trackers,
         * encouraging the decay of survey trackers after a corresponding concrete tracker has been discovered.
        **/
        bool survey_point; //will match with only other survey points

    private:
        double getParam(string property, double def=0);
        double getSimilarityCost(shared_ptr<Feature> F, FeatureDetection &det);
        vector<double> getSimilarityRow(vector<FeatureDetection> &detections, size_t featureIdx);

        double frequencyCalc(double frequency, double totalDet);
        double recencyCalc(double recency);
        bool diffTag(FeatureDetection &det); //managed by FeatureTracker and not Feature since concerns tag matching

        int predictedFeatureNum;
        int numDetections;

        //decay weights
        double recency_weight;
        double frequency_weight;
        double DECAY_THRESHOLD;
        double min_decay_time;

        //matching weights
        double mahalanobisThreshold;
        double tag_weight;
        double distance_weight;
        double orientation_weight;
        double recency_matching_weight;
        double frequency_matching_weight;
        double oversaturation_penalty;
        double tracker_bias;
        double total_weight; //used to approximately normalize total weights between tracker types

        vector<shared_ptr<Feature>> FeatureList;
        priority_queue<size_t, vector<size_t>, std::greater<size_t>> Zombies;

        //new zombies in queue!!!
        bool needsUpdating;

        ros::NodeHandle nodeHandler;
};


/**
 * GlobalMap communicates with rest of system and manages high-level matching of detections to trackers.
 * It is an interface for Feature search and commands the process flow of detection distribution.
 * Essentially tells FeatureTrackers who to assign detections to (its a whole bureaucracy) after getting
 * similarity information from them.
**/
class GlobalMap
{
    public:
        GlobalMap(ros::NodeHandle& handler);

        void updateTrackersInterface(const tauv_msgs::FeatureDetections::ConstPtr& detectionObjects);
        bool updateTrackers(vector<tauv_msgs::FeatureDetection> objdets);
        void addTracker(FeatureDetection &detection);

        //service callback functions
        //returns all detections matching selected tag
        bool find(tauv_msgs::MapFind::Request &req, tauv_msgs::MapFind::Response &res);
        //returns the most recent/highest count object with a certain tag
        bool findOne(tauv_msgs::MapFindOne::Request &req, tauv_msgs::MapFindOne::Response &res);
        //returns the tracker closest to the passed point estimate with a matching tag
        bool findClosest(tauv_msgs::MapFindClosest::Request &req, tauv_msgs::MapFindClosest::Response &res);
        bool reset(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res);

        //for synchronous interactions w map
        bool syncAddDetections(tauv_msgs::FeatureDetectionsSync::Request &req, tauv_msgs::FeatureDetectionsSync::Response &res); 

    private:
        void assignDetections(vector<FeatureDetection> &detections);
        vector<FeatureDetection> convertToStruct(vector<tauv_msgs::FeatureDetection> &detections);
        vector<pair<shared_ptr<FeatureTracker>, int>> generateSimilarityMatrix(vector<FeatureDetection> &detections, vector<vector<double>> &costMatrix, size_t costMatSize);

        unordered_map<string, shared_ptr<FeatureTracker>> MAP;
        void updateDecay(shared_ptr<FeatureTracker> F, int featureIdx);

        ros::Subscriber listener;
        ros::ServiceServer resetService;
        ros::ServiceServer syncDetectionsService;
        ros::ServiceServer findService;
        ros::ServiceServer findOneService;
        ros::ServiceServer findClosestService;
        ros::NodeHandle nodeHandler;
        mutex mtx;

        size_t featureCount;
        size_t totalDetections;
        double  DUMMY_FILL;
};