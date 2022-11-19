#include "global_map.h"
#include "hungarian-algorithm-cpp/Hungarian.h"

using namespace tauv_msgs;

GlobalMap::GlobalMap(ros::NodeHandle& handler)
{
    featureCount = 0;
    DUMMY_FILL=0;
    MAP = {};

    publisher = handler.advertise<tauv_msgs::BucketList>("/bucket_list", 100); //update to a service
    listener = handler.subscribe("/register_object_detection", 100, &GlobalMap::updateTrackers, this);
}

GlobalMap::~GlobalMap()
{
    for(pair<string,FeatureTracker*> Tracker: MAP)
    {
        delete Tracker.second;
    }
}

//change to error state return
void GlobalMap::updateTrackers(const RegisterObjectDetections::ConstPtr& detectionObjects)
{
    vector<BucketDetection> detections = detectionObjects->objdets;

    if(detections.size()==0){
        return;
    }

    mtx.lock();
    if(featureCount==0)
    {
        addTracker(detections[0]);
    }

    //mutex moment
    assignDetections(detections);
    mtx.unlock();
}

void GlobalMap::assignDetections(vector<BucketDetection> detections)
{
    size_t costMatSize = max(featureCount,detections.size());
    vector<vector<double>> costMatrix(costMatSize);

    //list for future detection addition
    vector<pair<FeatureTracker*, int>> trackerList(featureCount);

    size_t featureNum = 0;
    for(pair<string,FeatureTracker*> Tracker: MAP)
    {   
        featureNum = Tracker.second->generateSimilarityMatrix(detections, costMatrix, featureNum, trackerList);
    }

    //fill with dummy trackers, should only happen when new object detected
    for(size_t dummies = featureNum; dummies<costMatSize ; dummies++)
    {
        vector<double> DUMMY(detections.size());
        costMatrix[dummies] = DUMMY;
        for(size_t j=0; j<detections.size(); j++)
        {
            costMatrix[dummies][j] = DUMMY_FILL;
        }
    }

    //find best assignment
    vector<int> assignment;
    HungarianAlgorithm Matcher;
    Matcher.Solve(costMatrix, assignment);

    //if valid match, assign detection to tracker,
    //otherwise, create a new tracker
    for(size_t tracker=0; tracker<featureNum;tracker++)
    {
        int detectionIdx = assignment[tracker];

        //tracker had no corresponding detections
        if(detectionIdx<0){continue;}

        pair<FeatureTracker*, int> Tracker = trackerList[tracker];
        if(Tracker.first->validCost(costMatrix[tracker][detectionIdx]))
        {
            Tracker.first->addDetection(detections[detectionIdx], Tracker.second);
        }
        else 
        {
            addTracker(detections[detectionIdx]);
        }
    }

    //make new trackers for unmatched detections
    for(size_t unmatched=featureNum; unmatched<costMatSize; unmatched++)
    {
        int detectionIdx = assignment[unmatched];
        addTracker(detections[detectionIdx]);
    }
}

void GlobalMap::addTracker(BucketDetection detection)
{
    unordered_map<string,FeatureTracker*>::iterator NEW = MAP.find(detection.tag);
    featureCount+=1;

    if(NEW == MAP.end())
    {
        FeatureTracker *F = new FeatureTracker(detection);
        MAP.insert({detection.tag, F});
        DUMMY_FILL = max(DUMMY_FILL, F->getMahalanobisThreshold());
    }
    else
    {
        (NEW->second)->addFeature(detection);
    }
}