#include "global_map.h"
#include "hungarian-algorithm-cpp/Hungarian.h"

Eigen::Vector3d point_to_vec(geometry_msgs::Point point)
{
    Eigen::Vector3d vec;
    vec<<point.x, point.y, point.z;
    return vec;
}

GlobalMap::GlobalMap(ros::NodeHandle& handler)
{
    featureCount = 0;
    totalDetections = 0;
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

vector<BucketDetection> GlobalMap::convertToStruct(vector<tauv_msgs::BucketDetection> &detections)
{
    vector<BucketDetection> trueDets(detections.size());
    
    for(size_t i=0;i<detections.size();i++)
    {
        tauv_msgs::BucketDetection detection = detections[i];
        trueDets[i] = BucketDetection{point_to_vec(detection.position), point_to_vec(detection.orientation), detection.tag};
    }

    return trueDets;
}

//change to error state return
void GlobalMap::updateTrackers(const tauv_msgs::RegisterObjectDetections::ConstPtr& detectionObjects)
{
    //convert trackers into readable cpp format to save later repeated computation
    vector<tauv_msgs::BucketDetection> objdets = detectionObjects->objdets;
    vector<BucketDetection> detections = convertToStruct(objdets);

    size_t len = detections.size();
    
    if(len==0){
        return;
    }

    totalDetections+=len;

    mtx.lock();

    //mutex moment
    assignDetections(detections);

    mtx.unlock();
}

vector<pair<FeatureTracker*, int>> GlobalMap::generateSimilarityMatrix(vector<BucketDetection> &detections, vector<vector<double>> &costMatrix, size_t costMatSize)
{
    //list for detection addition
    vector<pair<FeatureTracker*, int>> trackerList(featureCount);

    size_t featureNum = 0;
    for(pair<string,FeatureTracker*> Tracker: MAP)
    {
        //delete any retired trackers prior to matching
        Tracker.second->makeUpdates();
        featureNum = Tracker.second->generateSimilarityMatrix(detections, costMatrix, featureNum, trackerList);
    }

    //fill with dummy trackers, only happens when new object detected
    for(size_t dummies = featureNum; dummies<costMatSize ; dummies++)
    {
        vector<double> DUMMY(detections.size());
        costMatrix[dummies] = DUMMY;
        for(size_t j=0; j<detections.size(); j++)
        {
            costMatrix[dummies][j] = DUMMY_FILL;
        }
    }

    return trackerList;
}

void GlobalMap::assignDetections(vector<BucketDetection> &detections)
{
    size_t costMatSize = max(featureCount,detections.size());
    vector<vector<double>> costMatrix(costMatSize);
    //vector<pair<FeatureTracker*, int>> trackerList = generateSimilarityMatrix(detections, costMatrix, costMatSize);

    //list for detection addition
    vector<pair<FeatureTracker*, int>> trackerList(featureCount);

    cout<<"count: "<<featureCount<<"\n";

    size_t initFeatureNum = 0;
    for(pair<string,FeatureTracker*> Tracker: MAP)
    {
        //delete any retired trackers prior to matching
        Tracker.second->makeUpdates();
        initFeatureNum = Tracker.second->generateSimilarityMatrix(detections, costMatrix, initFeatureNum, trackerList);
    }

    //fill with dummy trackers, only happens when new object detected
    for(size_t dummies = initFeatureNum; dummies<costMatSize ; dummies++)
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
    for(size_t tracker=0; tracker<initFeatureNum;tracker++)
    {
        int detectionIdx = assignment[tracker];
        pair<FeatureTracker*, int> Tracker = trackerList[tracker];

        //tracker had no corresponding detections
        if(detectionIdx<0){updateDecay(Tracker.first, Tracker.second); continue;}

        if(Tracker.first->validCost(costMatrix[tracker][detectionIdx]))
        {
            Tracker.first->addDetection(detections[detectionIdx], Tracker.second);
        }
        else 
        {
            updateDecay(Tracker.first, Tracker.second);
            addTracker(detections[detectionIdx]);
        }
    }

    //make new trackers for unmatched detections
    for(size_t unmatched=initFeatureNum; unmatched<costMatSize; unmatched++)
    {
        int detectionIdx = assignment[unmatched];
        addTracker(detections[detectionIdx]);
    }
}

void GlobalMap::updateDecay(FeatureTracker *F, int featureIdx)
{
    if(F->decay(featureIdx, totalDetections))
    {
        F->deleteFeature(featureIdx);
        featureCount-=1;
    }
}

void GlobalMap::addTracker(BucketDetection &detection)
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