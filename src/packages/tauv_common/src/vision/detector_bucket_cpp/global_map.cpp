#include "global_map.h"
#include "hungarian-algorithm-cpp/Hungarian.h"

Eigen::Vector3d point_to_vec(geometry_msgs::Point point)
{
    Eigen::Vector3d vec;
    vec<<point.x, point.y, point.z;
    return vec;
}

geometry_msgs::Point vec_to_point(Eigen::Vector3d vec)
{
    geometry_msgs::Point point{};
    point.x = vec[0];
    point.y = vec[1];
    point.z = vec[2];
    return point;
}

GlobalMap::GlobalMap(ros::NodeHandle& handler)
{
    featureCount = 0;
    totalDetections = 0;
    DUMMY_FILL=0;
    MAP = {};

    listener = handler.subscribe("/register_object_detection", 100, &GlobalMap::updateTrackers, this);
    resetService = handler.advertiseService("/global_map/reset", &GlobalMap::reset, this);
    findService = handler.advertiseService("/global_map/find", &GlobalMap::find, this);
    findClosestService = handler.advertiseService("/global_map/find_closest", &GlobalMap::findClosest, this);

    timer =
        handler.createTimer(ros::Duration(1.0),
                        &GlobalMap::publishMap, this);
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
    size_t initFeatureNum = featureCount; //will change during iterations if new trackers added

    size_t costMatSize = max(featureCount,detections.size());
    vector<vector<double>> costMatrix(costMatSize);
    vector<pair<FeatureTracker*, int>> trackerList = generateSimilarityMatrix(detections, costMatrix, costMatSize);

    //find best assignment
    vector<int> assignment;
    HungarianAlgorithm Matcher;
    Matcher.Solve(costMatrix, assignment);

    // cout<<"ASSIGNMENT: \n";
    // for(int A : assignment){
    //     cout<<A<<" ";
    // }
    // cout<<"\n";

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
        DUMMY_FILL = max(DUMMY_FILL, F->getMaxThreshold());
    }
    else
    {
        (NEW->second)->addFeature(detection);
    }
}

bool GlobalMap::find(tauv_msgs::MapFind::Request &req, tauv_msgs::MapFind::Response &res)
{
    unordered_map<string,FeatureTracker*>::iterator Tracker = MAP.find(req.tag);

    if(Tracker == MAP.end()){res.success=false; return false;}

    vector<Feature*> detections = (Tracker->second)->getTrackers();
    vector<tauv_msgs::BucketDetection> returnDetections(detections.size());
    for(Feature *detection : detections)
    {
        tauv_msgs::BucketDetection returnDetection{};
        returnDetection.position = vec_to_point(detection->position);
        returnDetection.orientation = vec_to_point(detection->orientation);
        returnDetection.tag = detection->tag;

        returnDetections.push_back(returnDetection);
    }

    res.detections = returnDetections;
    res.success = true;

    return true;
}

bool GlobalMap::findClosest(tauv_msgs::MapFindClosest::Request &req, tauv_msgs::MapFindClosest::Response &res)
{
    unordered_map<string,FeatureTracker*>::iterator Tracker = MAP.find(req.tag);
    Eigen::Vector3d position = point_to_vec(req.point);

    if(Tracker == MAP.end()){res.success=false; return false;}

    vector<Feature*> detections = (Tracker->second)->getTrackers();

    int minInd = 0;
    double minDist = -1;
    for(size_t i = 0; i<detections.size(); i++)
    {
        Feature *detection = detections[i];
        double dist = (detection->position - position).norm();

        if(minDist<0 || dist<minDist)
        {
            minInd = i;
            minDist = dist;
        }
    }

    tauv_msgs::BucketDetection returnDetection{};
    returnDetection.position = vec_to_point(detections[i]->position);
    returnDetection.orientation = vec_to_point(detections[i]->orientation);
    returnDetection.tag = tag;

    req.detection = returnDetection;
    req.success = true;

    return true;
}

bool GlobalMap::reset(std_srvs::Trigger::Request &req, std_srvs::Trigger::Response &res)
{
    for(pair<string,FeatureTracker*> Tracker: MAP)
    {
        delete Tracker.second;
    }

    featureCount = 0;
    totalDetections = 0;
    DUMMY_FILL=0;
    MAP = {};

    res.success = true;
    return true;
}

/*void GlobalMap::publishMap(const ros::TimerEvent& event)
{
    cout<<"FEATURECOUNT: "<<featureCount<<"\n";
    vector<tauv_msgs::BucketDetection> buckets1 = find("badge").bucket_list;
    vector<tauv_msgs::BucketDetection> buckets2 = find("notebook").bucket_list;
    vector<tauv_msgs::BucketDetection> buckets3 = find("phone").bucket_list;

    buckets1.insert(buckets1.end(), buckets2.begin(), buckets2.end());
    buckets1.insert(buckets1.end(), buckets3.begin(), buckets3.end());

    tauv_msgs::BucketList buckets{};
    buckets.bucket_list = buckets1;
    publisher.publish(buckets);
}*/