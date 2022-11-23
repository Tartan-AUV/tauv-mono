#include "global_map.h"
#include <cmath>

using namespace std;

FeatureTracker::FeatureTracker(BucketDetection &initial_detection)
{
    FeatureList= {};
    Zombies = {};

    addFeature(initial_detection);
    tag = initial_detection.tag; //change ownership to Feature?

    //init mahalnobis distance from params
    mahalanobisThreshold = getParam("mahalanobis_threshold");
    tag_weight = getParam("matching_weights/tag_weight");
    distance_weight = getParam("matching_weights/distance_weight");
    orientation_weight = getParam("matching_weights/orientation_weight");
    recency_weight = getParam("matching_weights/recency_weight");
    frequency_weight = getParam("matching_weights/frequency_weight");
    oversaturation_penalty = getParam("matching_weights/oversaturation_penalty");
    DECAY_THRESHOLD = getParam("DECAY_THRESHOLD");

    predictedFeatureNum = max(size_t(getParam("feature_count")), size_t(1));
}

FeatureTracker::~FeatureTracker()
{
    for(Feature *F:FeatureList)
    {
        delete F;
    }
}

void FeatureTracker::makeUpdates()
{
    //122 moment
    if(Zombies.size()==0 || Zombies.size()<FeatureList.size()/2){return;}

    cout<<"UPDATING\n";

    size_t del = Zombies.top();
    Zombies.pop();

    size_t curIdx = del;

    for(size_t featureIdx = del; featureIdx < FeatureList.size(); featureIdx++)
    {
        if(featureIdx<del){
            FeatureList[curIdx]=FeatureList[featureIdx];
            curIdx++;
        }

        else{
            delete FeatureList[del];

            if(Zombies.size()==0){
                del=FeatureList.size();
            }
            else{
                del = Zombies.top();
                Zombies.pop();
            }
        }
    }

    FeatureList.erase(FeatureList.begin()+curIdx, FeatureList.end());
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

void FeatureTracker::addFeature(BucketDetection &detection)
{
    if(Zombies.size()>0)
    {
        int featureIdx = Zombies.top();
        Zombies.pop();

        FeatureList[featureIdx]->reinit(detection);
        return;
    }
    
    Feature *NEW = new Feature(detection);
    FeatureList.push_back(NEW);
}

void FeatureTracker::deleteFeature(int featureIdx)
{
    cout<<"DELETING!!!\n";
    FeatureList[featureIdx]->reset();
    Zombies.push(featureIdx);
}

double FeatureTracker::recencyCalc(double recency, double totalDet)
{
    //return exp(1-(recency/totalDet))/1.718 - 0.582;
    return 1.582 - exp(recency/totalDet)/1.718;
}

double FeatureTracker::frequencyCalc(double frequency, double totalDet)
{
    return log(frequency)/log(totalDet);
}

double FeatureTracker::decay(int featureIdx, int totalDetections)
{
    Feature *F = FeatureList[featureIdx];
    F->incrementRecency();

    double decay = (recencyCalc(F->getReceny(), totalDetections)+frequencyCalc(F->getNumDetections(), totalDetections))/2.0;

    cout<<"decay: "<<decay<<"\n";

    return totalDetections>2 && decay<DECAY_THRESHOLD;
}

void FeatureTracker::addDetection(BucketDetection &detection, int featureIdx)
{
    FeatureList[featureIdx]->addDetection(detection);
}

double FeatureTracker::getSimilarityCost(Feature *F, BucketDetection &det)
{
    //feature-based similarity costs
    double distance = distance_weight*(F->getDistance(det));
    double orientation = orientation_weight*(F->getRotation(det));
    double tagDist = tag_weight*(F->diffTag(det));

    return distance+orientation+tagDist;
}

//finish similarity cost
vector<double> FeatureTracker::getSimilarityRow(vector<BucketDetection> &detections, size_t featureIdx)
{
    Feature *Feat = FeatureList[featureIdx];

    vector<double> featureSimMatrix(detections.size());

    cout<<"tag: "<<tag<<"\n";
    cout<<"Position: "<<Feat->getPosition()<<"\n";

    for(size_t i=0; i<detections.size(); i++)
    {
        BucketDetection det = detections[i];
        featureSimMatrix[i] = getSimilarityCost(Feat, det);
    }

    return featureSimMatrix;
}

int FeatureTracker::generateSimilarityMatrix(vector<BucketDetection> &detections, vector<vector<double>> &costMatrix, size_t trackerNum, vector<pair<FeatureTracker*, int>> &trackerList)
{
    size_t featureNum = trackerNum;
    for(size_t featureIdx = 0; featureIdx<FeatureList.size(); featureIdx++)
    {
        if(FeatureList[featureIdx]->State==ZOMBIE){continue;}

        costMatrix[featureNum] = getSimilarityRow(detections, featureIdx);
        trackerList[featureNum] = make_pair(this, featureIdx);
        featureNum++;
    }

    return featureNum;
}

bool FeatureTracker::validCost(double cost)
{
    //increase threshold for tracker creation if too many trackers
    int oversaturated = (predictedFeatureNum<FeatureList.size());
    return cost<(mahalanobisThreshold+(oversaturation_penalty*oversaturated));
}

Feature::Feature(BucketDetection &initial_detection)
{
    tag = initial_detection.tag;
    State = ACTIVE;

    kPosition = new ConstantKalmanFilter((initial_detection.tag+"/position"), initial_detection.position);
    kOrientation = new ConstantKalmanFilter((initial_detection.tag+"/orientation"), initial_detection.orientation);

    numDetections = 1;
    recency = 1;
}

Feature::~Feature()
{
    delete kPosition;
    delete kOrientation;
}

void Feature::reset()
{
    State = ZOMBIE;
    numDetections = 0;
}

void Feature::reinit(BucketDetection initial_detection)
{
    State = ACTIVE;

    kPosition->reset(initial_detection.position);
    kOrientation->reset(initial_detection.orientation);

    numDetections = 1;
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

double Feature::getReceny(){return recency;}

void Feature::incrementRecency(){recency++;}

void Feature::addDetection(BucketDetection& detection)
{
    if(State==ZOMBIE){
        reinit(detection);
        return;
    }

    numDetections++;

    kPosition->updateEstimate(detection.position);
    kOrientation->updateEstimate(detection.orientation);
}

double Feature::getDistance(BucketDetection& detection)
{
    return (getPosition()-detection.position).norm();
}

double Feature::getRotation(BucketDetection& detection)
{
    return (getOrientation()-detection.orientation).norm();
}

//true if tags are different, false if same
bool Feature::diffTag(BucketDetection& detection)
{
    return (detection.tag.compare(tag)!=0 && detection.tag.compare("unknown")!=0);
}