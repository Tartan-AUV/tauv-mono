#include "global_map.h"
#include <cmath>

#define NO_PREDICTED_NUM -1
#define MIN_DETECTIONS 5

using namespace std;

FeatureTracker::FeatureTracker(FeatureDetection &initial_detection)
{
    FeatureList= {};
    Zombies = {};
    needsUpdating = false;
    numDetections = 1;

    addFeature(initial_detection);
    tag = initial_detection.tag;

    //init params
    mahalanobisThreshold = getParam("mahalanobis_threshold");
    tag_weight = getParam("matching_weights/tag_weight");
    distance_weight = getParam("matching_weights/distance_weight");
    orientation_weight = getParam("matching_weights/orientation_weight");
    recency_matching_weight = getParam("matching_weights/recency_weight");
    frequency_matching_weight = getParam("matching_weights/frequency_weight");
    recency_weight = getParam("recency_weight");
    frequency_weight = getParam("frequency_weight");
    oversaturation_penalty = getParam("matching_weights/oversaturation_penalty");
    DECAY_THRESHOLD = getParam("DECAY_THRESHOLD", 100);

    predictedFeatureNum = getParam("expected_count", NO_PREDICTED_NUM);
}

vector<shared_ptr<Feature>> FeatureTracker::getFeatures(){return FeatureList;}

size_t FeatureTracker::getNumFeatures(){return FeatureList.size() - Zombies.size();}

//this will delete Zombies if they are taking up more than half of the Feature List array
void FeatureTracker::makeUpdates()
{
    needsUpdating = false;

    //122 moment
    if(Zombies.size()==0 || Zombies.size()<FeatureList.size()/2){return;}

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
            //delete FeatureList[del];

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


double FeatureTracker::getParam(string property, double def)
{
    double val;

    //try tag-specific
    if(!ros::param::get("/tracker_params/"+tag+"/"+property, val)){
        //try default
        if(!ros::param::get("/tracker_params/default/"+property, val))
        {
            return def;
        }
    }

    return val;
}

double FeatureTracker::getMaxThreshold(){return mahalanobisThreshold+oversaturation_penalty;}

void FeatureTracker::addFeature(FeatureDetection &detection)
{
    if(Zombies.size()>0)
    {
        int featureIdx = Zombies.top();
        Zombies.pop();

        FeatureList[featureIdx]->reinit(detection);
        return;
    }
    shared_ptr<Feature> NEW (new Feature(detection));
    FeatureList.push_back(NEW);
}

void FeatureTracker::deleteFeature(int featureIdx)
{
    FeatureList[featureIdx]->reset();
    Zombies.push(featureIdx);
    needsUpdating = true;
}

double FeatureTracker::recencyCalc(double recency)
{
    //return exp(1-(recency/totalDet))/1.718 - 0.582;
    //return (1-recency/totalDet);
    return 1.582 - exp(recency/numDetections)/1.718;
}

double FeatureTracker::frequencyCalc(double frequency, double totalDet)
{
    return log(frequency)/log(totalDet);
}

double FeatureTracker::decay(int featureIdx, int totalDetections)
{
    shared_ptr<Feature> F= FeatureList[featureIdx];
    F->incrementRecency();

    double recDecay = recency_weight*recencyCalc(F->getRecency());
    double freqDecay = frequency_weight*frequencyCalc(F->getNumDetections(), totalDetections);
    double decay = (recDecay+freqDecay)/(frequency_weight+recency_weight);

    //cout<<"DECAY: "<<decay<<"\n";

    return totalDetections>MIN_DETECTIONS && decay<DECAY_THRESHOLD;
}

void FeatureTracker::addDetection(FeatureDetection &detection, int featureIdx)
{
    numDetections+=1;
    FeatureList[featureIdx]->addDetection(detection);
}

double FeatureTracker::getSimilarityCost(shared_ptr<Feature> F, FeatureDetection &det)
{
    //feature-based similarity costs
    double distance = distance_weight*(F->getDistance(det));
    double orientation = orientation_weight*(F->getRotation(det));
    double tagDist = tag_weight*(F->diffTag(det));

    //tracker-bias similarity costs
    double freqDist = frequency_matching_weight*(1-(F->getNumDetections()/(numDetections)));
    double recDist = recency_matching_weight*(F->getRecency()/numDetections);

    return distance+orientation+tagDist+freqDist+recDist;
}

//finish similarity cost
vector<double> FeatureTracker::getSimilarityRow(vector<FeatureDetection> &detections, size_t featureIdx)
{
    shared_ptr<Feature> Feat = FeatureList[featureIdx];

    vector<double> featureSimMatrix(detections.size());

    cout<<"tag: "<<tag<<"\n";
    cout<<"Position: "<<Feat->getPosition()<<"\n";

    for(size_t i=0; i<detections.size(); i++)
    {
        FeatureDetection det = detections[i];
        featureSimMatrix[i] = getSimilarityCost(Feat, det);
    }

    return featureSimMatrix;
}

int FeatureTracker::generateSimilarityMatrix(vector<FeatureDetection> &detections, vector<vector<double>> &costMatrix, size_t trackerNum, vector<pair<shared_ptr<FeatureTracker>, int>> &trackerList)
{
    //make any needed updates prior to matching
    if(needsUpdating){makeUpdates();}

    size_t featureNum = trackerNum;
    for(size_t featureIdx = 0; featureIdx<FeatureList.size(); featureIdx++)
    {
        if(FeatureList[featureIdx]->State==ZOMBIE){continue;}

        costMatrix[featureNum] = getSimilarityRow(detections, featureIdx);
        trackerList[featureNum] = make_pair(shared_from_this(), featureIdx);
        featureNum++;
    }

    return featureNum;
}

bool FeatureTracker::validCost(double cost)
{
    //increase threshold for tracker creation if too many trackers
    int oversaturated = predictedFeatureNum!=NO_PREDICTED_NUM && (predictedFeatureNum<=int(FeatureList.size()));
    return cost<(mahalanobisThreshold+(oversaturation_penalty*oversaturated));
}

Feature::Feature(FeatureDetection &initial_detection) :
    kPosition (new ConstantKalmanFilter((initial_detection.tag+"/position"), initial_detection.position)),
    kOrientation (new ConstantKalmanFilter((initial_detection.tag+"/orientation"), initial_detection.orientation))
{
    tag = initial_detection.tag;
    State = ACTIVE;

    numDetections = 1;
    recency = 1;
}

void Feature::reset()
{
    State = ZOMBIE;
    numDetections = 0;
}

void Feature::reinit(FeatureDetection &initial_detection)
{
    State = ACTIVE;

    kPosition->reset(initial_detection.position);
    kOrientation->reset(initial_detection.orientation);

    numDetections = 1;
}

Eigen::Vector3d Feature::getPosition()
{
    return kPosition->getEstimate();
}

Eigen::Vector3d Feature::getOrientation()
{
    return kOrientation->getEstimate();
}

size_t Feature::getNumDetections(){return numDetections;}

double Feature::getRecency(){return recency;}

void Feature::incrementRecency(){recency++;}

void Feature::addDetection(FeatureDetection& detection)
{
    if(State==ZOMBIE){
        reinit(detection);
        return;
    }

    numDetections++;

    kPosition->updateEstimate(detection.position);
    kOrientation->updateEstimate(detection.orientation);
}

double Feature::getDistance(FeatureDetection& detection)
{
    return (getPosition()-detection.position).norm();
}

double Feature::getRotation(FeatureDetection& detection)
{
    return (getOrientation()-detection.orientation).norm();
}

//true if tags are different, false if same
bool Feature::diffTag(FeatureDetection& detection)
{
    return (detection.tag.compare(tag)!=0 && detection.tag.compare("unknown")!=0);
}