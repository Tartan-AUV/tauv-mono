#include "./kalman_filter.hpp"

using namespace std;

KalmanFilter::KalmanFilter(string det_tag, Eigen::VectorXd initial_estimate, double confidence)
{
    estimate = initial_estimate;
    tag = det_tag;
    dataDim = initial_estimate.size();
    max_confidence = confidence;

    readParams();
}

shared_ptr<KalmanFilter> KalmanFilter::copy(string new_tag)
{
    shared_ptr<KalmanFilter> NEW (new KalmanFilter(new_tag, estimate, max_confidence));
    NEW->Pk = Pk;
    NEW->Q = Q;
    NEW->R = R;
    NEW->A = A;
    NEW->B = B;
    NEW->H = H;
    return NEW;
}

void KalmanFilter::readParams()
{
    //read in the kalman filter ros parameters
    Pk = getParam("Pk");
    Q = getParam("Q");
    R = getParam("R");
    A = getParam("A");
    B = getParam("B");
    H = getParam("H");
}

Eigen::MatrixXd KalmanFilter::getParam(string property)
{
    vector<double> vec;

    //try tag-specific
    if(!ros::param::get("/kalman_params/"+tag+"/"+property, vec)){
        //try default
        if(!ros::param::get("/kalman_params/default/"+property, vec))
        {
            //initialize to identity
            return Eigen::MatrixXd::Identity(dataDim,dataDim);
        }
    }

    Eigen::Map<Eigen::MatrixXd> MAT(vec.data(), dataDim, dataDim);
    return MAT;
}

double KalmanFilter::getEstimateConfidence(double confidence)
{
    max_confidence = max(max_confidence, confidence);
    return (max_confidence==0.0 ? 1.0 : confidence/max_confidence);
}

void KalmanFilter::makeUpdates(Eigen::VectorXd newEstimate, Eigen::VectorXd zk, double confidence)
{
    Eigen::MatrixXd PkEstimate = (A*Pk * A.transpose()) + Q;

    Eigen::MatrixXd kGain = (PkEstimate*H.transpose())*((H*PkEstimate*H.transpose())+R).inverse();

    estimate = newEstimate+confidence*kGain*(zk-(H*newEstimate));
    Pk = (Eigen::MatrixXd::Identity(dataDim, dataDim) - (kGain*H))*PkEstimate;
}

void KalmanFilter::updateEstimate(Eigen::VectorXd zk, Eigen::VectorXd uk, double confidence)
{
    Eigen::VectorXd newEstimate = (A * estimate) + (B*uk);
    makeUpdates(newEstimate, zk, getEstimateConfidence(confidence));
}

void KalmanFilter::updateEstimate(Eigen::VectorXd zk, double confidence)
{
    Eigen::VectorXd newEstimate = A * estimate; //independent of state B = 0
    makeUpdates(newEstimate, zk, getEstimateConfidence(confidence));
}

void KalmanFilter::reset(Eigen::VectorXd zk, double confidence)
{
    Pk = getParam("Pk");
    estimate = zk;
    max_confidence = confidence;
}

Eigen::VectorXd KalmanFilter::getEstimate(){return estimate;}
double KalmanFilter::getConfidence(){return max_confidence;}

ConstantKalmanFilter::ConstantKalmanFilter(string tag, Eigen::VectorXd initial_estimate, double confidence) : KalmanFilter(tag, initial_estimate, confidence) {}

shared_ptr<ConstantKalmanFilter> ConstantKalmanFilter::copy(string new_tag)
{
    return static_pointer_cast<ConstantKalmanFilter>(KalmanFilter::copy(new_tag));
}

void ConstantKalmanFilter::updateEstimate(Eigen::VectorXd zk, double confidence)
{
    Eigen::VectorXd newEstimate = estimate;
    Eigen::MatrixXd PkEstimate = Pk + Q;

    Eigen::MatrixXd kGain = PkEstimate*(PkEstimate+R).inverse();

    estimate = newEstimate+getEstimateConfidence(confidence)*kGain*(zk-newEstimate);
    Pk = (Eigen::MatrixXd::Identity(dataDim,dataDim) - kGain)*PkEstimate;
}

void ConstantKalmanFilter::updateEstimate(Eigen::VectorXd zk, Eigen::VectorXd uk, double confidence){updateEstimate(zk, confidence);}
