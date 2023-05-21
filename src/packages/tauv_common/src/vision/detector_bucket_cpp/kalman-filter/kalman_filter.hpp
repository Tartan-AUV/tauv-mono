#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <vector>
#include <string>

/**
 * General linear Kalman Filter implementation created for filtering positions of static objects in 
 * the mapping system but no backwards tie to system exists so can be generalized.
 * All notation and formulas found in: http://www.cs.unc.edu/~tracker/media/pdf/SIGGRAPH2001_CoursePack_08.pdf
**/

using namespace std;

class KalmanFilter
{
    public:
        KalmanFilter(string tag, Eigen::VectorXd initial_estimate);
        virtual ~KalmanFilter(){};

        virtual void updateEstimate(Eigen::VectorXd zk, Eigen::VectorXd uk, double confidence = 1.0); //state dependent
        virtual void updateEstimate(Eigen::VectorXd zk, double confidence = 1.0); //state independent
        double getEstimateConfidence(double confidence);

        Eigen::VectorXd getEstimate();
        double getConfidence();
        
        void reset(Eigen::VectorXd zk);
        
        Eigen::VectorXd estimate;
        Eigen::MatrixXd Pk;
        Eigen::MatrixXd Q;
        Eigen::MatrixXd R;
        Eigen::MatrixXd A;
        Eigen::MatrixXd B;
        Eigen::MatrixXd H;

        size_t dataDim;

    private:
        string tag;
        double max_confidence;
        Eigen::MatrixXd getParam(string property);
        void makeUpdates(Eigen::VectorXd newEstimate, Eigen::VectorXd zk, double confidence);
};

/**
 * Kalman filter based on constant-direct-measurement assumption of position.
 * Used in mapping static objects.
**/
class ConstantKalmanFilter : public KalmanFilter
{
    public:
        ConstantKalmanFilter(string tag, Eigen::VectorXd initial_estimate);
        void updateEstimate(Eigen::VectorXd zk, Eigen::VectorXd uk, double confidence = 1.0);
        void updateEstimate(Eigen::VectorXd zk, double confidence = 1.0);
};