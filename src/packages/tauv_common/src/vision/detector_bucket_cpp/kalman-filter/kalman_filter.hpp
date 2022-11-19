#include <eigen3/Eigen/Dense>
#include <ros/ros.h>
#include <vector>
#include <string>

using namespace std;

//field notation according to: http://www.cs.unc.edu/~tracker/media/pdf/SIGGRAPH2001_CoursePack_08.pdf
//standard kalman filter
class KalmanFilter
{
    public:
        KalmanFilter(string tag, Eigen::VectorXd initial_estimate);
        virtual ~KalmanFilter(){};
        virtual void updateEstimate(Eigen::VectorXd zk, Eigen::VectorXd uk); //state dependent
        virtual void updateEstimate(Eigen::VectorXd zk); //state independent
        Eigen::VectorXd getEstimate();
        
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
        Eigen::MatrixXd getParam(string property);
        void makeUpdates(Eigen::VectorXd newEstimate, Eigen::VectorXd zk);
};

//Kalman filter based on constant-direct-measurement assumption of position
class ConstantKalmanFilter : public KalmanFilter
{
    public:
        ConstantKalmanFilter(string tag, Eigen::VectorXd initial_estimate);
        void updateEstimate(Eigen::VectorXd zk, Eigen::VectorXd uk);
        void updateEstimate(Eigen::VectorXd zk);
};