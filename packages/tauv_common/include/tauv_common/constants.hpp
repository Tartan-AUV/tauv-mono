#pragma once

namespace tauv::constants {
    constexpr double PI = 3.1415926535897931;
    constexpr double FRAC_PI_1 = 3.1415926535897931;
    constexpr double FRAC_PI_2 = 1.5707963267948966;
    constexpr double FRAC_PI_3 = 1.0471975511965976;
    constexpr double FRAC_PI_4 = 0.78539816339744828;
    constexpr double FRAC_PI_5 = 0.62831853071795862;
    constexpr double FRAC_PI_6 = 0.52359877559829882;
    constexpr double FRAC_PI_7 = 0.44879895051282759;
    constexpr double FRAC_PI_8 = 0.39269908169872414;
    constexpr double FRAC_PI_1_3 = 1.0471975511965976;
    constexpr double FRAC_PI_2_3 = 2.0943951023931953;
    constexpr double FRAC_PI_1_5 = 0.62831853071795862;
    constexpr double FRAC_PI_2_5 = 1.2566370614359172;
    constexpr double FRAC_PI_3_5 = 1.8849555921538759;
    constexpr double FRAC_PI_4_5 = 2.5132741228718345;
    constexpr double FRAC_PI_1_7 = 0.44879895051282759;
    constexpr double FRAC_PI_2_7 = 0.89759790102565518;
    constexpr double FRAC_PI_3_7 = 1.3463968515384828;
    constexpr double FRAC_PI_4_7 = 1.7951958020513104;
    constexpr double FRAC_PI_5_7 = 2.2439947525641379;
    constexpr double FRAC_PI_6_7 = 2.6927937030769655;
    constexpr double FRAC_PI_1_8 = 0.39269908169872414;
    constexpr double FRAC_PI_3_8 = 1.1780972450961724;
    constexpr double FRAC_PI_5_8 = 1.9634954084936207;
    constexpr double FRAC_PI_7_8 = 2.748893571891069;
    constexpr double SQRT_2 = 1.4142135623730951;
    constexpr double SQRT_3 = 1.7320508075688772;
    constexpr double SQRT_5 = 2.2360679774997898;
    constexpr double FRAC_1_SQRT_2 = 0.70710678118654746;
    constexpr double FRAC_1_SQRT_3 = 0.57735026918962584;
    constexpr double FRAC_1_SQRT_5 = 0.44721359549995793;
    constexpr double E = 2.7182818284590451;
    constexpr double G_EARTH_EQ = 9.8100000000000005;
    constexpr double RHO_FRESH_WATER = 997;
    constexpr double GAS_CONSTANT_R = 8.3144626180000003;
    constexpr double STANDARD_ATMOSPHERE = 101325;
    constexpr double SPEED_OF_SOUND_WATER = 1481;
    constexpr double DEG_TO_RAD = 0.017453292519943295;
    constexpr double RAD_TO_DEG = 57.295779513082323;

    // Eigen identity matrices
    const Eigen::Matrix<double, 2, 2> I_2x2 = Eigen::Matrix<double, 2, 2>::Identity();
    const Eigen::Matrix<double, 3, 3> I_3x3 = Eigen::Matrix<double, 3, 3>::Identity();
    const Eigen::Matrix<double, 4, 4> I_4x4 = Eigen::Matrix<double, 4, 4>::Identity();
    const Eigen::Matrix<double, 5, 5> I_5x5 = Eigen::Matrix<double, 5, 5>::Identity();
    const Eigen::Matrix<double, 6, 6> I_6x6 = Eigen::Matrix<double, 6, 6>::Identity();
    const Eigen::Matrix<double, 7, 7> I_7x7 = Eigen::Matrix<double, 7, 7>::Identity();
    const Eigen::Matrix<double, 8, 8> I_8x8 = Eigen::Matrix<double, 8, 8>::Identity();
    const Eigen::Matrix<double, 9, 9> I_9x9 = Eigen::Matrix<double, 9, 9>::Identity();
    
}  // namespace tauv::constants
