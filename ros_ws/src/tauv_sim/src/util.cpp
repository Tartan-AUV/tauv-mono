#include "tauv_sim/util.h"

#include "Eigen/Eigenvalues"

Eigen::Matrix3d sf_to_eigen_matrix(const sf::Matrix3& m) {
    Eigen::Matrix3d e;

    auto row = m.getRow(0);
    e(0, 0) = row[0];
    e(0, 1) = row[1];
    e(0, 2) = row[2];
    row = m.getRow(1);
    e(1, 0) = row[0];
    e(1, 1) = row[1];
    e(1, 2) = row[2];
    row = m.getRow(2);
    e(2, 0) = row[0];
    e(2, 1) = row[1];
    e(2, 2) = row[2];

    return e;
}

sf::Matrix3 eigen_to_sf_matrix(const Eigen::Matrix3d& m) {
    sf::Matrix3 r;
    r[0][0] = m(0, 0);
    r[0][1] = m(0, 1);
    r[0][2] = m(0, 2);
    r[1][0] = m(1, 0);
    r[1][1] = m(1, 1);
    r[1][2] = m(1, 2);
    r[2][0] = m(2, 0);
    r[2][1] = m(2, 1);
    r[2][2] = m(2, 2);
    return r;
}

std::pair<sf::Transform, sf::Vector3> get_sf_inertia(const config::osprey::InertialBuoyancy& cfg,
                                                     const sf::Matrix3 body_R_cad) {
    auto hull_inertia_COM_B_sf = body_R_cad * cfg.hull_inertia_COM_C;
    auto I_B = sf_to_eigen_matrix(hull_inertia_COM_B_sf);

    // Symmetrize
    I_B = 0.5 * (I_B + I_B.transpose());

    // Get eigenvectors
    auto es = Eigen::EigenSolver<Eigen::Matrix3d>{I_B};
    auto eigenvectors = es.eigenvectors().real();
    auto eigenvalues = es.eigenvalues().real();

    // Ensure determinant is +1
    if (eigenvectors.determinant() < 0.0) {
        eigenvectors.col(2) *= -1.0;
    }

    //
    auto I_CG = sf::Vector3{eigenvalues(0), eigenvalues(1), eigenvalues(2)};

    auto body_R_CG = eigen_to_sf_matrix(eigenvectors);

    auto body_T_CG = sf::Transform{body_R_CG, body_R_cad * cfg.t_hull_com_C};

    return {body_T_CG, I_CG};
}
