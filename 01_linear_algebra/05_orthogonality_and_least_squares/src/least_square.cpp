#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() {
    constexpr int N = 10000;
    constexpr int CN = 6;
    Eigen::MatrixXd A(N, CN);
    Eigen::VectorXd y(N);

    std::vector<double> x_vals, y_vals;

    double a = -0.3, b = 2.3, c = 3.5, d = 15.5, e = 4.1, f = 1.0;
    Eigen::VectorXd gt(CN);
    // gt << a, b, c, d;
    gt << a, b, c, d, e, f;

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.0, 3.0);  // mean: 0, sigma: 5 (1 sigma = 68.27%)

    for (int i = 0; i < N; ++i) {
        double x = static_cast<double>(i) / N * 10.0;
        double true_y = gt[0] * std::pow(x,5) + gt[1] * std::pow(x,4) + gt[2] * std::pow(x,3) + gt[3] * std::pow(x,2) + gt[4] * x + gt[5];
        double noisy_y = true_y + noise(generator);

        // if (i % 500 == 0) noisy_y *= 3;
        y(i) = noisy_y;
        for (int j = 0; j < CN; ++j) 
            A(i, j) = std::pow(x, CN-j-1);

        x_vals.push_back(x);
        y_vals.push_back(noisy_y);
    }

    // Least-squares solution: solve AᵗA x = Aᵗy
    Eigen::VectorXd coeffs = (A.transpose() * A).ldlt().solve(A.transpose() * y);

    // 결과 출력
    std::cout.precision(3);

    std::cout << "Estimated coefficients (a b c d):\n" << coeffs.transpose() << std::endl;
    std::cout << "\ngt: (a b c d):\n" << gt.transpose() << std::endl;
    std::cout << "\n=====" << std::endl;
    std::cout << "\ndiff: " << coeffs.transpose() - gt.transpose() << std::endl;

    // ========================================
    // Plotting
    // ========================================
    std::vector<double> y_gt_vals, y_fit_vals;
    for (const auto& x : x_vals) {
        double y_gt = gt[0] * std::pow(x,5) + gt[1] * std::pow(x,4) + gt[2] * std::pow(x,3) + gt[3] * std::pow(x,2) + gt[4] * x + gt[5];
        double y_fit = coeffs(0) * std::pow(x,5) + coeffs(1) * std::pow(x,4) + coeffs(2) * std::pow(x,3) + coeffs(3) * std::pow(x,2) + coeffs(4) * x + coeffs(5);

        y_gt_vals.push_back(y_gt);
        y_fit_vals.push_back(y_fit);
    }

    std::cout << "\n=====" << std::endl;
    Eigen::VectorXd y_pred = A * coeffs;        // 근사값 Ax̂
    Eigen::VectorXd residual = y - y_pred;      // 오차 벡터 r = b - Ax̂
    double error_norm = residual.norm();        // 오차의 유클리드 노름 (‖r‖)
    double mse = residual.squaredNorm() / N;    // 평균 제곱 오차 (MSE)

    std::cout << "\nError vector norm ‖r‖ = " << error_norm << std::endl;
    std::cout << "Mean Squared Error (MSE) = " << mse << std::endl;


    // 노이즈 점
    plt::scatter(x_vals, y_vals, 1.0);

    // gt 곡선 (실선)
    plt::named_plot("Ground Truth", x_vals, y_gt_vals, "g-");

    // 추정된 곡선 (점선)
    plt::named_plot("Least Squares", x_vals, y_fit_vals, "r--");

    plt::title("Noisy Data with Ground Truth vs. Least Squares Fit");
    plt::xlabel("x");
    plt::ylabel("y");
    plt::legend();
    plt::grid(true);
    plt::show();
}
