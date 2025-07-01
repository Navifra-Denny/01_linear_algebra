#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <cmath>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() {
    constexpr int N = 10000;
    Eigen::MatrixXd A(N, 6);
    Eigen::VectorXd y(N);

    std::vector<double> x_vals, y_vals;

    double a = -0.3, b = 2.3, c = 3.5, d = 15.5, e = 4.1, f = 1.0;
    Eigen::VectorXd gt(6);
    // gt << a, b, c, d;
    gt << a, b, c, d, e, f;

    std::default_random_engine generator;
    std::normal_distribution<double> noise(0.0, 5.0);  // 평균 0, 표준편차 5인 노이즈

    for (int i = 0; i < N; ++i) {
        double x = static_cast<double>(i) / N * 10.0;
        double true_y = gt[0] * std::pow(x,5) + gt[1] * std::pow(x,4) + gt[2] * std::pow(x,3) + gt[3] * std::pow(x,2) + gt[4] * x + gt[5];
        double noisy_y = true_y + noise(generator);

        y(i) = noisy_y;
        A(i, 0) = std::pow(x, 5);
        A(i, 1) = std::pow(x, 4);
        A(i, 2) = std::pow(x, 3);
        A(i, 3) = std::pow(x, 2);
        A(i, 4) = x;
        A(i, 5) = 1.0;

        x_vals.push_back(x);
        y_vals.push_back(noisy_y);
    }

    // Least-squares solution: solve AᵗA x = Aᵗy
    Eigen::VectorXd coeffs = (A.transpose() * A).ldlt().solve(A.transpose() * y);

    // 결과 출력
    std::cout.precision(3);

    std::cout << "Estimated coefficients (a b c d):\n" << coeffs.transpose() << std::endl;
    std::cout << "\ngt: (a b c d):\n" << gt.transpose() << std::endl;
    std::cout << "=====" << std::endl;
    std::cout << "diff: " << coeffs.transpose() - gt.transpose() << std::endl;

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
