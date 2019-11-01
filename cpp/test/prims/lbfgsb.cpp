

/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>

#include <gtest/gtest.h>
#include <optimization/lbfgs_b.h>
#include <cstdio>
#include <random>
#include <vector>
// #include "random/rng.h"
// #include "test_utils.h"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

namespace MLCommon {
namespace Optimization {

template <typename T>
struct LBFGSBInputs {
  T tolerance;
};

using std::pow;
using std::vector;

double rosenbrock(const Eigen::VectorXd& x, double a = 1.0, double b = 100.0) {
  return pow(a - x[0], 2.0) + b * pow(x[1] - pow(x[0], 2.0), 2.0);
}

void g_rosenbrock(const Eigen::VectorXd& x, Eigen::Ref<Eigen::VectorXd> g,
                  double a = 1.0, double b = 100.0) {
  // g = np.array([-2*a - 4*b*x[0]*(-x[0]**2 + x[1]) + 2*x[0],
  // b*(-2*x[0]**2 + 2*x[1])])
  g[0] = -2.0 * a - 4 * b * x[0] * (-pow(x[0], 2.0) + x[1]) + 2.0 * x[0];
  g[1] = b * (-2.0 * pow(x[0], 2.0) + 2.0 * x[1]);
}

void h_rosenbrock(const Eigen::VectorXd& x, Eigen::Ref<Eigen::MatrixXd> H,
                  double a = 1.0, double b = 100.0) {
  // g = np.array([-2*a - 4*b*x[0]*(-x[0]**2 + x[1]) + 2*x[0],
  // b*(-2*x[0]**2 + 2*x[1])])

  H(0, 0) = 8 * b * pow(x[0], 2) - 4 * b * (-pow(x[0], 2) + x[1]) + 2;
  H(0, 1) = -4 * b * x[0];
  H(1, 0) = -4 * b * x[0];
  H(1, 1) = 2 * b;
}

void batched_rosenbrock(const vector<Eigen::VectorXd>& x, int batchSize,
                        const vector<double>& a, const vector<double>& b,
                        vector<double>& fx) {
  fx.resize(batchSize);
  for (int i = 0; i < batchSize; i++) {
    fx[i] = rosenbrock(x[i], a[i], b[i]);
  }
}

void batched_g_rosenbrock(const vector<Eigen::VectorXd>& x, int batchSize,
                          const vector<double>& a, const vector<double>& b,
                          vector<Eigen::VectorXd>& gx) {
  gx.resize(batchSize);
  for (int i = 0; i < batchSize; i++) {
    gx[i].resize(2);
    g_rosenbrock(x[i], gx[i], a[i], b[i]);
  }
}
void batched_h_rosenbrock(const vector<Eigen::VectorXd>& x, int batchSize,
                          const vector<double>& a, const vector<double>& b,
                          vector<Eigen::MatrixXd>& Hx) {
  Hx.resize(batchSize);
  for (int i = 0; i < batchSize; i++) {
    Hx[i].resize(2, 2);
    h_rosenbrock(x[i], Hx[i], a[i], b[i]);
  }
}

template <typename T>
class LBFGSBTest : public ::testing::TestWithParam<LBFGSBInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<LBFGSBInputs<T>>::GetParam();
    std::cout << "Running LBFGSB Test!\n";

    int batchSize = 2;
    vector<double> ar(batchSize);
    vector<double> br(batchSize);

    std::random_device rd{};
    // std::mt19937 gen(rd());
    std::mt19937 gen(42);
    std::normal_distribution<> da{1, 0.01};
    std::normal_distribution<> db{100, 1};
    for (int i = 0; i < batchSize; i++) {
      ar[i] = da(gen);
      br[i] = db(gen);
    }

    auto f = [&](const vector<Eigen::VectorXd>& x, vector<double>& fx) {
      batched_rosenbrock(x, batchSize, ar, br, fx);
    };
    auto gf = [&](const vector<Eigen::VectorXd>& x,
                  vector<Eigen::VectorXd>& gfx) {
      batched_g_rosenbrock(x, batchSize, ar, br, gfx);
    };
    auto hf = [&](const vector<Eigen::VectorXd>& x,
                  vector<Eigen::MatrixXd>& hfx) {
      batched_h_rosenbrock(x, batchSize, ar, br, hfx);
    };

    Batched_LBFGS_B opt(1, 100);
    std::vector<LBFGSB_RESULT> status;
    std::vector<Eigen::VectorXd> x0(batchSize);
    for (int ib = 0; ib < batchSize; ib++) {
      x0[ib].resize(2);
      x0[ib][0] = ar[ib] + 0.1;
      x0[ib][1] = ar[ib] * ar[ib] + 0.1;
    }

    // test gradient
    vector<Eigen::VectorXd> grad_fd(batchSize);
    vector<Eigen::VectorXd> grad_test(batchSize);
    double h = 1e-12;
    for (int ib = 0; ib < batchSize; ib++) {
      grad_fd[ib] = Eigen::VectorXd::Zero(2);
      for (int i = 0; i < 2; i++) {
        auto x0_0 = x0[ib][i];

        vector<double> gfx_ph;
        vector<double> gfx_mh;

        // +h
        x0[ib][i] = x0_0 + h;
        f(x0, gfx_ph);

        // -h
        x0[ib][i] = x0_0 - h;
        f(x0, gfx_mh);

        // restore x0
        x0[ib][i] = x0_0;

        // second order
        grad_fd[ib][i] = (gfx_ph[ib] - gfx_mh[ib]) / (2 * h);
      }
    }

    gf(x0, grad_test);
    printf("g=[\n");
    for (int ib = 0; ib < batchSize; ib++) {
      for (int i = 0; i < 2; i++) {
        printf("fd:%f vs an:%f,\n", grad_fd[ib][i], grad_test[ib][i]);
      }
    }
    printf("]\n");

    std::string info_str;
    std::vector<double> fx0;
    f(x0, fx0);
    std::vector<Eigen::VectorXd> gx0;
    gf(x0, gx0);
    std::vector<std::vector<Eigen::VectorXd>> xk_all;
    std::vector<std::vector<Eigen::VectorXd>> xk_Newton_all;
    std::vector<Eigen::VectorXd> x0H = x0;
    opt.minimize(f, gf, fx0, gx0, x0, status, info_str, xk_all);
    opt.minimizeNewton(f, gf, hf, x0H, status, info_str, xk_Newton_all);
    for (int ib = 0; ib < batchSize; ib++) {
      if (status[ib] != LBFGSB_STOP_GTOL) {
        printf(
          "MINIMIZATION WARNING: Stopped due to criterion other than "
          "||gradient||: %s\n",
          LBFGSB_RESULT_STRING(status[ib]).c_str());
      }
      printf("Res[%d]=(%e,%e)\n", ib, x0[ib][0], x0[ib][1]);
    }

    auto x_to_xy = [](std::vector<std::vector<Eigen::VectorXd>>& xin, int bid,
                      std::vector<double>& x, std::vector<double>& y) {
      // auto n_dof = xin[0][0].size() / 2;
      auto n_elem = xin.size();
      x.resize(n_elem);
      y.resize(n_elem);
      for (int i = 0; i < n_elem; i++) {
        x[i] = xin[i][bid][0];
        y[i] = xin[i][bid][1];
      }
    };

    // plot results
    plt::subplot(2, 1, 1);
    std::vector<double> x0t;
    std::vector<double> y0t;
    std::vector<double> x1t;
    std::vector<double> y1t;
    std::vector<double> x0Ht;
    std::vector<double> y0Ht;
    std::vector<double> x1Ht;
    std::vector<double> y1Ht;
    x_to_xy(xk_all, 0, x0t, y0t);
    x_to_xy(xk_all, 1, x1t, y1t);
    x_to_xy(xk_Newton_all, 0, x0Ht, y0Ht);
    x_to_xy(xk_Newton_all, 1, x1Ht, y1Ht);
    plt::plot(x0t, y0t, "k-o", x0Ht, y0Ht, "g-o");
    plt::plot({ar[0]}, {ar[0] * ar[0]}, "r*");
    plt::subplot(2, 1, 2);
    plt::plot(x1t, y1t, "k-", x1Ht, y1Ht, "g-");
    plt::plot({ar[1]}, {ar[1] * ar[0]}, "r*");
    plt::show();
  }
  LBFGSBInputs<T> params;
};

using LBFGSBTestD = LBFGSBTest<double>;
TEST_P(LBFGSBTestD, Result) { std::cout << "Finished Test\n"; }

const vector<LBFGSBInputs<double>> inputsd = {{1e-6}};

INSTANTIATE_TEST_CASE_P(LBFGSBInputs, LBFGSBTestD,
                        ::testing::ValuesIn(inputsd));

}  // namespace Optimization
}  // namespace MLCommon
