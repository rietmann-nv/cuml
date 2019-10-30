

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
  g[0] = -2 * a - 4 * b * x[0] * (pow(-x[0], 2) + x[1]) + 2 * x[0];
  g[1] = -2 * pow(x[0], 2) + 2 * x[1];
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
    std::mt19937 gen(rd());
    std::normal_distribution<> da{1, 0.1};
    std::normal_distribution<> db{100, 10};
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

    Batched_LBFGS_B opt(1, 10);
    std::vector<LBFGSB_RESULT> status;
    std::vector<Eigen::VectorXd> x0(batchSize);
    for (int ib = 0; ib < batchSize; ib++) {
      x0[ib] = Eigen::VectorXd::Zero(2);
    }
    std::string info_str;
    std::vector<double> fx0;
    f(x0, fx0);
    std::vector<Eigen::VectorXd> gx0;
    gf(x0, gx0);
    opt.minimize(f, gf, fx0, gx0, x0, status, info_str);
    for (int ib = 0; ib < batchSize; ib++) {
      if (status[ib] != LBFGSB_STOP_GTOL) {
        printf(
          "MINIMIZATION WARNING: Stopped due to criterion other than "
          "||gradient||: %s\n",
          LBFGSB_RESULT_STRING(status[ib]).c_str());
      }
      printf("Res[%d]=(%e,%e)\n", ib, x0[ib][0], x0[ib][1]);
    }
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
