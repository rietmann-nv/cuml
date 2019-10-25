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

#pragma once

#include <deque>
#include <string>
#include <vector>

#include <functional>
#include "Eigen/Dense"

namespace MLCommon {
namespace Optimization {

void testf(std::function<void(double*, void*)> pywrapper, void* pyf) {
  double num = 42.0;
  double* num2 = new double[10];
  num2[0] = 43.0;
  pywrapper(num2, pyf);
}

enum LBFGSB_RESULT { LBFGSB_STOP_GTOL, LBFGSB_STOP_ITER, LBFGSB_STOP_FTOL };

class Batched_LBFGS_B {
  void minimize(const std::vector<double>& fx0,
                const std::vector<Eigen::VectorXd>& gx0,
                std::vector<Eigen::VectorXd>& x, std::string& info_str);

  Batched_LBFGS_B(int verbosity = -1, int M = 10, double pgtol = 1e-5,
                  double factr = 1e7, int maxiter = 1000, int maxls = 20);

 private:
  void compute_pk_single(const Eigen::VectorXd& gk,
                         const std::deque<Eigen::VectorXd>& sk,
                         const std::deque<Eigen::VectorXd>& yk,
                         Eigen::Ref<Eigen::VectorXd> pk);

  int
    m_verbosity;  // -1,0=silent, 1-100 (output every N steps), >100 (maximum output every step)
  int m_M;         // number of previous gradients to store
  double m_pgtol;  // stop when ||g|| < pgtol
  double m_factr;  // stop when ||f_k-f_k+1|| < factr * MACHINE_EPS
  int m_maxiter;   // stop when 'L-BFGS iter' > maxiter
  int m_maxls;     // reset storage when "line-search iterations" > maxls
  std::vector<std::vector<std::pair<int, int>>>
    m_bounds;                                       // parameter box constraints
  int m_k;                                          // current iteration
  std::vector<Eigen::VectorXd> m_xk;                // current parameters
  std::vector<std::deque<Eigen::VectorXd>> m_M_sk;  // storage sk (column-major)
  std::vector<std::deque<Eigen::VectorXd>> m_M_yk;  // storage yk (column-major)
  std::vector<Eigen::VectorXd> m_gk;                // storage yk (column-major)
  std::vector<Eigen::VectorXd> m_x0;                // storage yk (column-major)
};

Batched_LBFGS_B::Batched_LBFGS_B(int verbosity, int M, double pgtol,
                                 double factr, int maxiter, int maxls)
  : m_verbosity(verbosity),
    m_M(M),
    m_pgtol(pgtol),
    m_factr(factr),
    m_maxiter(maxiter),
    m_maxls(maxls) {}

/**
   Implements Alg. 7.4 from "Numerical Optimization"
 */
void Batched_LBFGS_B::compute_pk_single(const Eigen::VectorXd& gk,
                                        const std::deque<Eigen::VectorXd>& sk,
                                        const std::deque<Eigen::VectorXd>& yk,
                                        Eigen::Ref<Eigen::VectorXd> pk) {
  Eigen::VectorXd q = gk;
  Eigen::VectorXd alpha(m_M);
  Eigen::VectorXd rho(m_M);
  for (int i = m_M - 1; i >= 0; i--) {
    alpha(i) = rho(i) * sk[i].dot(q);
    q = q - alpha(i) * yk[i];
  }
  double gamma_k = sk[m_M - 1].dot(yk[m_M - 1]) / yk[m_M - 1].dot(yk[m_M - 1]);
  Eigen::VectorXd r = gamma_k * q;
  for (int i = 0; i < m_M; i++) {
    double beta = rho(i) * yk[i].dot(r);
    r = r + sk[i] * (alpha(i) - beta);
  }
  pk = -r;
}

}  // namespace Optimization
}  // namespace MLCommon
