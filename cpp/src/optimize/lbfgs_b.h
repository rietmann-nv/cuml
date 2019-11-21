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

enum LBFGSB_RESULT {
  LBFGSB_INCOMPLETE,
  LBFGSB_STOP_GTOL,
  LBFGSB_STOP_ITER,
  LBFGSB_STOP_MAXLS,
  LBFGSB_STOP_FTOL
};

// How to update `pk`: BFGS or L-BFGS
enum LBFGS_PK_METHOD {
  LBFGSB_PK_LBFGS,  // Standard L-BFGS 2-loop recursion
  LBFGSB_PK_BFGS    // Use BFGS with full inverse hessian
};

std::string LBFGSB_RESULT_STRING(LBFGSB_RESULT res);

class Batched_LBFGS_B {
 public:
  void minimizeNewton(std::function<void(const std::vector<Eigen::VectorXd>& x,
                                         std::vector<double>& fx)>
                        f,
                      std::function<void(const std::vector<Eigen::VectorXd>& x,
                                         std::vector<Eigen::VectorXd>& gfx)>
                        g,
                      std::function<void(const std::vector<Eigen::VectorXd>& x,
                                         std::vector<Eigen::MatrixXd>& hfx)>
                        hess,
                      std::vector<Eigen::VectorXd>& x,
                      std::vector<LBFGSB_RESULT>& status, std::string& info_str,
                      std::vector<std::vector<Eigen::VectorXd>>& xk_all);

  void minimize(std::function<void(const std::vector<Eigen::VectorXd>& x,
                                   std::vector<double>& fx)>
                  f,
                std::function<void(const std::vector<Eigen::VectorXd>& x,
                                   std::vector<Eigen::VectorXd>& gfx)>
                  g,
                const std::vector<double>& fx0,
                const std::vector<Eigen::VectorXd>& gx0,
                std::vector<Eigen::VectorXd>& x,
                std::vector<LBFGSB_RESULT>& status, std::string& info_str,
                std::vector<std::vector<Eigen::VectorXd>>& xk_all);

  Batched_LBFGS_B(int verbosity = -1, int maxiter = 1000, int M = 10,
                  double pgtol = 1e-5, double factr = 1e7, int maxls = 20,
                  LBFGS_PK_METHOD method = LBFGSB_PK_LBFGS);

 private:
  void compute_pk_single_bfgs(const Eigen::VectorXd& gk,
                              const Eigen::VectorXd& sk,
                              const Eigen::VectorXd& yk, Eigen::MatrixXd& Hk,
                              Eigen::Ref<Eigen::VectorXd> pk);
  void compute_pk_single(const Eigen::VectorXd& gk,
                         const std::deque<Eigen::VectorXd>& sk,
                         const std::deque<Eigen::VectorXd>& yk,
                         Eigen::Ref<Eigen::VectorXd> pk, int k);

  ////////////////////////////////////////////////////////////////////////////////
  // Runtime control parameters
  int
    m_verbosity;  // -1,0=silent, 1-100 (output every N steps), >100 (maximum output every step)
  int m_maxiter;   // stop when 'L-BFGS iter' > maxiter
  int m_M;         // number of previous gradients to store
  double m_pgtol;  // stop when ||g|| < pgtol
  double m_factr;  // stop when ||f_k-f_k+1|| < factr * MACHINE_EPS

  int m_maxls;  // reset storage when "line-search iterations" > maxls
  std::vector<std::vector<std::pair<int, int>>>
    m_bounds;                   // parameter box constraints
  LBFGS_PK_METHOD m_pk_method;  // L-BFGS (default) or BFGS

  ////////////////////////////////////////////////////////////////////////////////
  // Internal State
  int m_k;                                          // current iteration
  std::vector<Eigen::VectorXd> m_xk;                // current parameters
  std::vector<std::deque<Eigen::VectorXd>> m_M_sk;  // storage sk (column-major)
  std::vector<std::deque<Eigen::VectorXd>> m_M_yk;  // storage yk (column-major)
  std::vector<Eigen::VectorXd> m_gk;                // storage yk (column-major)
  std::vector<Eigen::VectorXd> m_x0;                // storage yk (column-major)
};

}  // namespace Optimization
}  // namespace MLCommon
