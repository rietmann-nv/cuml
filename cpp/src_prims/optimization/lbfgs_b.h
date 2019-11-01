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

enum LBFGSB_RESULT {
  LBFGSB_INCOMPLETE,
  LBFGSB_STOP_GTOL,
  LBFGSB_STOP_ITER,
  LBFGSB_STOP_MAXLS,
  LBFGSB_STOP_FTOL
};

std::string LBFGSB_RESULT_STRING(LBFGSB_RESULT res) {
  switch (res) {
    case LBFGSB_INCOMPLETE:
      return "LBFGSB_INCOMPLETE";
    case LBFGSB_STOP_GTOL:
      return "LBFGSB_STOP_GTOL";
    case LBFGSB_STOP_FTOL:
      return "LBFGSB_STOP_FTOL";
    case LBFGSB_STOP_ITER:
      return "LBFGSB_STOP_ITER";
    case LBFGSB_STOP_MAXLS:
      return "LBFGSB_STOP_MAXLS";
    default:
      return "UKNOWN_RESULT";
  }
}

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
                  double pgtol = 1e-5, double factr = 1e7, int maxls = 20);

 private:
  void compute_pk_single_bfgs(const Eigen::VectorXd& gk,
                              const Eigen::VectorXd& sk,
                              const Eigen::VectorXd& yk, Eigen::MatrixXd& Hk,
                              Eigen::Ref<Eigen::VectorXd> pk);
  void compute_pk_single(const Eigen::VectorXd& gk,
                         const std::deque<Eigen::VectorXd>& sk,
                         const std::deque<Eigen::VectorXd>& yk,
                         Eigen::Ref<Eigen::VectorXd> pk, int k);

  int
    m_verbosity;  // -1,0=silent, 1-100 (output every N steps), >100 (maximum output every step)
  int m_maxiter;   // stop when 'L-BFGS iter' > maxiter
  int m_M;         // number of previous gradients to store
  double m_pgtol;  // stop when ||g|| < pgtol
  double m_factr;  // stop when ||f_k-f_k+1|| < factr * MACHINE_EPS

  int m_maxls;  // reset storage when "line-search iterations" > maxls
  std::vector<std::vector<std::pair<int, int>>>
    m_bounds;                                       // parameter box constraints
  int m_k;                                          // current iteration
  std::vector<Eigen::VectorXd> m_xk;                // current parameters
  std::vector<std::deque<Eigen::VectorXd>> m_M_sk;  // storage sk (column-major)
  std::vector<std::deque<Eigen::VectorXd>> m_M_yk;  // storage yk (column-major)
  std::vector<Eigen::VectorXd> m_gk;                // storage yk (column-major)
  std::vector<Eigen::VectorXd> m_x0;                // storage yk (column-major)
};

Batched_LBFGS_B::Batched_LBFGS_B(int verbosity, int maxiter, int M,
                                 double pgtol, double factr, int maxls)
  : m_verbosity(verbosity),
    m_maxiter(maxiter),
    m_M(M),
    m_pgtol(pgtol),
    m_factr(factr),
    m_maxls(maxls) {}

void Batched_LBFGS_B::minimizeNewton(
  std::function<void(const std::vector<Eigen::VectorXd>&, std::vector<double>&)>
    f,
  std::function<void(const std::vector<Eigen::VectorXd>&,
                     std::vector<Eigen::VectorXd>&)>
    g,
  std::function<void(const std::vector<Eigen::VectorXd>&,
                     std::vector<Eigen::MatrixXd>&)>
    hess,
  std::vector<Eigen::VectorXd>& x, std::vector<LBFGSB_RESULT>& status,
  std::string& info_str, std::vector<std::vector<Eigen::VectorXd>>& xk_all) {
  const int batchSize = x.size();
  const int N = x[0].size();
  status.resize(batchSize);
  xk_all.push_back(x);
  std::vector<Eigen::VectorXd> xk(batchSize);
  std::vector<Eigen::VectorXd> xkp1(batchSize);
  std::vector<Eigen::VectorXd> pk(batchSize);
  std::vector<Eigen::VectorXd> gk(batchSize);
  std::vector<Eigen::MatrixXd> hk(batchSize);
  std::vector<double> alpha(batchSize);
  std::vector<double> fk(batchSize);
  std::vector<double> fkp1(batchSize);
  for (int i = 0; i < batchSize; i++) {
    status[i] = LBFGSB_INCOMPLETE;
    gk[i].resize(N);
    xk[i] = x[i];
  }
  for (int k = 0; k < m_maxiter; k++) {
    g(xk, gk);
    hess(xk, hk);

    // compute search direction
    for (int ib = 0; ib < batchSize; ib++) {
      pk[ib] = -hk[ib].colPivHouseholderQr().solve(gk[ib]);
      alpha[ib] = 1.0;
    }

    std::vector<bool> ls_success(batchSize, false);
    for (int ils = 0; ils < m_maxls; ils++) {
      f(xk, fk);
      for (int ib = 0; ib < batchSize; ib++) {
        if (ls_success[ib]) continue;
        xkp1[ib] = xk[ib] + alpha[ib] * pk[ib];
      }
      f(xkp1, fkp1);
      for (int ib = 0; ib < batchSize; ib++) {
        if (ls_success[ib]) continue;
        if (fkp1[ib] < fk[ib]) {
          ls_success[ib] = true;
          if (m_verbosity > 0 && k % m_verbosity == 0) {
            printf("k=%d, bid=%d: line search iterations=%d\n", k, ib, ils);
          }
          if (m_verbosity >= 100)
            printf("k=%d, bid=%d: successful alpha=%f\n", k, ib, alpha[ib]);
        } else {
          if (m_verbosity >= 100)
            printf("k=%d, bid=%d: unsuccessful alpha=%f\n", k, ib, alpha[ib]);
        }
        alpha[ib] *= 0.5;  // shrink stepsized by half;
      }
      // if all true, stop line search
      if (std::all_of(ls_success.begin(), ls_success.end(),
                      [](bool v) { return v; }))
        break;

      // if we needed all line-search iterations, return with error.
      if (ils == m_maxls - 1) {
        for (int ib = 0; ib < batchSize; ib++) {
          if (ls_success[ib] == false) status[ib] = LBFGSB_STOP_MAXLS;
          x = xk;
          return;
        }
      }
    }

    // take step and update LBFGS-sk variable
    for (int ib = 0; ib < batchSize; ib++) {
      xkp1[ib] = xk[ib] + alpha[ib] * pk[ib];
    }

    // update gradient and LBFGS-gk variables
    g(xkp1, gk);

    // stopping criterion
    f(xk, fk);
    f(xkp1, fkp1);
    for (int ib = 0; ib < batchSize; ib++) {
      auto g_norm = gk[ib].norm();
      if (k % m_verbosity == 0 || m_verbosity >= 100) {
        printf("k=%d, bid=%d: |f|=%e -> %e, |g|=%e\n", k, ib, std::abs(fk[ib]),
               std::abs(fkp1[ib]), g_norm);
      }
      if (g_norm < m_pgtol) {
        status[ib] = LBFGSB_STOP_GTOL;
      }
      if (status[ib] == LBFGSB_INCOMPLETE && k == m_maxiter - 1) {
        status[ib] = LBFGSB_STOP_ITER;
      }
    }
    bool stop = true;
    for (int ib = 0; ib < batchSize; ib++) {
      if (status[ib] == LBFGSB_INCOMPLETE) {
        stop = false;
      }
    }
    if (stop) break;
    xk = xkp1;
    xk_all.push_back(xk);
  }
}

void Batched_LBFGS_B::minimize(
  std::function<void(const std::vector<Eigen::VectorXd>& x,
                     std::vector<double>& fx)>
    f,
  std::function<void(const std::vector<Eigen::VectorXd>& x,
                     std::vector<Eigen::VectorXd>& gfx)>
    g,
  const std::vector<double>& fx0, const std::vector<Eigen::VectorXd>& gx0,
  std::vector<Eigen::VectorXd>& x, std::vector<LBFGSB_RESULT>& status,
  std::string& info_str, std::vector<std::vector<Eigen::VectorXd>>& xk_all) {
  const int batchSize = x.size();
  const int N = x[0].size();
  status.resize(batchSize);

  m_M_sk.resize(batchSize);
  m_M_yk.resize(batchSize);

  xk_all.push_back(x);

  std::vector<Eigen::VectorXd> xk(batchSize);
  std::vector<Eigen::VectorXd> xkp1(batchSize);
  std::vector<Eigen::VectorXd> pk(batchSize);
  std::vector<double> alpha(batchSize);
  std::vector<double> fk(batchSize);
  std::vector<double> fkp1(batchSize);
  std::vector<Eigen::VectorXd> gk(batchSize);
  std::vector<Eigen::VectorXd> gkp1(batchSize);
  std::vector<Eigen::MatrixXd> Hk(batchSize);
  for (int i = 0; i < batchSize; i++) {
    status[i] = LBFGSB_INCOMPLETE;
    gk[i].resize(N);
    xk[i] = x[i];
    Hk[i] = Eigen::MatrixXd::Identity(N, N);
  }

  for (int k = 0; k < m_maxiter; k++) {
    for (int ib = 0; ib < batchSize; ib++) {
      // compute search direction pk
      if (k == 0) {
        // first step just does steepest descent
        pk[ib] = -gx0[ib];
        alpha[ib] = 0.05;
      } else {
        compute_pk_single(gk[ib], m_M_sk[ib], m_M_yk[ib], pk[ib], k);
        alpha[ib] = 0.5;
      }
      if (m_verbosity >= 100) {
        // std::cout << "pk=" << pk[ib] << "\n";
        printf("pk=[");
        for (int i = 0; i < pk[ib].size(); i++) {
          printf("%e,", pk[ib][i]);
        }
        printf("]\n");
      }
    }

    // line search
    // TODO: Implement good line search
    std::vector<bool> ls_success(batchSize, false);
    for (int ils = 0; ils < m_maxls; ils++) {
      f(xk, fk);
      for (int ib = 0; ib < batchSize; ib++) {
        if (ls_success[ib]) continue;
        xkp1[ib] = xk[ib] + alpha[ib] * pk[ib];
      }
      f(xkp1, fkp1);
      for (int ib = 0; ib < batchSize; ib++) {
        if (ls_success[ib]) continue;
        if (fkp1[ib] < fk[ib]) {
          ls_success[ib] = true;
          if (m_verbosity > 0 && k % m_verbosity == 0) {
            printf("k=%d, bid=%d: line search iterations=%d\n", k, ib, ils);
          }
          if (m_verbosity >= 100)
            printf("k=%d, bid=%d: successful alpha=%f\n", k, ib, alpha[ib]);
        } else {
          if (m_verbosity >= 100)
            printf("k=%d, bid=%d: unsuccessful alpha=%f\n", k, ib, alpha[ib]);
        }
        alpha[ib] *= 0.5;  // shrink stepsized by half;
      }
      // if all true, stop line search
      if (std::all_of(ls_success.begin(), ls_success.end(),
                      [](bool v) { return v; }))
        break;

      // if we needed all line-search iterations, return with error.
      if (ils == m_maxls - 1) {
        for (int ib = 0; ib < batchSize; ib++) {
          if (ls_success[ib] == false) status[ib] = LBFGSB_STOP_MAXLS;
          x = xk;
          return;
        }
      }
    }

    // take step and update LBFGS-sk variable
    for (int ib = 0; ib < batchSize; ib++) {
      xkp1[ib] = xk[ib] + alpha[ib] * pk[ib];
      m_M_sk[ib].push_back(xkp1[ib] - xk[ib]);
      m_M_sk[ib].pop_front();
    }

    // update gradient and LBFGS-gk variables
    g(xkp1, gkp1);
    for (int ib = 0; ib < batchSize; ib++) {
      m_M_yk[ib].push_back(gkp1[ib] - gk[ib]);
      m_M_yk[ib].pop_front();
      gk[ib] = gkp1[ib];
    }

    // stopping criterion
    f(xk, fk);
    f(xkp1, fkp1);
    for (int ib = 0; ib < batchSize; ib++) {
      auto g_norm = gk[ib].norm();
      if (k % m_verbosity == 0 || m_verbosity >= 100) {
        printf("k=%d, bid=%d: |f|=%e -> %e, |g|=%e\n", k, ib, std::abs(fk[ib]),
               std::abs(fkp1[ib]), g_norm);
      }
      if (g_norm < m_pgtol) {
        status[ib] = LBFGSB_STOP_GTOL;
      }
      if (status[ib] == LBFGSB_INCOMPLETE && k == m_maxiter - 1) {
        status[ib] = LBFGSB_STOP_ITER;
      }
    }
    bool stop = true;
    for (int ib = 0; ib < batchSize; ib++) {
      if (status[ib] == LBFGSB_INCOMPLETE) {
        stop = false;
      }
    }
    if (stop) break;
    xk = xkp1;
    xk_all.push_back(xk);
  }  // for k in maxiter

  x = xk;
}

/**
   Implements Alg. 7.4 from "Numerical Optimization"
 */
void Batched_LBFGS_B::compute_pk_single(const Eigen::VectorXd& gk,
                                        const std::deque<Eigen::VectorXd>& sk,
                                        const std::deque<Eigen::VectorXd>& yk,
                                        Eigen::Ref<Eigen::VectorXd> pk, int k) {
  Eigen::VectorXd q = gk;
  int M = sk.size();
  Eigen::VectorXd alpha(M);
  Eigen::VectorXd rho(M);
  for (int i = M - 1; i >= 0; i--) {
    alpha(i) = rho(i) * sk[i].dot(q);
    q = q - alpha(i) * yk[i];
  }
  double gamma_k = sk[M - 1].dot(yk[M - 1]) / yk[M - 1].dot(yk[M - 1]);
  Eigen::VectorXd r = gamma_k * q;
  for (int i = 0; i < M; i++) {
    double beta = rho(i) * yk[i].dot(r);
    r = r + sk[i] * (alpha(i) - beta);
  }
  pk = -r;
  // pk = -gk;  // use SD for now...
}

/**
   Implements Alg. 7.4 from "Numerical Optimization"
*/
void Batched_LBFGS_B::compute_pk_single_bfgs(const Eigen::VectorXd& gk,
                                             const Eigen::VectorXd& sk,
                                             const Eigen::VectorXd& yk,
                                             Eigen::MatrixXd& Hk,
                                             Eigen::Ref<Eigen::VectorXd> pk) {
  using Eigen::MatrixXd;
  int N = Hk.rows();
  double rhok = 1 / (yk.dot(sk));
  MatrixXd Hkp1 = (MatrixXd::Identity(N, N) - rhok * sk * yk.transpose()) * Hk *
                    (MatrixXd::Identity(N, N) - rhok * yk * sk.transpose()) +
                  rhok * sk * sk.transpose();
  pk = -Hkp1 * gk;  // use SD for now...
  Hk = Hkp1;
}

}  // namespace Optimization
}  // namespace MLCommon
