#include "linesearch.h"
#include <iostream>
#include "f_util.h"
extern "C" void dcsrch_(double* f, double* g, double* stp, double* ftol,
                        double* gtol, double* xtol, double* stpmin,
                        double* stpmax, char* task, int* isave, double* dsave);

using std::vector;

template <class T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
  os << "[";
  for (typename std::vector<T>::const_iterator ii = v.begin(); ii != v.end();
       ++ii) {
    os << " " << *ii;
  }
  os << "]";
  return os;
}

vector<double> linesearch_backtracking(
  std::function<void(const vector<Eigen::VectorXd>& x, vector<double>& fx)>
    func,
  std::function<void(const std::vector<Eigen::VectorXd>& x,
                     std::vector<Eigen::VectorXd>& gfx)>
    grad,
  const vector<Eigen::VectorXd>& x, const vector<Eigen::VectorXd>& p,
  double alpha0, int maxls, int verbosity, LS_RESULT& result) {
  const int batchSize = p.size();
  const int ndim = x[0].size();

  vector<Eigen::VectorXd> xk = x;
  vector<Eigen::VectorXd> xkp1(batchSize);
  vector<double> fk(batchSize, 0);
  vector<double> fkp1(batchSize, 0);
  vector<double> alpha(batchSize, alpha0);

  std::vector<bool> ls_success(batchSize, false);
  for (int ils = 0; ils < maxls; ils++) {
    func(xk, fk);
    for (int ib = 0; ib < batchSize; ib++) {
      if (ls_success[ib]) continue;
      xkp1[ib] = xk[ib] + alpha[ib] * p[ib];
    }
    func(xkp1, fkp1);
    for (int ib = 0; ib < batchSize; ib++) {
      if (ls_success[ib]) continue;
      if (fkp1[ib] < fk[ib]) {
        ls_success[ib] = true;
        if (verbosity > 0) {
          printf("LS(bid=%d): line search iterations=%d\n", ib, ils);
        }
        if (verbosity >= 100)
          printf("LS(bid=%d): successful alpha=%f\n", ib, alpha[ib]);
      } else {
        if (verbosity >= 100)
          printf("LS(bid=%d): unsuccessful alpha=%f\n", ib, alpha[ib]);
      }
      alpha[ib] *= 0.5;  // shrink stepsized by half;
    }
    // if all true, stop line search
    if (std::all_of(ls_success.begin(), ls_success.end(),
                    [](bool v) { return v; })) {
      result = LS_SUCCESS;
      break;
    }

    // if we needed all line-search iterations, return with error.
    if (ils == maxls - 1) {
      result = LS_FAIL_MAXITER;
      break;
    }
  }
  return alpha;
}

vector<double> linesearch_minpack(
  std::function<void(const vector<Eigen::VectorXd>& x, vector<double>& fx)>
    func,
  std::function<void(const std::vector<Eigen::VectorXd>& x,
                     std::vector<Eigen::VectorXd>& gfx)>
    grad,
  const vector<Eigen::VectorXd>& x, const vector<Eigen::VectorXd>& p,
  double alpha0, vector<LS_RESULT>& result) {
  const int batchSize = p.size();
  const int ndim = x[0].size();

  vector<Eigen::VectorXd> p_norm = p;
  std::cout << "p=" << p << "\n";
  for (auto& pi : p_norm) {
    pi = pi / pi.norm();
  }
  // std::cout << "p_norm=" << p_norm << "\n";
  // sanity check
  for (auto& pi : p_norm) {
    std::cout << "pi=" << pi << "\n";
    std::cout << "|pi|: " << pi.norm() << "\n";
    // printf("|p|: %f\n", pi.norm());
    assert(std::abs(pi.norm() - 1.0) < 1e-12);
  }

  auto deriv_p = [&](const vector<Eigen::VectorXd>& x, vector<double>& d_x_p) {
    vector<Eigen::VectorXd> g_x;
    d_x_p.resize(batchSize, 0.0);
    grad(x, g_x);
    for (int ib = 0; ib < batchSize; ib++) {
      d_x_p[ib] = g_x[ib].dot(p[ib]);
    }
  };

  vector<double> alpha(batchSize, alpha0);  // called `stp` in fortran
  vector<double> f;
  vector<double> f0;
  func(x, f);
  f0 = f;
  // std::cout << "f(x0) = " << f << "\n";

  vector<double> f_tmp = f;
  vector<double> g;
  vector<double> g0;
  deriv_p(x, g);
  g0 = g;
  // std::cout << "f'(x0) = " << g << "\n";
  vector<double> g_tmp = g;
  double ftol = 1e-3;
  double gtol = 0.9;
  double xtol = 0.1;
  double stpmin = 0;
  double stpmax = 5;
  vector<vector<int>> isave(batchSize);
  vector<vector<double>> dsave(batchSize);
  vector<vector<char>> task(batchSize);

  for (int i = 0; i < batchSize; i++) {
    isave[i].resize(2, 0);
    dsave[i].resize(13, 0);
    task[i].resize(60, 0);
    setStr(task[i], "START");
  }

  vector<bool> ls_converged(batchSize, false);

  // default is success
  result.resize(batchSize);
  for (int i = 0; i < batchSize; i++) {
    result[i] = LS_SUCCESS;
  }

  Eigen::VectorXd ls_iter = Eigen::VectorXd::Zero(batchSize);

  while (std::all_of(ls_converged.begin(), ls_converged.end(),
                     [](bool v) { return !v; })) {
    // call linesearch
    for (int ib = 0; ib < batchSize; ib++) {
      if (ls_converged[ib]) continue;
      double alpha_k = alpha[ib];
      dcsrch_(&f[ib], &g[ib], &alpha[ib], &ftol, &gtol, &xtol, &stpmin, &stpmax,
              task[ib].data(), isave[ib].data(), dsave[ib].data());
      if (matchStr(task[ib], "CONVERGENCE")) {
        printf(
          "LS(%d): CONVERGED alpha=%e, (SDC =? %f <= %f, CC =? %f <= %f)\n", ib,
          alpha[ib], f[ib], f0[ib] + ftol * alpha_k * g0[ib], std::abs(g[ib]),
          gtol * std::abs(g0[ib]));
        ls_converged[ib] = true;
        ls_iter[ib]++;
      } else if (matchStr(task[ib], "WARN")) {
        printf("LS(%d): WARNING: %s\n", ib, task[ib].data());
        result[ib] = LS_FAIL;
        ls_converged[ib] = true;
        // std::cout << "p=" <<
      } else if (matchStr(task[ib], "FG")) {
        printf("LS(%d): FG (SDC =? %e <= %e, CC =? %e <= %e)\n", ib, f[ib],
               f0[ib] + ftol * alpha_k * g0[ib], std::abs(g[ib]),
               gtol * std::abs(g0[ib]));
      } else {
        printf("LS(%d): %s (a=%f, f=%f, g=%f)\n", ib, task[ib].data(), alpha_k,
               f[ib], g[ib]);
      }
    }

    // evaluate line-search
    // x_ap = x + alpha * p
    vector<Eigen::VectorXd> x_ap(batchSize, Eigen::VectorXd::Zero(ndim));
    for (int ib = 0; ib < batchSize; ib++) {
      if (matchStr(task[ib], "FG")) {
        x_ap[ib] = x[ib] + alpha[ib] * p[ib];
      }
    }
    func(x_ap, f_tmp);
    deriv_p(x_ap, g_tmp);

    // if line-search task == "FG", give it evaluated 'f' and 'g'
    for (int ib = 0; ib < batchSize; ib++) {
      if (matchStr(task[ib], "FG")) {
        f[ib] = f_tmp[ib];
        g[ib] = g_tmp[ib];
      }
    }
  }
  if (std::any_of(alpha.begin(), alpha.end(),
                  [](double a) { return std::isinf(a) || std::isnan(a); })) {
    printf("NaN/Inf detected!\n");
    std::cout << "alpha=" << alpha << "\n";
    assert(false);
  }

  std::cout << "LS Iterations: " << ls_iter.transpose() << "\n";
  std::cout << "alpha=" << alpha << "\n";
  return alpha;
}
