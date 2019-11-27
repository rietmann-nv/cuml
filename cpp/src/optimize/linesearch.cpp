#include "linesearch.h"
#include "f_util.h"

extern "C" void dcsrch_(double* stp, double* f, double* g, double* ftol,
                        double* gtol, double* xtol, char* task, double* stpmin,
                        double* stpmax, int* isave, double* dsave);

using std::vector;

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
                    [](bool v) { return v; }))
      break;

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
  double alpha0, LS_RESULT& result) {
  const int batchSize = p.size();
  const int ndim = x[0].size();

  vector<Eigen::VectorXd> p_norm = p;
  for (auto& pi : p_norm) {
    pi = pi / pi.norm();
  }

  // sanity check
  for (auto& pi : p_norm) {
    assert(std::abs(pi.norm() - 1.0) < 1e-12);
  }

  auto deriv_p = [&](const vector<Eigen::VectorXd>& x, vector<double>& d_x_p) {
    vector<Eigen::VectorXd> g_x;
    d_x_p.resize(batchSize, 0.0);
    grad(x, g_x);
    for (int ib = 0; ib < batchSize; ib++) {
      d_x_p[ib] = g_x[ib].dot(p_norm[ib]);
    }
  };

  vector<double> alpha(batchSize, alpha0);  // called `stp` in fortran
  vector<double> f;
  func(x, f);
  vector<double> f_tmp = f;
  vector<double> g;
  deriv_p(x, g);
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
  result = LS_SUCCESS;

  vector<int> ls_iter(batchSize, 0);

  while (std::all_of(ls_converged.begin(), ls_converged.end(),
                     [](bool v) { return !v; })) {
    // call linesearch
    for (int ib = 0; ib < batchSize; ib++) {
      if (ls_converged[ib]) continue;
      dcsrch_(&alpha[ib], &f[ib], &g[ib], &ftol, &gtol, &xtol, task[ib].data(),
              &stpmin, &stpmax, isave[ib].data(), dsave[ib].data());
      if (matchStr(task[ib], "CONVERGENCE")) {
        ls_converged[ib] = true;
      } else if (matchStr(task[ib], "WARN")) {
        result = LS_FAIL;
        ls_converged[ib] = true;
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
}
