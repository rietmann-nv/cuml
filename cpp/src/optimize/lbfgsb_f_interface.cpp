#include "lbfgs_b.h"

// from fortran
extern "C" void setulb_(int* n, int* m, double* x, double* l, double* u,
                        int* nbd, double* f, double* g, double* factr,
                        double* pgtol, double* wa, int* iwa, char* task,
                        int* iprint, char* csave, int* lsave, int* isave,
                        double* dsave, int* maxls);

namespace MLCommon {
namespace Optimization {

void setStr(std::vector<char>& array, std::string string) {
  for (int i = 0; i < string.size(); i++) {
    array[i] = string[i];
  }
  array[string.size()] = 0;
}

bool matchStr(const std::vector<char>& array, std::string string) {
  bool allmatch = true;
  for (int i = 0; i < string.size(); i++) {
    allmatch = array[i] == string[i];
  }
  return allmatch;
}

void Batched_LBFGS_B::minimize_fortran(
  std::function<void(const std::vector<Eigen::VectorXd>& x,
                     std::vector<double>& fx)>
    f,
  std::function<void(const std::vector<Eigen::VectorXd>& x,
                     std::vector<Eigen::VectorXd>& gfx)>
    g,
  const std::vector<double>& fx0, const std::vector<Eigen::VectorXd>& gx0,
  std::vector<Eigen::VectorXd>& x, std::vector<LBFGSB_RESULT>& status,
  std::string& info_str, std::vector<std::vector<Eigen::VectorXd>>& xk_all) {
  // working arrays needed by L-BFGS-B implementation;
  const int batchSize = x.size();
  int ndim = x[0].size();
  std::vector<double> f_b(batchSize);
  std::vector<Eigen::VectorXd> g_b(batchSize);

  std::vector<std::vector<double>> wa_b(batchSize);
  std::vector<std::vector<int>> iwa_b(batchSize);
  std::vector<std::vector<char>> task_b(batchSize);
  std::vector<std::vector<char>> csave_b(batchSize);
  std::vector<std::vector<int>> lsave_b(batchSize);
  std::vector<std::vector<int>> isave_b(batchSize);
  std::vector<std::vector<double>> dsave_b(batchSize);

  std::vector<double> f_b_tmp(batchSize);
  std::vector<Eigen::VectorXd> g_b_tmp(batchSize);

  for (int i = 0; i < batchSize; i++) {
    const int n = ndim;
    const int m = m_M;
    f_b[i] = 0.0;
    g_b[i] = Eigen::VectorXd::Zero(n);
    wa_b[i].resize(2 * m * n + 5 * n + 11 * m * m + 8 * m, 0.0);
    iwa_b[i].resize(3 * n, 0);
    task_b[i].resize(60, 0);
    setStr(task_b[i], "START");

    csave_b[i].resize(60, 0);
    lsave_b[i].resize(4, false);
    isave_b[i].resize(44, 0);
    dsave_b[i].resize(29, 0.0);

    f_b_tmp[i] = 0.0;
    g_b_tmp[i] = Eigen::VectorXd::Zero(n);
  }

  std::vector<bool> converged(batchSize, false);
  std::vector<LBFGSB_RESULT> stop_reason(batchSize);

  // nbp tells about bounds. 0 is no bound.
  std::vector<int> nbp(batchSize, 0);
  std::vector<double> lb(batchSize, 0);
  std::vector<double> ub(batchSize, 0);

  std::vector<int> n_iterations(batchSize, 0);
  while (std::all_of(converged.begin(), converged.end(),
                     [](bool v) { return !v; })) {
    for (int ib = 0; ib < batchSize; ib++) {
      if (converged[ib]) continue;

      setulb_(&ndim, &m_M, x[ib].data(), lb.data(), ub.data(), nbp.data(),
              f_b.data(), g_b[ib].data(), &m_factr, &m_pgtol, wa_b[ib].data(),
              iwa_b[ib].data(), task_b[ib].data(), &m_verbosity,
              csave_b[ib].data(), lsave_b[ib].data(), isave_b[ib].data(),
              dsave_b[ib].data(), &m_maxls);
    }

    // re-evaluate f,g every buffer
    f(x, f_b_tmp);
    g(x, g_b_tmp);
    for (int ib = 0; ib < batchSize; ib++) {
      if (converged[ib]) continue;

      // function eval required
      if (matchStr(task_b[ib], "FG")) {
        f_b[ib] = f_b_tmp[ib];
        g_b[ib] = g_b_tmp[ib];
      } else if (matchStr(task_b[ib], "NEW_X")) {
        xk_all.push_back(x);
        n_iterations[ib]++;
        if (n_iterations[ib] >= m_maxiter) {
          converged[ib] = true;
          stop_reason[ib] = LBFGSB_STOP_ITER;
        }
      } else if (matchStr(task_b[ib], "CONV")) {
        converged[ib] = true;
        stop_reason[ib] = LBFGSB_STOP_GTOL;
      } else {
        printf("MIN FAIL: %s\n", task_b[ib].data());
        converged[ib] = true;
        stop_reason[ib] = LBFGSB_STOP_ITER;
      }
    }

  }  // while
}
}  // namespace Optimization
}  // namespace MLCommon
