#include "lbfgs_b.h"

namespace MLCommon {
namespace Optimization {

void setulb_(int* n, int* m, double* x, double* l, double* u, int* nbd,
             double* f, double* g, double* factr, double* pgtol, double* wa,
             int* iwa, char* task, int* iprint, char* csave, int* lsave,
             int* isave, double* dsave, int* maxls);

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
  std::vector<std::vector<double>> g_b(batchSize);
  std::vector<std::vector<double>> wa_b(batchSize);
  std::vector<std::vector<int>> iwa_b(batchSize);
  std::vector<std::vector<char>> task_b(batchSize);
  std::vector<std::vector<char>> csave_b(batchSize);
  std::vector<std::vector<int>> lsave_b(batchSize);
  std::vector<std::vector<int>> isave_b(batchSize);
  std::vector<std::vector<double>> dsave_b(batchSize);

  for (int i = 0; i < batchSize; i++) {
    const int n = ndim;
    const int m = m_M;
    f_b[i] = 0.0;
    g_b[i].resize(n, 0.0);
    wa_b[i].resize(2 * m * n + 5 * n + 11 * m * m + 8 * m, 0.0);
    iwa_b[i].resize(3 * n, 0);
    task_b[i].resize(20, 0);
    task_b[i] = {'S', 'T', 'A', 'R', 'T', 0};
    csave_b[i].resize(20, 0);
    lsave_b[i].resize(4, false);
    isave_b[i].resize(44, 0);
    dsave_b[i].resize(29, 0.0);
  }

  std::vector<bool> converged(batchSize, false);
  std::vector<double> lb(batchSize, 0);
  std::vector<double> ub(batchSize, 0);
  int nbp = 0.0;
  for (int k = 0; k < m_maxiter; k++) {
    for (int ib = 0; ib < batchSize; ib++) {
      if (converged[ib]) continue;

      setulb_(&ndim, &m_M, x[ib].data(), lb.data(), ub.data(), &nbp, f_b.data(),
              g_b[ib].data(), &m_factr, &m_pgtol, wa_b[ib].data(),
              iwa_b[ib].data(), task_b[ib].data(), &m_verbosity,
              csave_b[ib].data(), lsave_b[ib].data(), isave_b[ib].data(),
              dsave_b[ib].data(), &m_maxls);
    }
    // std::all_of(ls_success.begin(), ls_success.end(),
    // [](bool v) { return v; })
  }

}  // namespace Optimization
}  // namespace Optimization
