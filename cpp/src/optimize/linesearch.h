#include <functional>
#include <vector>
#include "Eigen/Dense"

enum LS_RESULT {
  LS_SUCCESS,
  LS_FAIL_XTOL,
  LS_FAIL_STPMIN,
  LS_FAIL_MAXITER,
  LS_FAIL
};

std::vector<double> linesearch_minpack(
  std::function<void(const std::vector<Eigen::VectorXd>& x,
                     std::vector<double>& fx)>
    func,
  std::function<void(const std::vector<Eigen::VectorXd>& x,
                     std::vector<Eigen::VectorXd>& gfx)>
    grad,
  const std::vector<Eigen::VectorXd>& x, const std::vector<Eigen::VectorXd>& p,
  double alpha0, int verbosity, std::vector<LS_RESULT>& result);

std::vector<double> linesearch_backtracking(
  std::function<void(const std::vector<Eigen::VectorXd>& x,
                     std::vector<double>& fx)>
    func,
  std::function<void(const std::vector<Eigen::VectorXd>& x,
                     std::vector<Eigen::VectorXd>& gfx)>
    grad,
  const std::vector<Eigen::VectorXd>& x, const std::vector<Eigen::VectorXd>& p,
  double alpha0, int maxls, int verbosity, LS_RESULT& result);
