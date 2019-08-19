#ifndef ARIMA_BATCHED_KALMAN_H
#define ARIMA_BATCHED_KALMAN_H

#include <string>
#include <vector>

// reference implementation
void batched_kalman_filter_cpu(
  const std::vector<double*>&
    h_ys_b,  // { vector size batches, each item size nobs }
  int nobs,
  const std::vector<double*>&
    h_Zb,  // { vector size batches, each item size Zb }
  const std::vector<double*>&
    h_Rb,  // { vector size batches, each item size Rb }
  const std::vector<double*>&
    h_Tb,  // { vector size batches, each item size Tb }
  int r, std::vector<double>& h_loglike_b,
  std::vector<std::vector<double>>& h_vs_b,
  bool initP_with_kalman_iterations = false);

void batched_kalman_filter(double* d_ys_b, int nobs,
                           const std::vector<double>& b_ar_params,
                           const std::vector<double>& b_ma_params, int p, int q,
                           int num_batches, std::vector<double>& loglike_b,
                           std::vector<std::vector<double>>& h_vs_b,
                           bool initP_with_kalman_iterations = false);

void nvtx_range_push(std::string msg);

void nvtx_range_pop();

void batched_jones_transform(int p, int q, int batchSize, bool isInv,
                             const std::vector<double>& ar,
                             const std::vector<double>& ma,
                             std::vector<double>& Tar,
                             std::vector<double>& Tma);

#endif