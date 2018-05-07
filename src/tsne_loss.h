#ifndef TSNE_LOSS_H_
#define TSNE_LOSS_H_

#include "tensorflow/core/framework/op_kernel.h"

struct TsneLossParams {
  float perplexity;
  float rho;
  float delta;
  int n_neighbors;
  int n_max_candidates;
  int n_max_iters;
  int n_datapoints;
  int n_input_dimensions;
  int n_output_dimensions;
  int n_dimensions;
  bool exact;
};

template <typename Device, typename T>
struct TsneLossFunctor {
  void operator()(
    const Device& d,
    const TsneLossParams& params,
    const T* x, const T* y, T* loss, T* betas, T* sum_p, T* sum_q);
};

template <typename Device, typename T>
struct TsneLossGradFunctor {
  void operator()(
    const Device& d,
    int n_input_dimensions, int n_output_dimensions, int n_datapoints,
    const T* x, const T* y, const T* betas, const T* sum_p, const T* sum_q, T* grad);
};

#endif // TSNE_LOSS_H_

