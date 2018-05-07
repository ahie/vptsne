#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tsne_loss.h"
#include "cuda.h"
#include "curand.h"
#include "curand_kernel.h"

#define INIT_KERNEL_IDX(max)                       \
  int idx = blockIdx.x * blockDim.x + threadIdx.x; \
  if (idx >= max) return;

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

template <typename T>
struct NNElement {
  T distance;
  int index;
  bool flag;
};

__device__ void lock(int* mutex) {
  while(atomicCAS(mutex, 0, 1) != 0);
  __threadfence();
}

__device__ void unlock(int* mutex) {
  __threadfence();
  atomicExch(mutex, 0);
}

__global__ void InitRngStates(
  unsigned long long seed, curandState_t* rng_states, int n_datapoints) {

  INIT_KERNEL_IDX(n_datapoints);

  curand_init(seed, idx, 0, &rng_states[idx]);
}

// TODO: support other distance functions
template <typename T>
__device__ T Distance(const T* x, int i, int j, int n_dimensions) {

  const T* a = &x[i * n_dimensions];
  const T* b = &x[j * n_dimensions];

  T distance{};
  for (int k = 0; k < n_dimensions; ++k) {
    distance += (a[k] - b[k]) * (a[k] - b[k]);
  }

  return distance;
}


template <typename T>
__global__ void InitNearestNeighbors(
  const T* x, NNElement<T>* B,
  curandState_t* rng_states,
  TsneLossParams params) {

  INIT_KERNEL_IDX(params.n_datapoints);

  int n_neighbors = params.n_neighbors;
  int n_datapoints = params.n_datapoints;
  int n_dimensions = params.n_dimensions;

  for (int i = 0; i < n_neighbors; ++i) {
    bool reject = true;
    int candidate = 0;

    while (reject) {
      reject = false;
      candidate = curand(&rng_states[idx]) % n_datapoints;
      for (int j = 0; j < i; ++j) {
        if (candidate == B[idx * n_neighbors + j].index) {
          reject = true;
          break;
        }
      }
    }

    T distance = Distance(x, idx, candidate, n_dimensions);
    B[idx * n_neighbors + i] = { distance, candidate, true };
  }

  for (int i = 1; i < n_neighbors; ++i) {
    int index = i;

    while (index > 0) {
      int parent_index;

      if (index % 2 == 0) {
        parent_index = index / 2;
      } else {
        parent_index = (index - 1) / 2;
      }

      if (B[idx * n_neighbors + parent_index].distance < B[idx * n_neighbors + index].distance) {
        NNElement<T> tmp = B[idx * n_neighbors + parent_index];
        B[idx * n_neighbors + parent_index] = B[idx * n_neighbors + index];
        B[idx * n_neighbors + index] = tmp;
        index = parent_index;
      } else {
        break;
      }
    }
  }
}

__global__ void InitCandidates(
  int* Old, int* New,
  int* OldIdx, int* NewIdx,
  int n_max_candidates, int n_datapoints) {

  INIT_KERNEL_IDX(n_datapoints);

  OldIdx[idx] = 0;
  NewIdx[idx] = 0;

  for (int i = 0; i < n_max_candidates; ++i) {
    Old[idx * n_max_candidates + i] = -1;
    New[idx * n_max_candidates + i] = -1;
  }
}

__device__ bool Push(int* Arr, int to_push, int* search_start, int max) {
  for (int i = *search_start; i < max; ++i) {
    if(atomicCAS(&Arr[i], -1, to_push) == -1) {
      *search_start = i + 1;
      return true;
    }
  }
  return false;
}

template <typename T>
__global__ void PopulateCandidates(
  NNElement<T>* B,
  int* Old, int* New,
  int* OldIdx, int* NewIdx,
  TsneLossParams params,
  curandState_t* rng_states) {

  INIT_KERNEL_IDX(params.n_datapoints);

  int n_max = params.n_max_candidates;
  int n_neighbors = params.n_neighbors;

  for (int i = 0; i < n_neighbors; ++i) {
    NNElement<T> nn_element = B[idx * n_neighbors + i];
    float rand = curand_uniform(&rng_states[idx]);

    if (rand < params.rho) {
      if (nn_element.flag) {

        bool inserted = false;
        inserted |= Push(&New[idx * n_max], nn_element.index, &NewIdx[idx], n_max);
        inserted |= Push(&New[nn_element.index * n_max], idx, &NewIdx[nn_element.index], n_max);

        if (inserted) {
          B[idx * n_neighbors + i].flag = false;
        }

      } else {
        Push(&Old[idx * n_max], nn_element.index, &OldIdx[idx], n_max);
        Push(&Old[nn_element.index * n_max], idx, &OldIdx[nn_element.index], n_max);
      }
    }
  }
}

template <typename T>
__global__ void HeapifyCandidates(
  const T* x, int* Candidates, TsneLossParams params) {

  INIT_KERNEL_IDX(params.n_datapoints);

  for (int i = 1; i < params.n_max_candidates; ++i) {
    int index = i;
    int offset = idx * params.n_max_candidates + index;

    if (Candidates[offset] < 0) {
      break;
    }

    T dist_to_index = Distance(x, idx, Candidates[offset], params.n_dimensions);

    while (index > 0) {
      int parent_index;

      if (index % 2 == 0) {
        parent_index = index / 2;
      } else {
        parent_index = (index - 1) / 2;
      }

      int parent_offset = idx * params.n_max_candidates + parent_index;
      T dist_to_parent = Distance(x, idx, Candidates[parent_offset], params.n_dimensions);

      if (dist_to_parent < dist_to_index) {
        int tmp = Candidates[offset];
        Candidates[offset] = Candidates[parent_offset];
        Candidates[parent_offset] = tmp;
        index = parent_index;
      } else {
        break;
      }
    }
  }
}

template <typename T>
__device__ int UpdateHeap(NNElement<T>* B, int n_neighbors, NNElement<T> elem) {

  if (B[0].distance <= elem.distance) {
    return 0;
  }

  for (int i = 0; i < n_neighbors; ++i) {
    if (elem.index == B[i].index) {
      return 0;
    }
  }

  B[0] = elem;

  int index = 0;
  int swap_index;
  while (true) {

    int child_index_1 = 2 * index + 1;
    int child_index_2 = child_index_1 + 1;

    if (child_index_1 >= n_neighbors) {
      break;
    } else if (child_index_2 >= n_neighbors) {
      if (B[child_index_1].distance > elem.distance) {
        swap_index = child_index_1;
      } else {
        break;
      }
    } else if (B[child_index_1].distance >= B[child_index_2].distance) {
      if (B[child_index_1].distance > elem.distance) {
        swap_index = child_index_1;
      } else {
        break;
      }
    } else if (B[child_index_2].distance > elem.distance) {
      swap_index = child_index_2;
    } else {
      break;
    }

    B[index] = B[swap_index];
    index = swap_index;
  }

  B[index] = elem;
  return 1;
}

template <typename T>
__global__ void UpdateNeighbors(
  const T* x, NNElement<T>* B,
  int* B_mutex, int* Old, int* New,
  TsneLossParams params, int* c) {

  INIT_KERNEL_IDX(params.n_datapoints);

  int n_max_candidates = params.n_max_candidates;
  int n_neighbors = params.n_neighbors;
  int n_dimensions = params.n_dimensions;

  #define WARPLOCK(mutex, critical_section) \
    for (int w = 0; w < 32; ++w) {          \
      if ((idx % 32) == w) {                \
        lock(&B_mutex[mutex]);              \
        int inserted = critical_section;    \
        if (inserted) {                     \
          atomicAdd(c, inserted);           \
        }                                   \
        unlock(&B_mutex[mutex]);            \
      }                                     \
    }

  for (int i = 0; i < n_max_candidates; ++i) {
    int u1 = New[idx * n_max_candidates + i];
    if (u1 < 0) {
      continue;
    }

    for (int j = i; j < n_max_candidates; ++j) {
      int u2 = New[idx * n_max_candidates + j];
      if (u2 < 0) {
        continue;
      }

      T distance = Distance(x, u1, u2, n_dimensions);
      NNElement<T> elem_u1 = { distance, u1, true };
      NNElement<T> elem_u2 = { distance, u2, true };

      WARPLOCK(u1, UpdateHeap(&B[u1 * n_neighbors], n_neighbors, elem_u2));
      WARPLOCK(u2, UpdateHeap(&B[u2 * n_neighbors], n_neighbors, elem_u1));
    }

    for (int j = 0; j < n_max_candidates; ++j) {
      int u2 = Old[idx * n_max_candidates + j];
      if (u2 < 0) {
        continue;
      }

      T distance = Distance(x, u1, u2, n_dimensions);
      NNElement<T> elem_u1 = { distance, u1, true };
      NNElement<T> elem_u2 = { distance, u2, true };

      WARPLOCK(u1, UpdateHeap(&B[u1 * n_neighbors], n_neighbors, elem_u2));
      WARPLOCK(u2, UpdateHeap(&B[u2 * n_neighbors], n_neighbors, elem_u1));
    }
  }
  #undef WARPLOCK
}

template <typename T>
void RunNNDescent(
  const GPUDevice& d,
  const T* x, NNElement<T>* B, curandState_t* rng_states,
  const TsneLossParams& params) {

  int n_datapoints = params.n_datapoints;
  int n_max_candidates = params.n_max_candidates;
  int n_max_iters = params.n_max_iters;

  int block_size = 256;
  int n_blocks = (n_datapoints + block_size - 1) / block_size;

  InitNearestNeighbors<<<n_blocks, block_size, 0, d.stream()>>>(x, B, rng_states, params);

  int *Old, *New, *OldIdx, *NewIdx;
  cudaMallocManaged(&Old, n_datapoints * n_max_candidates * sizeof(int));
  cudaMallocManaged(&New, n_datapoints * n_max_candidates * sizeof(int));
  cudaMallocManaged(&OldIdx, n_datapoints * sizeof(int));
  cudaMallocManaged(&NewIdx, n_datapoints * sizeof(int));

  int *B_mutex;
  cudaMallocManaged(&B_mutex, n_datapoints * sizeof(int));
  cudaMemset(B_mutex, 0, n_datapoints * sizeof(int));

  int *c, c_host;
  cudaMallocManaged(&c, sizeof(int));

  for (int i = 0; i < n_max_iters; ++i) {
    cudaMemset(c, 0, sizeof(int));

    InitCandidates<<<n_blocks, block_size, 0, d.stream()>>>(Old, New, OldIdx, NewIdx, n_max_candidates, n_datapoints);
    PopulateCandidates<<<n_blocks, block_size, 0, d.stream()>>>(B, Old, New, OldIdx, NewIdx, params, rng_states);
    HeapifyCandidates<<<n_blocks, block_size, 0, d.stream()>>>(x, New, params);
    UpdateNeighbors<<<n_blocks, block_size, 0, d.stream()>>>(x, B, B_mutex, Old, New, params, c);

    cudaMemcpy(&c_host, c, sizeof(int), cudaMemcpyDeviceToHost);
    if (c_host < params.delta * n_datapoints * params.n_neighbors) {
      break;
    }
  }

  cudaFree(Old);
  cudaFree(New);
  cudaFree(OldIdx);
  cudaFree(NewIdx);
  cudaFree(B_mutex);
  cudaFree(c);
}

template <typename T>
__device__ T CalculatePij(
  const T* sum_p, const T* betas, const T* x,
  int n_dimensions, int n_datapoints, int i, int j) {
    T input_space_distance = Distance(x, j, i, n_dimensions);

    T p_jci = exp(-input_space_distance * betas[j]) / sum_p[j];
    p_jci = isnan(p_jci) || isinf(p_jci) ? T{} : p_jci;

    T p_icj = exp(-input_space_distance * betas[i]) / sum_p[i];
    p_icj = isnan(p_icj) || isinf(p_icj) ? T{} : p_icj;

    return (p_icj + p_jci) / (2 * n_datapoints);
}

template <typename T>
__global__ void CalculateBetas(
  const T* x, TsneLossParams params, T* betas, T* sum_p) {

  INIT_KERNEL_IDX(params.n_datapoints);

  int n_datapoints = params.n_datapoints;
  int n_dimensions = params.n_input_dimensions;
  float perplexity = params.perplexity;

  // TODO: pass these in as parameters
  float tolerance = 0.0001;
  int n_max_iters = 30;

  float log_perplexity = log(perplexity);

  T beta(-1);
  T beta_min(-1);
  T beta_max(-1);

  // find initial guess for beta that is proportional to
  // the largest dist_sq to avoid e^(-dist_sq * beta)
  // being rounded to zero
  T max_dist_sq(-1);
  for (int i = 0; i < n_datapoints; ++i) {
    if (i == idx) {
      continue;
    }
    T dist_sq = Distance(x, idx, i, n_dimensions);
    if (dist_sq > max_dist_sq) {
      max_dist_sq = dist_sq;
    }
  }
  beta = 10 / max_dist_sq;

  T sum_exp{};
  T sum_exp_sq{};

  float abs_diff = tolerance * 10;
  int n_iters = 0;

  while (abs_diff > tolerance && n_iters < n_max_iters) {

    n_iters += 1;

    sum_exp = T{};
    sum_exp_sq = T{};

    for (int i = 0; i < n_datapoints; ++i) {
      if (i == idx) {
        continue;
      }
      T dist_sq = Distance(x, idx, i, n_dimensions);
      sum_exp += exp(-dist_sq * beta);
      sum_exp_sq += exp(-dist_sq * beta) * dist_sq;
    }

    float h = (float) (log(sum_exp) + (beta * sum_exp_sq) / sum_exp);
    float diff = h - log_perplexity;

    if (diff > 0) {
      beta_min = beta;
      if (beta_max < 0) {
        beta = beta * 2;
      } else {
        beta = (beta + beta_max) / 2;
      }
    } else {
      beta_max = beta;
      if (beta_min < 0) {
        beta = beta / 2;
      } else {
        beta = (beta + beta_min) / 2;
      }
    }

    abs_diff = abs(diff);
  }

  sum_p[idx] = sum_exp;
  betas[idx] = beta;
}

template <typename T>
__global__ void CalculateBetas(
  const NNElement<T>* B, TsneLossParams params, T* betas, T* sum_p) {

  INIT_KERNEL_IDX(params.n_datapoints);

  int n_neighbors = params.n_neighbors;
  float perplexity = params.perplexity;

  // TODO: pass these in as parameters
  float tolerance = 0.0001;
  int n_max_iters = 30;

  float log_perplexity = log(perplexity);

  T beta(-1);
  T beta_min(-1);
  T beta_max(-1);

  // find initial guess for beta that is proportional to
  // the largest dist_sq to avoid e^(-dist_sq * beta)
  // being rounded to zero
  T max_dist_sq(-1);
  for (int i = 0; i < n_neighbors; ++i) {
    if (B[idx * n_neighbors + i].index == idx) {
      continue;
    }
    T dist_sq = B[idx * n_neighbors + i].distance;
    if (dist_sq > max_dist_sq) {
      max_dist_sq = dist_sq;
    }
  }
  beta = 10 / max_dist_sq;

  T sum_exp{};
  T sum_exp_sq{};

  float abs_diff = tolerance * 10;
  int n_iters = 0;

  while (abs_diff > tolerance && n_iters < n_max_iters) {

    n_iters += 1;

    sum_exp = T{};
    sum_exp_sq = T{};

    for (int i = 0; i < n_neighbors; ++i) {
      if (B[idx * n_neighbors + i].index == idx) {
        continue;
      }
      T dist_sq = B[idx * n_neighbors + i].distance;
      sum_exp += exp(-dist_sq * beta);
      sum_exp_sq += exp(-dist_sq * beta) * dist_sq;
    }

    float h = (float) (log(sum_exp) + (beta * sum_exp_sq) / sum_exp);
    float diff = h - log_perplexity;

    if (diff > 0) {
      beta_min = beta;
      if (beta_max < 0) {
        beta = beta * 2;
      } else {
        beta = (beta + beta_max) / 2;
      }
    } else {
      beta_max = beta;
      if (beta_min < 0) {
        beta = beta / 2;
      } else {
        beta = (beta + beta_min) / 2;
      }
    }

    abs_diff = abs(diff);
  }

  sum_p[idx] = sum_exp;
  betas[idx] = beta;
}

template <typename T>
__global__ void CalculateSumQ(const T* x, TsneLossParams params, T* sum_q) {

  INIT_KERNEL_IDX(params.n_datapoints);

  T row_sum{};
  for (int i = 0; i < params.n_datapoints; ++i) {
    if (i == idx) {
      continue;
    }
    row_sum += 1 / (1 + Distance(x, idx, i, params.n_output_dimensions));
  }

  atomicAdd(sum_q, row_sum);
}

template <typename T>
__global__ void CalculateLoss(
  const T* x, const T* y,
  const T* betas, const T* sum_p,
  const T* sum_q, TsneLossParams params, T* loss) {

  INIT_KERNEL_IDX(params.n_datapoints);

  int i = idx;
  int n_datapoints = params.n_datapoints;
  int n_dimensions = params.n_output_dimensions;

  T loss_sum{};

  for (int j = 0; j < n_datapoints; ++j) {

    if (i == j) {
      continue;
    }

    T p_ij = CalculatePij(sum_p, betas, x, params.n_input_dimensions, n_datapoints, i, j);

    if (p_ij <= 0) {
      continue;
    }

    T dist_squared_ij = Distance(y, i, j, n_dimensions);
    T inv_dist_squared_ij = 1 / (1 + dist_squared_ij);
    T q_ij = inv_dist_squared_ij / *sum_q;

    loss_sum += p_ij * (log(p_ij) - log(q_ij));
  }

  atomicAdd(loss, loss_sum);
}

template <typename T>
__global__ void CalculateGradients(
  int n_input_dimensions, int n_output_dimensions, int n_datapoints,
  const T* x, const T* y,
  const T* betas, const T* sum_p, const T* sum_q, T* grad) {

  INIT_KERNEL_IDX(n_datapoints);

  int i = idx;

  for (int j = 0; j < n_datapoints; ++j) {

    if (i == j) {
      continue;
    }

    T p_ij = CalculatePij(sum_p, betas, x, n_input_dimensions, n_datapoints, i, j);

    T dist_squared_ij = Distance(y, i, j, n_output_dimensions);
    T inv_dist_squared_ij = 1 / (1 + dist_squared_ij);
    T q_ij = inv_dist_squared_ij / *sum_q;

    for (int d = 0; d < n_output_dimensions; ++d) {
      grad[i * n_output_dimensions + d] +=
        (p_ij - q_ij) *
        (y[i * n_output_dimensions + d] - y[j * n_output_dimensions + d]) *
        inv_dist_squared_ij;
    }
  }

  for (int d = 0; d < n_output_dimensions; ++d) {
    grad[i * n_output_dimensions + d] *= T(4);
  }
}

template <typename T>
struct TsneLossGradFunctor<GPUDevice, T> {
  void operator()(
    const GPUDevice& d,
    int n_input_dimensions, int n_output_dimensions, int n_datapoints,
    const T* x, const T* y, const T* betas, const T* sum_p, const T* sum_q, T* grad) {

    int block_size = 256;
    int n_blocks = (n_datapoints + block_size - 1) / block_size;

    cudaMemset(grad, 0, n_datapoints * n_output_dimensions * sizeof(T));
    CalculateGradients<<<block_size, n_blocks, 0, d.stream()>>>(
      n_input_dimensions, n_output_dimensions, n_datapoints, x, y, betas, sum_p, sum_q, grad);
  }
};

template <typename T>
struct TsneLossFunctor<GPUDevice, T> {
  void operator()(
    const GPUDevice& d,
    const TsneLossParams& params,
    const T* x, const T* y, T* loss, T* betas, T* sum_p, T* sum_q) {

    int block_size = 256;
    int n_blocks = (params.n_datapoints + block_size - 1) / block_size;

    if (params.exact) {
      CalculateBetas<<<block_size, n_blocks, 0, d.stream()>>>(x, params, betas, sum_p);
    } else {
      NNElement<T>* Bx;
      cudaMallocManaged(&Bx, params.n_datapoints * params.n_neighbors * sizeof(NNElement<T>));

      TsneLossParams nn_descent_params = params;
      nn_descent_params.n_dimensions = params.n_input_dimensions;

      curandState_t* rng_states;
      cudaMallocManaged(&rng_states, params.n_datapoints * sizeof(curandState_t));
      InitRngStates<<<n_blocks, block_size, 0, d.stream()>>>(0, rng_states, params.n_datapoints);

      RunNNDescent(d, x, Bx, rng_states, nn_descent_params);
      CalculateBetas<<<block_size, n_blocks, 0, d.stream()>>>(Bx, params, betas, sum_p);

      cudaFree(Bx);
      cudaFree(rng_states);
    }

    CalculateSumQ<<<block_size, n_blocks, 0, d.stream()>>>(y, params, sum_q);

    cudaMemset(loss, 0, sizeof(T));
    CalculateLoss<<<block_size, n_blocks, 0, d.stream()>>>(x, y, betas, sum_p, sum_q, params, loss);
  }
};

template struct TsneLossGradFunctor<GPUDevice, float>;
template struct TsneLossFunctor<GPUDevice, float>;

#undef INIT_KERNEL_IDX
#endif // GOOGLE_CUDA

