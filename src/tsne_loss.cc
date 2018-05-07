#include "tsne_loss.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

#include <stdexcept>

using namespace tensorflow;

using shape_inference::InferenceContext;
using shape_inference::ScalarShape;
using errors::InvalidArgument;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename T>
struct TsneLossFunctor<CPUDevice, T> {
  void operator()(
    const CPUDevice& d,
    const TsneLossParams& params,
    const T* x, const T* y, T* betas, T* sum_p, T* sum_q, T* loss) {
    throw std::runtime_error("TsneLossFunctor does not have a CPU implementation");
  }
};

template <typename T>
struct TsneLossGradFunctor<CPUDevice, T> {
  void operator()(
    const CPUDevice& d,
    int n_input_dimensions, int n_output_dimensions, int n_datapoints,
    const T* x, const T* y, const T* betas, const T* sum_p, const T* sum_q, T* grad) {
    throw std::runtime_error("TsneLossGradFunctor does not have a CPU implementation");
  }
};

REGISTER_OP("TsneLoss")
  .Attr("perplexity: float = 15")
  .Attr("exact: bool = true")
  .Attr("delta: float = 0.001")
  .Attr("rho: float = 0.5")
  .Attr("n_neighbors: int = 45")
  .Attr("n_max_candidates: int = 60")
  .Attr("n_max_iters: int = 10")
  .Attr("T: {half, float}")
  .Input("x: T")
  .Input("y: T")
  .Output("loss: T")
  .Output("betas: T")
  .Output("sum_p: T")
  .Output("sum_q: T")
  .SetShapeFn([](InferenceContext* ctx) {
    ctx->set_output(0, ctx->Scalar());
    ctx->set_output(1, ctx->UnknownShapeOfRank(1));
    ctx->set_output(2, ctx->UnknownShapeOfRank(1));
    ctx->set_output(3, ctx->Scalar());
    return Status::OK();
  })
  .Doc(R"doc(
T-SNE Loss
)doc");

template <typename Device, typename T>
class TsneLossOp : public OpKernel {
  public:
    explicit TsneLossOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("perplexity", &perplexity_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("delta", &delta_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("rho", &rho_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("n_neighbors", &n_neighbors_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("n_max_candidates", &n_max_candidates_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("n_max_iters", &n_max_iters_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("exact", &exact_));
    }

    void Compute(OpKernelContext* ctx) override {
      const Tensor& x_tensor = ctx->input(0);
      const Tensor& y_tensor = ctx->input(1);

      OP_REQUIRES(ctx, x_tensor.dims() == 2 && y_tensor.dims() == 2,
        InvalidArgument("only 2D tensors are supported by this operation"));
      OP_REQUIRES(ctx, x_tensor.dim_size(0) == y_tensor.dim_size(0),
        InvalidArgument("shapes of tensors x and y must match along axis 0"));

      auto n_datapoints = x_tensor.dim_size(0);

      auto x = x_tensor.flat<T>();
      auto y = y_tensor.flat<T>();

      Tensor *loss, *betas, *sum_p, *sum_q;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &loss));
      OP_REQUIRES_OK(ctx, ctx->allocate_output(1, TensorShape({n_datapoints}), &betas));
      OP_REQUIRES_OK(ctx, ctx->allocate_output(2, TensorShape({n_datapoints}), &sum_p));
      OP_REQUIRES_OK(ctx, ctx->allocate_output(3, TensorShape({}), &sum_q));

      TsneLossParams params = GetParams(x_tensor, y_tensor);
      TsneLossFunctor<Device, T>()(
        ctx->eigen_device<Device>(), params,
        x.data(), y.data(),
        loss->flat<T>().data(),
        betas->flat<T>().data(),
        sum_p->flat<T>().data(),
        sum_q->flat<T>().data());
    }

  private:
    TsneLossParams GetParams(const Tensor& x_tensor, const Tensor& y_tensor) const {
      TsneLossParams params;

      params.perplexity = this->perplexity_;
      params.rho = this->rho_;
      params.delta = this->delta_;
      params.n_neighbors = this->n_neighbors_;
      params.n_max_candidates = this->n_max_candidates_;
      params.n_max_iters = this->n_max_iters_;
      params.n_datapoints = x_tensor.dim_size(0);
      params.n_input_dimensions = x_tensor.dim_size(1);
      params.n_output_dimensions = y_tensor.dim_size(1);
      params.exact = this->exact_;

      return params;
    }

    float perplexity_;
    float delta_;
    float rho_;
    int n_neighbors_;
    int n_max_candidates_;
    int n_max_iters_;
    bool exact_;
};

REGISTER_OP("TsneLossGrad")
  .Attr("T: {half, float}")
  .Input("x: T")
  .Input("y: T")
  .Input("betas: T")
  .Input("sum_p: T")
  .Input("sum_q: T")
  .Output("gradient: T")
  .SetShapeFn([](InferenceContext* ctx) {
    ctx->set_output(0, ctx->input(1));
    return Status::OK();
  })
  .Doc(R"doc(
T-SNE Loss Gradient
)doc");

template <typename Device, typename T>
class TsneLossGradOp : public OpKernel {
  public:
    explicit TsneLossGradOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
      const Tensor& x_tensor = ctx->input(0);
      const Tensor& y_tensor = ctx->input(1);

      OP_REQUIRES(ctx, x_tensor.dims() == 2 && y_tensor.dims() == 2,
        InvalidArgument("only 2D tensors are supported by this operation"));
      OP_REQUIRES(ctx, x_tensor.dim_size(0) == y_tensor.dim_size(0),
        InvalidArgument("shapes of tensors x and y must match along axis 0"));

      auto x = x_tensor.flat<T>().data();
      auto y = y_tensor.flat<T>().data();
      auto betas = ctx->input(2).flat<T>().data();
      auto sum_p = ctx->input(3).flat<T>().data();
      auto sum_q = ctx->input(4).flat<T>().data();

      Tensor* grad = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, y_tensor.shape(), &grad));

      TsneLossGradFunctor<Device, T>()(
        ctx->eigen_device<Device>(),
        x_tensor.dim_size(1), y_tensor.dim_size(1), x_tensor.dim_size(0),
        x, y, betas, sum_p, sum_q, grad->flat<T>().data());
    }
};

#define REGISTER_KB(NAME, T)                               \
  REGISTER_KERNEL_BUILDER(                                 \
    Name(#NAME).Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    NAME##Op<CPUDevice, T>);

// TODO: add cpu implementation
// REGISTER_KB(TsneLoss, float);
// REGISTER_KB(TsneLossGrad, float);

#undef REGISTER_KB

#ifdef GOOGLE_CUDA
#define REGISTER_KB_GPU(NAME, T)                           \
  REGISTER_KERNEL_BUILDER(                                 \
    Name(#NAME).Device(DEVICE_GPU).TypeConstraint<T>("T"), \
    NAME##Op<GPUDevice, T>);

REGISTER_KB_GPU(TsneLoss, float);
REGISTER_KB_GPU(TsneLossGrad, float);

#undef REGISTER_KB_GPU
#endif // GOOGLE_CUDA

