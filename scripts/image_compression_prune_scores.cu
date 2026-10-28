#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

__global__ void removal_scores_kernel(
    int width, int height, const float2* xy, const float2* scales,
    const float* rotations, const float* colors, const float* gradients,
    const float* softness, const float* image, const float* target,
    const int* selected, bool rich, float* scores, float* max_error) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const int pixel = y * width + x;
    float weights[10], local_colors[10][3];
    float sum = 0.f;
    for (int k = 0; k < 10; ++k) {
        const int id = selected[pixel * 10 + k];
        weights[k] = 0.f;
        if (id < 0) continue;
        const float2 delta = {xy[id].x - x, xy[id].y - y};
        const float soft = rich ? softness[id] : 10.f;
        float peak = 0.f;
        weights[k] = hexel_alpha(delta, scales[id], rotations[id], 0.f,
                                {soft, soft, soft}, 0.f, peak);
        sum += weights[k];
        for (int c = 0; c < 3; ++c) {
            const float base = colors[id * 3 + c];
            local_colors[k][c] = rich ? base + 0.1f * (
                delta.x * gradients[(id * 2) * 3 + c] +
                delta.y * gradients[(id * 2 + 1) * 3 + c]) : base;
        }
    }
    const float denominator = sum + 1e-4f;
    const float error = fabsf(denominator - image[pixel * 4 + 3]);
    if (error > *max_error) atomicMax(reinterpret_cast<unsigned int*>(max_error), __float_as_uint(error));
    float before = 0.f;
    for (int c = 0; c < 3; ++c) {
        const float difference = image[pixel * 4 + c] - target[pixel * 3 + c];
        before += difference * difference / 3.f;
    }
    for (int k = 0; k < 10; ++k) {
        const int id = selected[pixel * 10 + k];
        if (id < 0 || weights[k] == 0.f) continue;
        const float remaining = fmaxf(denominator - weights[k], 1e-4f);
        float after = 0.f;
        for (int c = 0; c < 3; ++c) {
            const float removed = (image[pixel * 4 + c] * denominator -
                                   weights[k] * local_colors[k][c]) / remaining;
            const float difference = removed - target[pixel * 3 + c];
            after += difference * difference / 3.f;
        }
        atomicAdd(scores + id, after - before);
    }
}

std::vector<torch::Tensor> removal_scores_cuda(
    torch::Tensor xy, torch::Tensor scales, torch::Tensor rotations,
    torch::Tensor colors, torch::Tensor gradients, torch::Tensor softness,
    torch::Tensor image, torch::Tensor target, torch::Tensor selected, bool rich) {
    for (const auto& tensor : {xy, scales, rotations, colors, gradients, softness, image, target}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 && tensor.is_contiguous(),
                    "Prune inputs must be contiguous CUDA float32");
        TORCH_CHECK(tensor.device() == xy.device(), "Prune input devices must match");
    }
    const auto count = xy.size(0);
    TORCH_CHECK(xy.dim() == 2 && xy.size(1) == 2 && scales.sizes() == xy.sizes(),
                "Prune centers/scales must have shape (N,2)");
    TORCH_CHECK(rotations.dim() == 2 && rotations.size(0) == count && rotations.size(1) == 1,
                "Prune rotations must have shape (N,1)");
    TORCH_CHECK(colors.dim() == 2 && colors.size(0) == count && colors.size(1) == 3,
                "Prune colors must have shape (N,3)");
    TORCH_CHECK(gradients.dim() == 3 && gradients.size(0) == count && gradients.size(1) == 2 && gradients.size(2) == 3,
                "Prune color gradients must have shape (N,2,3)");
    TORCH_CHECK(softness.dim() == 1 && softness.size(0) == count, "Prune softness must have shape (N)");
    TORCH_CHECK(image.dim() == 3 && image.size(2) == 4 && target.dim() == 3 && target.size(2) == 3 &&
                image.size(0) == target.size(0) && image.size(1) == target.size(1), "Prune image/target shapes disagree");
    TORCH_CHECK(selected.is_cuda() && selected.scalar_type() == torch::kInt32 && selected.is_contiguous() &&
                selected.device() == xy.device() && selected.dim() == 3 && selected.size(0) == image.size(0) &&
                selected.size(1) == image.size(1) && selected.size(2) == 10, "Prune IDs must have shape (H,W,10), CUDA int32");
    const c10::cuda::CUDAGuard guard(xy.device());
    auto scores = torch::zeros({count}, xy.options());
    auto error = torch::zeros({}, xy.options());
    const int height = image.size(0), width = image.size(1);
    removal_scores_kernel<<<dim3((width + 15) / 16, (height + 15) / 16), dim3(16, 16), 0,
                            at::cuda::getCurrentCUDAStream()>>>(
        width, height, reinterpret_cast<const float2*>(xy.data_ptr<float>()),
        reinterpret_cast<const float2*>(scales.data_ptr<float>()), rotations.data_ptr<float>(),
        colors.data_ptr<float>(), gradients.data_ptr<float>(), softness.data_ptr<float>(),
        image.data_ptr<float>(), target.data_ptr<float>(), selected.data_ptr<int>(), rich,
        scores.data_ptr<float>(), error.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {scores, error};
}
