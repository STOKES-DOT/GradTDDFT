#include <cuda_runtime.h>
#include "xla/ffi/api/ffi.h"

#include <cmath>

namespace {

namespace ffi = ::xla::ffi;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kMaxBoys = 8;
constexpr int kMax1dAngular = 8;

struct Ang {
    int x;
    int y;
    int z;
};

__host__ __device__ int lsum(Ang a) {
    return a.x + a.y + a.z;
}

__host__ __device__ int min2(Ang a, Ang b) {
    int out = a.x;
    out = out < a.y ? out : a.y;
    out = out < a.z ? out : a.z;
    out = out < b.x ? out : b.x;
    out = out < b.y ? out : b.y;
    out = out < b.z ? out : b.z;
    return out;
}

__host__ __device__ int axis_value(Ang a, int axis) {
    if (axis == 0) {
        return a.x;
    }
    if (axis == 1) {
        return a.y;
    }
    return a.z;
}

__host__ __device__ Ang dec_axis(Ang a, int axis) {
    if (axis == 0) {
        --a.x;
    } else if (axis == 1) {
        --a.y;
    } else {
        --a.z;
    }
    return a;
}

__host__ __device__ Ang inc2_axis(Ang a, int axis) {
    if (axis == 0) {
        a.x += 2;
    } else if (axis == 1) {
        a.y += 2;
    } else {
        a.z += 2;
    }
    return a;
}

__host__ __device__ Ang dec2_axis(Ang a, int axis) {
    if (axis == 0) {
        a.x -= 2;
    } else if (axis == 1) {
        a.y -= 2;
    } else {
        a.z -= 2;
    }
    return a;
}

__host__ __device__ int first_axis(Ang a) {
    if (a.x > 0) {
        return 0;
    }
    if (a.y > 0) {
        return 1;
    }
    return 2;
}

__host__ __device__ double dist2(const double* a, const double* b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

__host__ __device__ double primitive_cart_norm(double alpha, Ang angular) {
    const int ltot = lsum(angular);
    const double prefactor = pow(2.0 * alpha / kPi, 0.75);
    if (ltot == 0) {
        return prefactor;
    }
    if (ltot == 1) {
        return prefactor * sqrt(4.0 * alpha);
    }
    double factorial_l_plus_1 = 1.0;
    for (int value = 2; value <= ltot + 1; ++value) {
        factorial_l_plus_1 *= static_cast<double>(value);
    }
    double factorial_2l_plus_2 = 1.0;
    for (int value = 2; value <= 2 * ltot + 2; ++value) {
        factorial_2l_plus_2 *= static_cast<double>(value);
    }
    const double numerator = pow(2.0, 2 * ltot + 3) * factorial_l_plus_1;
    const double denominator = factorial_2l_plus_2 * sqrt(kPi);
    return sqrt(numerator * pow(2.0 * alpha, ltot + 1.5) / denominator);
}

__host__ __device__ double boys0(double t) {
    if (t < 1.0e-8) {
        double term = 1.0;
        double factorial = 1.0;
        double total = 0.0;
        for (int k = 0; k <= 12; ++k) {
            if (k > 0) {
                factorial *= static_cast<double>(k);
                term *= -t;
            }
            total += term / (factorial * static_cast<double>(2 * k + 1));
        }
        return total;
    }
    const double sqrt_t = sqrt(t);
    return 0.5 * sqrt(kPi) * erf(sqrt_t) / sqrt_t;
}

__host__ __device__ void boys_values(int max_n, double t, double* values) {
    max_n = max_n < kMaxBoys ? max_n : kMaxBoys;
    if (t < 1.0e-8) {
        for (int n = 0; n <= max_n; ++n) {
            double term = 1.0;
            double factorial = 1.0;
            double total = 0.0;
            for (int k = 0; k < 18; ++k) {
                if (k > 0) {
                    factorial *= static_cast<double>(k);
                    term *= -t;
                }
                total += term / (factorial * static_cast<double>(2 * n + 2 * k + 1));
            }
            values[n] = total;
        }
        return;
    }

    const double f0 = boys0(t);
    values[0] = f0;
    if (t > 50.0) {
        const double exp_neg_t = exp(-t);
        for (int n = 0; n < max_n; ++n) {
            values[n + 1] =
                ((2.0 * static_cast<double>(n) + 1.0) * values[n] - exp_neg_t) /
                (2.0 * t);
        }
        return;
    }
    const int work_max = max_n + 24;
    double work[kMaxBoys + 25] = {};
    work[work_max] = 0.0;
    const double exp_neg_t = exp(-t);
    for (int n = work_max; n > 0; --n) {
        work[n - 1] = (2.0 * t * work[n] + exp_neg_t) / static_cast<double>(2 * n - 1);
    }
    const double scale = work[0] != 0.0 ? f0 / work[0] : 1.0;
    for (int n = 1; n <= max_n; ++n) {
        values[n] = work[n] * scale;
    }
}

__host__ __device__ double overlap_1d(int la, int lb, double alpha, double beta, double a, double b) {
    if (la < 0 || lb < 0 || la > kMax1dAngular || lb > kMax1dAngular) {
        return 0.0;
    }
    const double p = alpha + beta;
    const double mu = alpha * beta / p;
    const double pcenter = (alpha * a + beta * b) / p;
    const double pa = pcenter - a;
    const double pb = pcenter - b;
    const double ab = a - b;
    double table[kMax1dAngular + 1][kMax1dAngular + 1] = {};
    table[0][0] = sqrt(kPi / p) * exp(-mu * ab * ab);
    for (int i = 1; i <= la; ++i) {
        table[i][0] = pa * table[i - 1][0];
        if (i > 1) {
            table[i][0] += static_cast<double>(i - 1) / (2.0 * p) * table[i - 2][0];
        }
    }
    for (int j = 1; j <= lb; ++j) {
        for (int i = 0; i <= la; ++i) {
            double value = pb * table[i][j - 1];
            if (i > 0) {
                value += static_cast<double>(i) / (2.0 * p) * table[i - 1][j - 1];
            }
            if (j > 1) {
                value += static_cast<double>(j - 1) / (2.0 * p) * table[i][j - 2];
            }
            table[i][j] = value;
        }
    }
    return table[la][lb];
}

__host__ __device__ double primitive_overlap(
    double alpha,
    double beta,
    const double* center_a,
    const double* center_b,
    Ang angular_a,
    Ang angular_b
) {
    return overlap_1d(angular_a.x, angular_b.x, alpha, beta, center_a[0], center_b[0]) *
           overlap_1d(angular_a.y, angular_b.y, alpha, beta, center_a[1], center_b[1]) *
           overlap_1d(angular_a.z, angular_b.z, alpha, beta, center_a[2], center_b[2]);
}

__host__ __device__ double primitive_kinetic(
    double alpha,
    double beta,
    const double* center_a,
    const double* center_b,
    Ang angular_a,
    Ang angular_b
) {
    double value = beta * static_cast<double>(2 * lsum(angular_b) + 3) *
                   primitive_overlap(alpha, beta, center_a, center_b, angular_a, angular_b);
    for (int axis = 0; axis < 3; ++axis) {
        const Ang b_plus_2 = inc2_axis(angular_b, axis);
        value -= 2.0 * beta * beta *
                 primitive_overlap(alpha, beta, center_a, center_b, angular_a, b_plus_2);
        const int b_axis = axis_value(angular_b, axis);
        if (b_axis >= 2) {
            const Ang b_minus_2 = dec2_axis(angular_b, axis);
            value -= 0.5 * static_cast<double>(b_axis * (b_axis - 1)) *
                     primitive_overlap(alpha, beta, center_a, center_b, angular_a, b_minus_2);
        }
    }
    return value;
}

struct NuclearContext {
    double p;
    const double* center_a;
    const double* center_b;
    const double* center_c;
    double center_p[3];
    double boys[kMaxBoys + 1];
};

__host__ __device__ double nuclear_rec(NuclearContext& ctx, Ang a, Ang b, int m);

__host__ __device__ double nuclear_rec(NuclearContext& ctx, Ang a, Ang b, int m) {
    if (min2(a, b) < 0 || m > kMaxBoys) {
        return 0.0;
    }
    if (lsum(a) + lsum(b) == 0) {
        return ctx.boys[m];
    }
    if (lsum(a) > 0) {
        const int axis = first_axis(a);
        const Ang a1 = dec_axis(a, axis);
        double out = (ctx.center_p[axis] - ctx.center_a[axis]) * nuclear_rec(ctx, a1, b, m);
        out -= (ctx.center_p[axis] - ctx.center_c[axis]) * nuclear_rec(ctx, a1, b, m + 1);
        if (axis_value(a1, axis) > 0) {
            const Ang a2 = dec_axis(a1, axis);
            const double coef = static_cast<double>(axis_value(a1, axis)) / (2.0 * ctx.p);
            out += coef * (nuclear_rec(ctx, a2, b, m) - nuclear_rec(ctx, a2, b, m + 1));
        }
        if (axis_value(b, axis) > 0) {
            const Ang b1 = dec_axis(b, axis);
            const double coef = static_cast<double>(axis_value(b, axis)) / (2.0 * ctx.p);
            out += coef * (nuclear_rec(ctx, a1, b1, m) - nuclear_rec(ctx, a1, b1, m + 1));
        }
        return out;
    }

    const int axis = first_axis(b);
    const Ang b1 = dec_axis(b, axis);
    double out = (ctx.center_p[axis] - ctx.center_b[axis]) * nuclear_rec(ctx, a, b1, m);
    out -= (ctx.center_p[axis] - ctx.center_c[axis]) * nuclear_rec(ctx, a, b1, m + 1);
    if (axis_value(b1, axis) > 0) {
        const Ang b2 = dec_axis(b1, axis);
        const double coef = static_cast<double>(axis_value(b1, axis)) / (2.0 * ctx.p);
        out += coef * (nuclear_rec(ctx, a, b2, m) - nuclear_rec(ctx, a, b2, m + 1));
    }
    if (axis_value(a, axis) > 0) {
        const Ang a1 = dec_axis(a, axis);
        const double coef = static_cast<double>(axis_value(a, axis)) / (2.0 * ctx.p);
        out += coef * (nuclear_rec(ctx, a1, b1, m) - nuclear_rec(ctx, a1, b1, m + 1));
    }
    return out;
}

__host__ __device__ double primitive_nuclear_for_atom(
    double alpha,
    double beta,
    const double* center_a,
    const double* center_b,
    const double* center_c,
    double charge,
    Ang angular_a,
    Ang angular_b
) {
    NuclearContext ctx;
    ctx.p = alpha + beta;
    ctx.center_a = center_a;
    ctx.center_b = center_b;
    ctx.center_c = center_c;
    const double mu = alpha * beta / ctx.p;
    const double rab2 = dist2(center_a, center_b);
    for (int axis = 0; axis < 3; ++axis) {
        ctx.center_p[axis] = (alpha * center_a[axis] + beta * center_b[axis]) / ctx.p;
    }
    const double rpc2 = dist2(ctx.center_p, center_c);
    double boys_raw[kMaxBoys + 1] = {};
    boys_values(kMaxBoys, ctx.p * rpc2, boys_raw);
    const double prefactor = -charge * 2.0 * kPi / ctx.p * exp(-mu * rab2);
    for (int n = 0; n <= kMaxBoys; ++n) {
        ctx.boys[n] = prefactor * boys_raw[n];
    }
    return nuclear_rec(ctx, angular_a, angular_b, 0);
}

__host__ __device__ void contracted_overlap_hcore(
    int max_nprim,
    int natom,
    const int* nprims,
    const int* angulars,
    const double* centers,
    const double* exponents,
    const double* coefficients,
    const double* atom_coords,
    const double* atom_charges,
    int i,
    int j,
    double* overlap_out,
    double* hcore_out
) {
    const double* center_i = centers + 3 * i;
    const double* center_j = centers + 3 * j;
    const Ang angular_i{angulars[3 * i], angulars[3 * i + 1], angulars[3 * i + 2]};
    const Ang angular_j{angulars[3 * j], angulars[3 * j + 1], angulars[3 * j + 2]};
    double overlap_value = 0.0;
    double hcore_value = 0.0;
    for (int ip = 0; ip < nprims[i]; ++ip) {
        const double alpha = exponents[i * max_nprim + ip];
        const double weight_i = coefficients[i * max_nprim + ip];
        for (int jp = 0; jp < nprims[j]; ++jp) {
            const double beta = exponents[j * max_nprim + jp];
            const double weight_j = coefficients[j * max_nprim + jp];
            const double weight = weight_i * weight_j;
            const double overlap = primitive_overlap(alpha, beta, center_i, center_j, angular_i, angular_j);
            double nuclear = 0.0;
            for (int atom = 0; atom < natom; ++atom) {
                nuclear += primitive_nuclear_for_atom(
                    alpha,
                    beta,
                    center_i,
                    center_j,
                    atom_coords + 3 * atom,
                    atom_charges[atom],
                    angular_i,
                    angular_j);
            }
            const double kinetic = primitive_kinetic(alpha, beta, center_i, center_j, angular_i, angular_j);
            overlap_value += weight * overlap;
            hcore_value += weight * (kinetic + nuclear);
        }
    }
    *overlap_out = overlap_value;
    *hcore_out = hcore_value;
}

__global__ void one_electron_kernel(
    int nao,
    int max_nprim,
    int natom,
    const int* nprims,
    const int* angulars,
    const double* centers,
    const double* exponents,
    const double* coefficients,
    const double* atom_coords,
    const double* atom_charges,
    double* overlap,
    double* hcore
) {
    const long long total = static_cast<long long>(nao) * nao;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int i = static_cast<int>(idx / nao);
        const int j = static_cast<int>(idx % nao);
        double s = 0.0;
        double h = 0.0;
        contracted_overlap_hcore(
            max_nprim,
            natom,
            nprims,
            angulars,
            centers,
            exponents,
            coefficients,
            atom_coords,
            atom_charges,
            i,
            j,
            &s,
            &h);
        overlap[i * nao + j] = s;
        hcore[i * nao + j] = h;
    }
}

ffi::Error CudaOneElectronDispatch(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64, 2> centers,
    ffi::Buffer<ffi::S32, 2> angulars,
    ffi::Buffer<ffi::F64, 2> exponents,
    ffi::Buffer<ffi::F64, 2> coefficients,
    ffi::Buffer<ffi::S32, 1> nprims,
    ffi::Buffer<ffi::F64, 2> atom_coords,
    ffi::Buffer<ffi::F64, 1> atom_charges,
    ffi::Result<ffi::Buffer<ffi::F64, 2>> overlap_out,
    ffi::Result<ffi::Buffer<ffi::F64, 2>> hcore_out) {
    const int64_t nao64 = centers.dimensions()[0];
    if (centers.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("centers must have shape (nao, 3)");
    }
    if (angulars.dimensions()[0] != nao64 || angulars.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("angulars must have shape (nao, 3)");
    }
    if (exponents.dimensions()[0] != nao64 || coefficients.dimensions()[0] != nao64) {
        return ffi::Error::InvalidArgument("primitive arrays must have leading dimension nao");
    }
    if (exponents.dimensions()[1] != coefficients.dimensions()[1]) {
        return ffi::Error::InvalidArgument("exponents and coefficients must share max_nprim");
    }
    if (nprims.dimensions()[0] != nao64) {
        return ffi::Error::InvalidArgument("nprims must have shape (nao,)");
    }
    if (atom_coords.dimensions()[1] != 3 || atom_coords.dimensions()[0] != atom_charges.dimensions()[0]) {
        return ffi::Error::InvalidArgument("atom coords/charges shape mismatch");
    }

    const int nao = static_cast<int>(nao64);
    const int max_nprim = static_cast<int>(exponents.dimensions()[1]);
    const int natom = static_cast<int>(atom_coords.dimensions()[0]);
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    const long long total = static_cast<long long>(nao) * nao;
    const int block = 128;
    const int grid = total > 0 ? static_cast<int>((total + block - 1) / block) : 1;
    one_electron_kernel<<<grid, block, 0, stream>>>(
        nao,
        max_nprim,
        natom,
        nprims.typed_data(),
        angulars.typed_data(),
        centers.typed_data(),
        exponents.typed_data(),
        coefficients.typed_data(),
        atom_coords.typed_data(),
        atom_charges.typed_data(),
        overlap_out->typed_data(),
        hcore_out->typed_data());
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

using StreamCtx = ffi::PlatformStream<cudaStream_t>;
using F64Buffer2 = ffi::Buffer<ffi::F64, 2>;
using F64Buffer1 = ffi::Buffer<ffi::F64, 1>;
using S32Buffer2 = ffi::Buffer<ffi::S32, 2>;
using S32Buffer1 = ffi::Buffer<ffi::S32, 1>;

auto CudaOneElectronBinding() {
    return ffi::Ffi::Bind()
        .Ctx<StreamCtx>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer2>()
        .Arg<F64Buffer2>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer1>()
        .Arg<F64Buffer2>()
        .Arg<F64Buffer1>()
        .Ret<F64Buffer2>()
        .Ret<F64Buffer2>();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TdGraddftCudaOneElectronFfi,
    CudaOneElectronDispatch,
    CudaOneElectronBinding());
