#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kBoysMax = 12;

struct Ang {
    int x;
    int y;
    int z;
};

void check_cuda(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        std::cerr << label << ": " << cudaGetErrorString(status) << "\n";
        std::exit(2);
    }
}

__host__ __device__ int lsum(Ang a) {
    return a.x + a.y + a.z;
}

__host__ __device__ int min4(Ang a, Ang b, Ang c, Ang d) {
    int out = a.x;
    out = out < a.y ? out : a.y;
    out = out < a.z ? out : a.z;
    out = out < b.x ? out : b.x;
    out = out < b.y ? out : b.y;
    out = out < b.z ? out : b.z;
    out = out < c.x ? out : c.x;
    out = out < c.y ? out : c.y;
    out = out < c.z ? out : c.z;
    out = out < d.x ? out : d.x;
    out = out < d.y ? out : d.y;
    out = out < d.z ? out : d.z;
    return out;
}

__host__ __device__ int min2(Ang a, Ang c) {
    int out = a.x;
    out = out < a.y ? out : a.y;
    out = out < a.z ? out : a.z;
    out = out < c.x ? out : c.x;
    out = out < c.y ? out : c.y;
    out = out < c.z ? out : c.z;
    return out;
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

__host__ __device__ Ang inc_axis(Ang a, int axis) {
    if (axis == 0) {
        ++a.x;
    } else if (axis == 1) {
        ++a.y;
    } else {
        ++a.z;
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

__host__ __device__ int axis_value(Ang a, int axis) {
    if (axis == 0) {
        return a.x;
    }
    if (axis == 1) {
        return a.y;
    }
    return a.z;
}

__host__ __device__ double axis_value3(const double* value, int axis) {
    return value[axis];
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
        for (int k = 0; k <= 10; ++k) {
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
    max_n = max_n < kBoysMax ? max_n : kBoysMax;
    if (t < 1.0e-8) {
        for (int n = 0; n <= max_n; ++n) {
            double term = 1.0;
            double factorial = 1.0;
            double total = 0.0;
            for (int k = 0; k < 16; ++k) {
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
    const int work_max = max_n + 24;
    double work[kBoysMax + 25] = {};
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

__host__ __device__ double dist2(const double* a, const double* b) {
    const double dx = a[0] - b[0];
    const double dy = a[1] - b[1];
    const double dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

struct PrimitiveContext {
    double p;
    double q;
    double bra_ratio;
    double ket_ratio;
    const double* center_a;
    const double* center_b;
    const double* center_c;
    const double* center_d;
    double center_p[3];
    double center_q[3];
    double center_w[3];
    double boys[kBoysMax + 1];
};

__host__ __device__ double vrr(PrimitiveContext& ctx, Ang a, Ang c, int m);
__host__ __device__ double hrr(PrimitiveContext& ctx, Ang a, Ang b, Ang c, Ang d);

__host__ __device__ double vrr(PrimitiveContext& ctx, Ang a, Ang c, int m) {
    if (min2(a, c) < 0) {
        return 0.0;
    }
    if (lsum(a) + lsum(c) == 0) {
        return ctx.boys[m];
    }

    if (lsum(a) > 0) {
        const int axis = first_axis(a);
        const Ang a1 = dec_axis(a, axis);
        double out = (axis_value3(ctx.center_p, axis) - axis_value3(ctx.center_a, axis)) * vrr(ctx, a1, c, m);
        out += (axis_value3(ctx.center_w, axis) - axis_value3(ctx.center_p, axis)) * vrr(ctx, a1, c, m + 1);
        if (axis_value(a1, axis) > 0) {
            const Ang a2 = dec_axis(a1, axis);
            const double coef = static_cast<double>(axis_value(a1, axis)) / (2.0 * ctx.p);
            out += coef * (vrr(ctx, a2, c, m) - ctx.bra_ratio * vrr(ctx, a2, c, m + 1));
        }
        if (axis_value(c, axis) > 0) {
            const Ang c1 = dec_axis(c, axis);
            const double coef = static_cast<double>(axis_value(c, axis)) / (2.0 * (ctx.p + ctx.q));
            out += coef * vrr(ctx, a1, c1, m + 1);
        }
        return out;
    }

    const int axis = first_axis(c);
    const Ang c1 = dec_axis(c, axis);
    double out = (axis_value3(ctx.center_q, axis) - axis_value3(ctx.center_c, axis)) * vrr(ctx, a, c1, m);
    out += (axis_value3(ctx.center_w, axis) - axis_value3(ctx.center_q, axis)) * vrr(ctx, a, c1, m + 1);
    if (axis_value(c1, axis) > 0) {
        const Ang c2 = dec_axis(c1, axis);
        const double coef = static_cast<double>(axis_value(c1, axis)) / (2.0 * ctx.q);
        out += coef * (vrr(ctx, a, c2, m) - ctx.ket_ratio * vrr(ctx, a, c2, m + 1));
    }
    if (axis_value(a, axis) > 0) {
        const Ang a1 = dec_axis(a, axis);
        const double coef = static_cast<double>(axis_value(a, axis)) / (2.0 * (ctx.p + ctx.q));
        out += coef * vrr(ctx, a1, c1, m + 1);
    }
    return out;
}

__host__ __device__ double hrr(PrimitiveContext& ctx, Ang a, Ang b, Ang c, Ang d) {
    if (min4(a, b, c, d) < 0) {
        return 0.0;
    }
    if (lsum(b) > 0) {
        const int axis = first_axis(b);
        const Ang b1 = dec_axis(b, axis);
        const Ang a1 = inc_axis(a, axis);
        return hrr(ctx, a1, b1, c, d) +
               (axis_value3(ctx.center_a, axis) - axis_value3(ctx.center_b, axis)) * hrr(ctx, a, b1, c, d);
    }
    if (lsum(d) > 0) {
        const int axis = first_axis(d);
        const Ang d1 = dec_axis(d, axis);
        const Ang c1 = inc_axis(c, axis);
        return hrr(ctx, a, b, c1, d1) +
               (axis_value3(ctx.center_c, axis) - axis_value3(ctx.center_d, axis)) * hrr(ctx, a, b, c, d1);
    }
    return vrr(ctx, a, c, 0);
}

__host__ __device__ double primitive_sp_eri(
    double alpha,
    double beta,
    double gamma,
    double delta,
    const double* center_a,
    const double* center_b,
    const double* center_c,
    const double* center_d,
    Ang angular_a,
    Ang angular_b,
    Ang angular_c,
    Ang angular_d
) {
    PrimitiveContext ctx;
    ctx.p = alpha + beta;
    ctx.q = gamma + delta;
    ctx.bra_ratio = ctx.q / (ctx.p + ctx.q);
    ctx.ket_ratio = ctx.p / (ctx.p + ctx.q);
    ctx.center_a = center_a;
    ctx.center_b = center_b;
    ctx.center_c = center_c;
    ctx.center_d = center_d;
    const double mu = alpha * beta / ctx.p;
    const double nu = gamma * delta / ctx.q;
    const double rab2 = dist2(center_a, center_b);
    const double rcd2 = dist2(center_c, center_d);
    for (int axis = 0; axis < 3; ++axis) {
        ctx.center_p[axis] = (alpha * center_a[axis] + beta * center_b[axis]) / ctx.p;
        ctx.center_q[axis] = (gamma * center_c[axis] + delta * center_d[axis]) / ctx.q;
        ctx.center_w[axis] = (ctx.p * ctx.center_p[axis] + ctx.q * ctx.center_q[axis]) / (ctx.p + ctx.q);
    }
    const double rpq2 = dist2(ctx.center_p, ctx.center_q);
    const double prefactor = 2.0 * pow(kPi, 2.5) / (ctx.p * ctx.q * sqrt(ctx.p + ctx.q)) *
                             exp(-mu * rab2 - nu * rcd2);
    double boys_raw[kBoysMax + 1] = {};
    const int total_l = lsum(angular_a) + lsum(angular_b) + lsum(angular_c) + lsum(angular_d);
    boys_values(kBoysMax, (ctx.p * ctx.q / (ctx.p + ctx.q)) * rpq2, boys_raw);
    for (int n = 0; n <= kBoysMax; ++n) {
        ctx.boys[n] = prefactor * boys_raw[n];
    }
    return hrr(ctx, angular_a, angular_b, angular_c, angular_d);
}

__host__ __device__ double contracted_sp_eri(
    int max_nprim,
    const int* nprims,
    const int* angulars,
    const double* centers,
    const double* exponents,
    const double* coefficients,
    int i,
    int j,
    int k,
    int l
) {
    const double* center_i = centers + 3 * i;
    const double* center_j = centers + 3 * j;
    const double* center_k = centers + 3 * k;
    const double* center_l = centers + 3 * l;
    const Ang angular_i{angulars[3 * i], angulars[3 * i + 1], angulars[3 * i + 2]};
    const Ang angular_j{angulars[3 * j], angulars[3 * j + 1], angulars[3 * j + 2]};
    const Ang angular_k{angulars[3 * k], angulars[3 * k + 1], angulars[3 * k + 2]};
    const Ang angular_l{angulars[3 * l], angulars[3 * l + 1], angulars[3 * l + 2]};
    double value = 0.0;
    for (int ip = 0; ip < nprims[i]; ++ip) {
        const double alpha = exponents[i * max_nprim + ip];
        const double weight_i = coefficients[i * max_nprim + ip] * primitive_cart_norm(alpha, angular_i);
        for (int jp = 0; jp < nprims[j]; ++jp) {
            const double beta = exponents[j * max_nprim + jp];
            const double weight_j = coefficients[j * max_nprim + jp] * primitive_cart_norm(beta, angular_j);
            for (int kp = 0; kp < nprims[k]; ++kp) {
                const double gamma = exponents[k * max_nprim + kp];
                const double weight_k = coefficients[k * max_nprim + kp] * primitive_cart_norm(gamma, angular_k);
                for (int lp = 0; lp < nprims[l]; ++lp) {
                    const double delta = exponents[l * max_nprim + lp];
                    const double weight_l = coefficients[l * max_nprim + lp] * primitive_cart_norm(delta, angular_l);
                    value += weight_i * weight_j * weight_k * weight_l *
                             primitive_sp_eri(
                                 alpha,
                                 beta,
                                 gamma,
                                 delta,
                                 center_i,
                                 center_j,
                                 center_k,
                                 center_l,
                                 angular_i,
                                 angular_j,
                                 angular_k,
                                 angular_l
                             );
                }
            }
        }
    }
    return value;
}

__global__ void sp_direct_jk_kernel(
    int nao,
    int max_nprim,
    const int* nprims,
    const int* angulars,
    const double* centers,
    const double* exponents,
    const double* coefficients,
    const double* density,
    double* j_mat,
    double* k_mat
) {
    const long long total = static_cast<long long>(nao) * nao * nao * nao;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long tmp = idx;
        const int l = static_cast<int>(tmp % nao);
        tmp /= nao;
        const int k = static_cast<int>(tmp % nao);
        tmp /= nao;
        const int j = static_cast<int>(tmp % nao);
        tmp /= nao;
        const int i = static_cast<int>(tmp);
        const double eri = contracted_sp_eri(
            max_nprim,
            nprims,
            angulars,
            centers,
            exponents,
            coefficients,
            i,
            j,
            k,
            l
        );
        atomicAdd(j_mat + i * nao + j, eri * density[k * nao + l]);
        atomicAdd(k_mat + i * nao + k, eri * density[j * nao + l]);
    }
}

template <typename T>
void read_vector(std::ifstream& in, std::vector<T>& values) {
    for (auto& value : values) {
        in >> value;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: cuda_s_shell_direct_jk INPUT REPEATS OUTPUT\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const int repeats = std::max(1, std::atoi(argv[2]));
    const std::string output_path = argv[3];

    std::ifstream input(input_path);
    if (!input) {
        std::cerr << "failed to open input: " << input_path << "\n";
        return 1;
    }

    int nao = 0;
    int max_nprim = 0;
    input >> nao >> max_nprim;
    if (nao < 0 || max_nprim < 0) {
        std::cerr << "invalid dimensions\n";
        return 1;
    }

    std::vector<int> h_nprims(nao);
    std::vector<int> h_angulars(static_cast<size_t>(nao) * 3);
    std::vector<double> h_centers(static_cast<size_t>(nao) * 3);
    std::vector<double> h_exponents(static_cast<size_t>(nao) * max_nprim);
    std::vector<double> h_coefficients(static_cast<size_t>(nao) * max_nprim);
    std::vector<double> h_density(static_cast<size_t>(nao) * nao);
    read_vector(input, h_nprims);
    read_vector(input, h_angulars);
    read_vector(input, h_centers);
    read_vector(input, h_exponents);
    read_vector(input, h_coefficients);
    read_vector(input, h_density);

    int* d_nprims = nullptr;
    int* d_angulars = nullptr;
    double* d_centers = nullptr;
    double* d_exponents = nullptr;
    double* d_coefficients = nullptr;
    double* d_density = nullptr;
    double* d_j = nullptr;
    double* d_k = nullptr;
    const size_t int_bytes = h_nprims.size() * sizeof(int);
    const size_t angular_bytes = h_angulars.size() * sizeof(int);
    const size_t centers_bytes = h_centers.size() * sizeof(double);
    const size_t prim_bytes = h_exponents.size() * sizeof(double);
    const size_t matrix_bytes = h_density.size() * sizeof(double);

    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    check_cuda(cudaDeviceSetLimit(cudaLimitStackSize, 16384), "set recursion stack");
    check_cuda(cudaMalloc(&d_nprims, int_bytes), "cudaMalloc nprims");
    check_cuda(cudaMalloc(&d_angulars, angular_bytes), "cudaMalloc angulars");
    check_cuda(cudaMalloc(&d_centers, centers_bytes), "cudaMalloc centers");
    check_cuda(cudaMalloc(&d_exponents, prim_bytes), "cudaMalloc exponents");
    check_cuda(cudaMalloc(&d_coefficients, prim_bytes), "cudaMalloc coefficients");
    check_cuda(cudaMalloc(&d_density, matrix_bytes), "cudaMalloc density");
    check_cuda(cudaMalloc(&d_j, matrix_bytes), "cudaMalloc j");
    check_cuda(cudaMalloc(&d_k, matrix_bytes), "cudaMalloc k");

    check_cuda(cudaMemcpy(d_nprims, h_nprims.data(), int_bytes, cudaMemcpyHostToDevice), "copy nprims");
    check_cuda(cudaMemcpy(d_angulars, h_angulars.data(), angular_bytes, cudaMemcpyHostToDevice), "copy angulars");
    check_cuda(cudaMemcpy(d_centers, h_centers.data(), centers_bytes, cudaMemcpyHostToDevice), "copy centers");
    check_cuda(cudaMemcpy(d_exponents, h_exponents.data(), prim_bytes, cudaMemcpyHostToDevice), "copy exponents");
    check_cuda(cudaMemcpy(d_coefficients, h_coefficients.data(), prim_bytes, cudaMemcpyHostToDevice), "copy coefficients");
    check_cuda(cudaMemcpy(d_density, h_density.data(), matrix_bytes, cudaMemcpyHostToDevice), "copy density");

    const long long total = static_cast<long long>(nao) * nao * nao * nao;
    const int block = 128;
    const int grid = std::max(1, static_cast<int>((total + block - 1) / block));

    cudaEvent_t start;
    cudaEvent_t stop;
    check_cuda(cudaEventCreate(&start), "event start");
    check_cuda(cudaEventCreate(&stop), "event stop");
    check_cuda(cudaEventRecord(start), "record start");
    for (int repeat = 0; repeat < repeats; ++repeat) {
        check_cuda(cudaMemset(d_j, 0, matrix_bytes), "zero j");
        check_cuda(cudaMemset(d_k, 0, matrix_bytes), "zero k");
        sp_direct_jk_kernel<<<grid, block>>>(
            nao,
            max_nprim,
            d_nprims,
            d_angulars,
            d_centers,
            d_exponents,
            d_coefficients,
            d_density,
            d_j,
            d_k
        );
        check_cuda(cudaGetLastError(), "launch sp_direct_jk_kernel");
    }
    check_cuda(cudaEventRecord(stop), "record stop");
    check_cuda(cudaEventSynchronize(stop), "sync stop");
    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");

    std::vector<double> h_j(h_density.size());
    std::vector<double> h_k(h_density.size());
    check_cuda(cudaMemcpy(h_j.data(), d_j, matrix_bytes, cudaMemcpyDeviceToHost), "copy j");
    check_cuda(cudaMemcpy(h_k.data(), d_k, matrix_bytes, cudaMemcpyDeviceToHost), "copy k");

    std::ofstream output(output_path);
    output << std::setprecision(17);
    output << "kernel_avg_ms " << static_cast<double>(elapsed_ms) / repeats << "\n";
    output << "J";
    for (const double value : h_j) {
        output << " " << value;
    }
    output << "\nK";
    for (const double value : h_k) {
        output << " " << value;
    }
    output << "\n";

    cudaFree(d_nprims);
    cudaFree(d_angulars);
    cudaFree(d_centers);
    cudaFree(d_exponents);
    cudaFree(d_coefficients);
    cudaFree(d_density);
    cudaFree(d_j);
    cudaFree(d_k);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
