#include <cuda_runtime.h>
#include "xla/ffi/api/ffi.h"

#include <cmath>
#include <vector>

#ifdef TD_GRADDFT_ENABLE_GPU4PYSCF_RYS
#include "gvhf-rys/vhf.cuh"

extern "C" int RYS_build_jk(
    double *vj, double *vk, double *dm, int n_dm, int nao,
    RysIntEnvVars *envs, int *shls_slice, int shm_size,
    int npairs_ij, int npairs_kl,
    uint32_t *pair_ij_mapping, uint32_t *pair_kl_mapping,
    float *q_cond, float *s_estimator, float *dm_cond, float cutoff,
    int *pool, int *atm, int natm, int *bas, int nbas, double *env);

extern "C" int RYS_build_jk_init(int shm_size);
#endif

extern "C" cudaError_t TdGraddftLaunchJoltQC1qnt(
    cudaStream_t stream,
    int nao,
    const double* basis_data,
    double* dm,
    double* vj,
    double* vk,
    const int* group_keys,
    int n_groups,
    const int* group_quartet_keys,
    const int* group_quartet_offsets,
    const int* shell_quartets,
    int n_shell_quartets,
    int n_group_quartets
);

namespace {

namespace ffi = ::xla::ffi;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kBoysMax = 12;
constexpr int kPairMatrixReductionMinNao = 160;
constexpr int kJoltQCBasisStride = 12;

struct __align__(4 * sizeof(double)) JoltQCData4 {
    double x;
    double y;
    double z;
    double w;
};

struct __align__(2 * sizeof(double)) JoltQCData2 {
    double c;
    double e;
};

__device__ __forceinline__ const JoltQCData2* load_joltqc_ce_ptr(
    const double* __restrict__ basis_data,
    int shell_id
) {
    return reinterpret_cast<const JoltQCData2*>(
        basis_data + static_cast<long long>(shell_id) * kJoltQCBasisStride + 4);
}

__global__ void expand_joltqc_density_kernel(
    int nao,
    int joltqc_nao,
    const int* ao_to_parent_ao,
    const double* density,
    double* joltqc_density
) {
    const long long total = static_cast<long long>(joltqc_nao) * joltqc_nao;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int i = static_cast<int>(idx / joltqc_nao);
        const int j = static_cast<int>(idx - static_cast<long long>(i) * joltqc_nao);
        const int parent_i = ao_to_parent_ao[i];
        const int parent_j = ao_to_parent_ao[j];
        joltqc_density[idx] = density[parent_i * nao + parent_j];
    }
}

__global__ void contract_joltqc_potential_kernel(
    int nao,
    int joltqc_nao,
    const int* ao_to_parent_ao,
    const double* joltqc_j,
    const double* joltqc_k,
    double* j_out,
    double* k_out
) {
    const long long total = static_cast<long long>(joltqc_nao) * joltqc_nao;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int i = static_cast<int>(idx / joltqc_nao);
        const int j = static_cast<int>(idx - static_cast<long long>(i) * joltqc_nao);
        const int parent_i = ao_to_parent_ao[i];
        const int parent_j = ao_to_parent_ao[j];
        atomicAdd(j_out + parent_i * nao + parent_j, joltqc_j[idx]);
        atomicAdd(k_out + parent_i * nao + parent_j, joltqc_k[idx]);
    }
}

__global__ void finalize_joltqc_potential_kernel(int n, double* j_mat, double* k_mat) {
    const long long total = static_cast<long long>(n) * n;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int i = static_cast<int>(idx / n);
        const int j = static_cast<int>(idx - static_cast<long long>(i) * n);
        if (i > j) {
            continue;
        }
        const double jv = 2.0 * (j_mat[i * n + j] + j_mat[j * n + i]);
        const double kv = k_mat[i * n + j] + k_mat[j * n + i];
        j_mat[i * n + j] = jv;
        j_mat[j * n + i] = jv;
        k_mat[i * n + j] = kv;
        k_mat[j * n + i] = kv;
    }
}

struct Ang {
    int x;
    int y;
    int z;
};

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

__host__ __device__ bool is_single_p_angular(Ang a) {
    return lsum(a) == 1;
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
        double out = (ctx.center_p[axis] - ctx.center_a[axis]) * vrr(ctx, a1, c, m);
        out += (ctx.center_w[axis] - ctx.center_p[axis]) * vrr(ctx, a1, c, m + 1);
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
    double out = (ctx.center_q[axis] - ctx.center_c[axis]) * vrr(ctx, a, c1, m);
    out += (ctx.center_w[axis] - ctx.center_q[axis]) * vrr(ctx, a, c1, m + 1);
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
               (ctx.center_a[axis] - ctx.center_b[axis]) * hrr(ctx, a, b1, c, d);
    }
    if (lsum(d) > 0) {
        const int axis = first_axis(d);
        const Ang d1 = dec_axis(d, axis);
        const Ang c1 = inc_axis(c, axis);
        return hrr(ctx, a, b, c1, d1) +
               (ctx.center_c[axis] - ctx.center_d[axis]) * hrr(ctx, a, b, c, d1);
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
    const int max_boys_order =
        lsum(angular_a) + lsum(angular_b) + lsum(angular_c) + lsum(angular_d);
    double boys_raw[kBoysMax + 1] = {};
    boys_values(max_boys_order, (ctx.p * ctx.q / (ctx.p + ctx.q)) * rpq2, boys_raw);
    for (int n = 0; n <= max_boys_order; ++n) {
        ctx.boys[n] = prefactor * boys_raw[n];
    }
    return hrr(ctx, angular_a, angular_b, angular_c, angular_d);
}

__host__ __device__ double primitive_pair_eri(
    double bra_p,
    double ket_p,
    const double* center_p,
    const double* center_q,
    double bra_prefactor,
    double ket_prefactor,
    const double* center_a,
    const double* center_b,
    const double* center_c,
    const double* center_d,
    Ang angular_a,
    Ang angular_b,
    Ang angular_c,
    Ang angular_d,
    int max_boys_order
) {
    PrimitiveContext ctx;
    ctx.p = bra_p;
    ctx.q = ket_p;
    ctx.bra_ratio = ctx.q / (ctx.p + ctx.q);
    ctx.ket_ratio = ctx.p / (ctx.p + ctx.q);
    ctx.center_a = center_a;
    ctx.center_b = center_b;
    ctx.center_c = center_c;
    ctx.center_d = center_d;
    for (int axis = 0; axis < 3; ++axis) {
        ctx.center_p[axis] = center_p[axis];
        ctx.center_q[axis] = center_q[axis];
        ctx.center_w[axis] = (ctx.p * ctx.center_p[axis] + ctx.q * ctx.center_q[axis]) / (ctx.p + ctx.q);
    }
    const double rpq2 = dist2(ctx.center_p, ctx.center_q);
    const double prefactor = bra_prefactor * ket_prefactor *
                             2.0 * pow(kPi, 2.5) / (ctx.p * ctx.q * sqrt(ctx.p + ctx.q));
    const double t = (ctx.p * ctx.q / (ctx.p + ctx.q)) * rpq2;
    if (max_boys_order == 0) {
        return prefactor * boys0(t);
    }
    double boys_raw[kBoysMax + 1] = {};
    boys_values(max_boys_order, t, boys_raw);
    for (int n = 0; n <= max_boys_order; ++n) {
        ctx.boys[n] = prefactor * boys_raw[n];
    }
    return hrr(ctx, angular_a, angular_b, angular_c, angular_d);
}

__host__ __device__ double primitive_pair_eri_ssss(
    double bra_p,
    double ket_p,
    const double* center_p,
    const double* center_q,
    double bra_prefactor,
    double ket_prefactor
) {
    const double pq = bra_p + ket_p;
    const double rpq2 = dist2(center_p, center_q);
    const double prefactor = bra_prefactor * ket_prefactor *
                             2.0 * pow(kPi, 2.5) / (bra_p * ket_p * sqrt(pq));
    const double t = (bra_p * ket_p / pq) * rpq2;
    return prefactor * boys0(t);
}

__host__ __device__ double contracted_pair_eri_ssss(
    int max_pair_nprim,
    const int* pair_nprims,
    const double* pair_exponents,
    const double* pair_centers,
    const double* pair_prefactors,
    long long pair_p,
    long long pair_q
) {
    double value = 0.0;
    for (int bra_prim = 0; bra_prim < pair_nprims[pair_p]; ++bra_prim) {
        const long long bra_offset = pair_p * max_pair_nprim + bra_prim;
        const double bra_p = pair_exponents[bra_offset];
        const double bra_prefactor = pair_prefactors[bra_offset];
        const double* center_p = pair_centers + 3 * bra_offset;
        for (int ket_prim = 0; ket_prim < pair_nprims[pair_q]; ++ket_prim) {
            const long long ket_offset = pair_q * max_pair_nprim + ket_prim;
            value += primitive_pair_eri_ssss(
                bra_p,
                pair_exponents[ket_offset],
                center_p,
                pair_centers + 3 * ket_offset,
                bra_prefactor,
                pair_prefactors[ket_offset]);
        }
    }
    return value;
}

__host__ __device__ double primitive_pair_eri_single_p(
    double bra_p,
    double ket_p,
    const double* center_p,
    const double* center_q,
    double bra_prefactor,
    double ket_prefactor,
    const double* center_a,
    const double* center_b,
    const double* center_c,
    const double* center_d,
    Ang angular_a,
    Ang angular_b,
    Ang angular_c,
    Ang angular_d
) {
    const double pq = bra_p + ket_p;
    double center_w[3];
    for (int axis = 0; axis < 3; ++axis) {
        center_w[axis] = (bra_p * center_p[axis] + ket_p * center_q[axis]) / pq;
    }
    const double rpq2 = dist2(center_p, center_q);
    const double prefactor = bra_prefactor * ket_prefactor *
                             2.0 * pow(kPi, 2.5) / (bra_p * ket_p * sqrt(pq));
    const double t = (bra_p * ket_p / pq) * rpq2;
    double boys_raw[2] = {};
    boys_values(1, t, boys_raw);
    const double boys0_scaled = prefactor * boys_raw[0];
    const double boys1_scaled = prefactor * boys_raw[1];

    if (is_single_p_angular(angular_a)) {
        const int axis = first_axis(angular_a);
        return (center_p[axis] - center_a[axis]) * boys0_scaled +
               (center_w[axis] - center_p[axis]) * boys1_scaled;
    }
    if (is_single_p_angular(angular_b)) {
        const int axis = first_axis(angular_b);
        return (center_p[axis] - center_b[axis]) * boys0_scaled +
               (center_w[axis] - center_p[axis]) * boys1_scaled;
    }
    if (is_single_p_angular(angular_c)) {
        const int axis = first_axis(angular_c);
        return (center_q[axis] - center_c[axis]) * boys0_scaled +
               (center_w[axis] - center_q[axis]) * boys1_scaled;
    }
    const int axis = first_axis(angular_d);
    return (center_q[axis] - center_d[axis]) * boys0_scaled +
           (center_w[axis] - center_q[axis]) * boys1_scaled;
}

__host__ __device__ double contracted_pair_eri_single_p(
    int max_pair_nprim,
    const int* pair_nprims,
    const int* angulars,
    const double* centers,
    const double* pair_exponents,
    const double* pair_centers,
    const double* pair_prefactors,
    long long pair_p,
    long long pair_q,
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
    for (int bra_prim = 0; bra_prim < pair_nprims[pair_p]; ++bra_prim) {
        const long long bra_offset = pair_p * max_pair_nprim + bra_prim;
        const double bra_p = pair_exponents[bra_offset];
        const double bra_prefactor = pair_prefactors[bra_offset];
        const double* center_p = pair_centers + 3 * bra_offset;
        for (int ket_prim = 0; ket_prim < pair_nprims[pair_q]; ++ket_prim) {
            const long long ket_offset = pair_q * max_pair_nprim + ket_prim;
            value += primitive_pair_eri_single_p(
                bra_p,
                pair_exponents[ket_offset],
                center_p,
                pair_centers + 3 * ket_offset,
                bra_prefactor,
                pair_prefactors[ket_offset],
                center_i,
                center_j,
                center_k,
                center_l,
                angular_i,
                angular_j,
                angular_k,
                angular_l);
        }
    }
    return value;
}

__host__ __device__ double primitive_pair_eri_two_p(
    double bra_p,
    double ket_p,
    const double* center_p,
    const double* center_q,
    double bra_prefactor,
    double ket_prefactor,
    const double* center_a,
    const double* center_b,
    const double* center_c,
    const double* center_d,
    Ang angular_a,
    Ang angular_b,
    Ang angular_c,
    Ang angular_d
) {
    const double pq = bra_p + ket_p;
    const double bra_ratio = ket_p / pq;
    const double ket_ratio = bra_p / pq;
    double center_w[3];
    for (int axis = 0; axis < 3; ++axis) {
        center_w[axis] = (bra_p * center_p[axis] + ket_p * center_q[axis]) / pq;
    }
    const double rpq2 = dist2(center_p, center_q);
    const double prefactor = bra_prefactor * ket_prefactor *
                             2.0 * pow(kPi, 2.5) / (bra_p * ket_p * sqrt(pq));
    const double t = (bra_p * ket_p / pq) * rpq2;
    double boys_raw[3] = {};
    boys_values(2, t, boys_raw);
    const double f0 = prefactor * boys_raw[0];
    const double f1 = prefactor * boys_raw[1];
    const double f2 = prefactor * boys_raw[2];

    const bool a_p = is_single_p_angular(angular_a);
    const bool b_p = is_single_p_angular(angular_b);
    const bool c_p = is_single_p_angular(angular_c);
    const bool d_p = is_single_p_angular(angular_d);

    if (a_p && b_p) {
        const int axis_a = first_axis(angular_a);
        const int axis_b = first_axis(angular_b);
        const double left_m0 = (center_p[axis_a] - center_a[axis_a]) * f0 +
                               (center_w[axis_a] - center_p[axis_a]) * f1;
        const double left_m1 = (center_p[axis_a] - center_a[axis_a]) * f1 +
                               (center_w[axis_a] - center_p[axis_a]) * f2;
        const double delta = axis_a == axis_b ? 1.0 : 0.0;
        return (center_p[axis_b] - center_b[axis_b]) * left_m0 +
               (center_w[axis_b] - center_p[axis_b]) * left_m1 +
               delta * (f0 - bra_ratio * f1) / (2.0 * bra_p);
    }

    if (c_p && d_p) {
        const int axis_c = first_axis(angular_c);
        const int axis_d = first_axis(angular_d);
        const double left_m0 = (center_q[axis_c] - center_c[axis_c]) * f0 +
                               (center_w[axis_c] - center_q[axis_c]) * f1;
        const double left_m1 = (center_q[axis_c] - center_c[axis_c]) * f1 +
                               (center_w[axis_c] - center_q[axis_c]) * f2;
        const double delta = axis_c == axis_d ? 1.0 : 0.0;
        return (center_q[axis_d] - center_d[axis_d]) * left_m0 +
               (center_w[axis_d] - center_q[axis_d]) * left_m1 +
               delta * (f0 - ket_ratio * f1) / (2.0 * ket_p);
    }

    const Ang bra_ang = a_p ? angular_a : angular_b;
    const double* bra_center = a_p ? center_a : center_b;
    const Ang ket_ang = c_p ? angular_c : angular_d;
    const double* ket_center = c_p ? center_c : center_d;
    const int axis_bra = first_axis(bra_ang);
    const int axis_ket = first_axis(ket_ang);
    const double ket_m0 = (center_q[axis_ket] - ket_center[axis_ket]) * f0 +
                          (center_w[axis_ket] - center_q[axis_ket]) * f1;
    const double ket_m1 = (center_q[axis_ket] - ket_center[axis_ket]) * f1 +
                          (center_w[axis_ket] - center_q[axis_ket]) * f2;
    const double delta = axis_bra == axis_ket ? 1.0 : 0.0;
    return (center_p[axis_bra] - bra_center[axis_bra]) * ket_m0 +
           (center_w[axis_bra] - center_p[axis_bra]) * ket_m1 +
           delta * f1 / (2.0 * pq);
}

__host__ __device__ double contracted_pair_eri_two_p(
    int max_pair_nprim,
    const int* pair_nprims,
    const int* angulars,
    const double* centers,
    const double* pair_exponents,
    const double* pair_centers,
    const double* pair_prefactors,
    long long pair_p,
    long long pair_q,
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
    for (int bra_prim = 0; bra_prim < pair_nprims[pair_p]; ++bra_prim) {
        const long long bra_offset = pair_p * max_pair_nprim + bra_prim;
        const double bra_p = pair_exponents[bra_offset];
        const double bra_prefactor = pair_prefactors[bra_offset];
        const double* center_p = pair_centers + 3 * bra_offset;
        for (int ket_prim = 0; ket_prim < pair_nprims[pair_q]; ++ket_prim) {
            const long long ket_offset = pair_q * max_pair_nprim + ket_prim;
            value += primitive_pair_eri_two_p(
                bra_p,
                pair_exponents[ket_offset],
                center_p,
                pair_centers + 3 * ket_offset,
                bra_prefactor,
                pair_prefactors[ket_offset],
                center_i,
                center_j,
                center_k,
                center_l,
                angular_i,
                angular_j,
                angular_k,
                angular_l);
        }
    }
    return value;
}

__host__ __device__ double contracted_pair_eri(
    int max_pair_nprim,
    const int* pair_nprims,
    const int* angulars,
    const double* centers,
    const double* pair_exponents,
    const double* pair_centers,
    const double* pair_prefactors,
    long long pair_p,
    long long pair_q,
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
    const int max_boys_order =
        lsum(angular_i) + lsum(angular_j) + lsum(angular_k) + lsum(angular_l);
    if (max_boys_order == 0) {
        return contracted_pair_eri_ssss(
            max_pair_nprim,
            pair_nprims,
            pair_exponents,
            pair_centers,
            pair_prefactors,
            pair_p,
            pair_q);
    }
    if (max_boys_order == 1) {
        return contracted_pair_eri_single_p(
            max_pair_nprim,
            pair_nprims,
            angulars,
            centers,
            pair_exponents,
            pair_centers,
            pair_prefactors,
            pair_p,
            pair_q,
            i,
            j,
            k,
            l);
    }
    const int np_angular =
        (is_single_p_angular(angular_i) ? 1 : 0) +
        (is_single_p_angular(angular_j) ? 1 : 0) +
        (is_single_p_angular(angular_k) ? 1 : 0) +
        (is_single_p_angular(angular_l) ? 1 : 0);
    if (max_boys_order == 2 && np_angular == 2) {
        return contracted_pair_eri_two_p(
            max_pair_nprim,
            pair_nprims,
            angulars,
            centers,
            pair_exponents,
            pair_centers,
            pair_prefactors,
            pair_p,
            pair_q,
            i,
            j,
            k,
            l);
    }
    double value = 0.0;
    for (int bra_prim = 0; bra_prim < pair_nprims[pair_p]; ++bra_prim) {
        const long long bra_offset = pair_p * max_pair_nprim + bra_prim;
        const double bra_p = pair_exponents[bra_offset];
        const double bra_prefactor = pair_prefactors[bra_offset];
        const double* center_p = pair_centers + 3 * bra_offset;
        for (int ket_prim = 0; ket_prim < pair_nprims[pair_q]; ++ket_prim) {
            const long long ket_offset = pair_q * max_pair_nprim + ket_prim;
            value += primitive_pair_eri(
                bra_p,
                pair_exponents[ket_offset],
                center_p,
                pair_centers + 3 * ket_offset,
                bra_prefactor,
                pair_prefactors[ket_offset],
                center_i,
                center_j,
                center_k,
                center_l,
                angular_i,
                angular_j,
                angular_k,
                angular_l,
                max_boys_order);
        }
    }
    return value;
}

__host__ __device__ void decode_lower_pair(long long pair_id, int* row, int* col) {
    long long i = static_cast<long long>((sqrt(8.0 * static_cast<double>(pair_id) + 1.0) - 1.0) * 0.5);
    while (((i + 1) * (i + 2)) / 2 <= pair_id) {
        ++i;
    }
    while ((i * (i + 1)) / 2 > pair_id) {
        --i;
    }
    *row = static_cast<int>(i);
    *col = static_cast<int>(pair_id - (i * (i + 1)) / 2);
}

__host__ __device__ void decode_pair_quartet(long long quartet_id, long long* pair_p, long long* pair_q) {
    long long p =
        static_cast<long long>((sqrt(8.0 * static_cast<double>(quartet_id) + 1.0) - 1.0) * 0.5);
    while (((p + 1) * (p + 2)) / 2 <= quartet_id) {
        ++p;
    }
    while ((p * (p + 1)) / 2 > quartet_id) {
        --p;
    }
    *pair_p = p;
    *pair_q = quartet_id - (p * (p + 1)) / 2;
}

__host__ __device__ long long lower_pair_id(int i, int j) {
    if (i < j) {
        const int tmp = i;
        i = j;
        j = tmp;
    }
    return (static_cast<long long>(i) * (i + 1)) / 2 + j;
}

__device__ void add_symmetric_j_pair(double* j_mat, int nao, int i, int j, double value) {
    atomicAdd(j_mat + i * nao + j, value);
    if (i != j) {
        atomicAdd(j_mat + j * nao + i, value);
    }
}

__host__ __device__ int cart_ao_count_from_l(int l) {
    return ((l + 1) * (l + 2)) / 2;
}

__host__ __device__ Ang cart_angular_from_l_index(int l, int local_index) {
    int index = 0;
    for (int lx = l; lx >= 0; --lx) {
        const int rem = l - lx;
        for (int ly = rem; ly >= 0; --ly) {
            const int lz = rem - ly;
            if (index == local_index) {
                return Ang{lx, ly, lz};
            }
            ++index;
        }
    }
    return Ang{0, 0, 0};
}

__device__ double joltqc_contracted_eri(
    const double* basis_data,
    const int* shell_l,
    const int* shell_nprims,
    int ish,
    int jsh,
    int ksh,
    int lsh,
    int ia,
    int ja,
    int ka,
    int la
) {
    const double* center_i = basis_data + static_cast<long long>(ish) * kJoltQCBasisStride;
    const double* center_j = basis_data + static_cast<long long>(jsh) * kJoltQCBasisStride;
    const double* center_k = basis_data + static_cast<long long>(ksh) * kJoltQCBasisStride;
    const double* center_l = basis_data + static_cast<long long>(lsh) * kJoltQCBasisStride;
    const JoltQCData2* cei = load_joltqc_ce_ptr(basis_data, ish);
    const JoltQCData2* cej = load_joltqc_ce_ptr(basis_data, jsh);
    const JoltQCData2* cek = load_joltqc_ce_ptr(basis_data, ksh);
    const JoltQCData2* cel = load_joltqc_ce_ptr(basis_data, lsh);
    const Ang angular_i = cart_angular_from_l_index(shell_l[ish], ia);
    const Ang angular_j = cart_angular_from_l_index(shell_l[jsh], ja);
    const Ang angular_k = cart_angular_from_l_index(shell_l[ksh], ka);
    const Ang angular_l = cart_angular_from_l_index(shell_l[lsh], la);

    double value = 0.0;
    for (int ip = 0; ip < shell_nprims[ish]; ++ip) {
        for (int jp = 0; jp < shell_nprims[jsh]; ++jp) {
            for (int kp = 0; kp < shell_nprims[ksh]; ++kp) {
                for (int lp = 0; lp < shell_nprims[lsh]; ++lp) {
                    value +=
                        cei[ip].c * cej[jp].c * cek[kp].c * cel[lp].c *
                        primitive_sp_eri(
                            cei[ip].e,
                            cej[jp].e,
                            cek[kp].e,
                            cel[lp].e,
                            center_i,
                            center_j,
                            center_k,
                            center_l,
                            angular_i,
                            angular_j,
                            angular_k,
                            angular_l);
                }
            }
        }
    }
    return value;
}

__device__ double abs_max_density_for_quartet(
    const double* density,
    int nao,
    int i,
    int j,
    int k,
    int l
) {
    double out = fabs(density[k * nao + l]);
    out = fmax(out, fabs(density[i * nao + j]));
    out = fmax(out, fabs(density[j * nao + l]));
    out = fmax(out, fabs(density[i * nao + l]));
    out = fmax(out, fabs(density[j * nao + k]));
    out = fmax(out, fabs(density[i * nao + k]));
    return out;
}

__global__ void pair_schwarz_kernel(
    int nao,
    int max_pair_nprim,
    const int* pair_nprims,
    const int* angulars,
    const double* centers,
    const double* pair_exponents,
    const double* pair_centers,
    const double* pair_prefactors,
    const int* pair_rows,
    const int* pair_cols,
    double* pair_schwarz
) {
    const long long npair = static_cast<long long>(nao) * (nao + 1) / 2;
    for (long long pair_p = blockIdx.x * blockDim.x + threadIdx.x;
         pair_p < npair;
         pair_p += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int i = pair_rows[pair_p];
        const int j = pair_cols[pair_p];
        const double self_eri = contracted_pair_eri(
            max_pair_nprim,
            pair_nprims,
            angulars,
            centers,
            pair_exponents,
            pair_centers,
            pair_prefactors,
            pair_p,
            pair_p,
            i,
            j,
            i,
            j);
        pair_schwarz[pair_p] = sqrt(fabs(self_eri));
    }
}

__global__ void unique_pair_quartet_direct_jk_kernel(
    int nao,
    int max_pair_nprim,
    const int* pair_nprims,
    const int* angulars,
    const double* centers,
    const double* pair_exponents,
    const double* pair_centers,
    const double* pair_prefactors,
    const double* density,
    const int* pair_rows,
    const int* pair_cols,
    const double* pair_schwarz,
    const double* density_cutoff,
    double* j_mat,
    double* k_mat
) {
    const long long npair = static_cast<long long>(nao) * (nao + 1) / 2;
    const long long total = npair * (npair + 1) / 2;
    const double cutoff_value = density_cutoff[0] < 0.0 ? exp(density_cutoff[0]) : density_cutoff[0];
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long pair_p;
        long long pair_q;
        decode_pair_quartet(idx, &pair_p, &pair_q);

        const int i = pair_rows[pair_p];
        const int j = pair_cols[pair_p];
        const int k = pair_rows[pair_q];
        const int l = pair_cols[pair_q];

        if (
            cutoff_value > 0.0 &&
            pair_schwarz[pair_p] * pair_schwarz[pair_q] *
                abs_max_density_for_quartet(density, nao, i, j, k, l) < cutoff_value
        ) {
            continue;
        }

        const double eri = contracted_pair_eri(
            max_pair_nprim,
            pair_nprims,
            angulars,
            centers,
            pair_exponents,
            pair_centers,
            pair_prefactors,
            pair_p,
            pair_q,
            i,
            j,
            k,
            l);
        const double pair_p_mult = i == j ? 1.0 : 2.0;
        const double pair_q_mult = k == l ? 1.0 : 2.0;

        add_symmetric_j_pair(j_mat, nao, i, j, pair_q_mult * eri * density[k * nao + l]);
        if (pair_p != pair_q) {
            add_symmetric_j_pair(j_mat, nao, k, l, pair_p_mult * eri * density[i * nao + j]);
        }

        atomicAdd(k_mat + i * nao + k, eri * density[j * nao + l]);
        if (i != j) {
            atomicAdd(k_mat + j * nao + k, eri * density[i * nao + l]);
        }
        if (k != l) {
            atomicAdd(k_mat + i * nao + l, eri * density[j * nao + k]);
        }
        if (i != j && k != l) {
            atomicAdd(k_mat + j * nao + l, eri * density[i * nao + k]);
        }
        if (pair_p != pair_q) {
            atomicAdd(k_mat + k * nao + i, eri * density[l * nao + j]);
            if (k != l) {
                atomicAdd(k_mat + l * nao + i, eri * density[k * nao + j]);
            }
            if (i != j) {
                atomicAdd(k_mat + k * nao + j, eri * density[l * nao + i]);
            }
            if (i != j && k != l) {
                atomicAdd(k_mat + l * nao + j, eri * density[k * nao + i]);
            }
        }
    }
}

struct ShellQuartetTask {
    int i;
    int j;
    int k;
    int l;
};

__device__ double shell_abs_max_density(
    const double* shell_dm_cond,
    int nshell,
    int i,
    int j,
    int k,
    int l
) {
    double out = shell_dm_cond[k * nshell + l];
    out = fmax(out, shell_dm_cond[i * nshell + j]);
    out = fmax(out, shell_dm_cond[j * nshell + l]);
    out = fmax(out, shell_dm_cond[i * nshell + l]);
    out = fmax(out, shell_dm_cond[j * nshell + k]);
    out = fmax(out, shell_dm_cond[i * nshell + k]);
    return out;
}

__global__ void screen_shell_quartet_tasks(
    int nshell,
    int ntiles,
    int tile_size,
    const int* tile_pair_ids,
    long long n_tile_pairs,
    const int* tile_shell_indices,
    const int* tile_shell_pad_mask,
    const double* shell_log_q_matrix,
    const double* shell_dm_cond,
    const double* log_cutoff,
    long long max_tasks,
    ShellQuartetTask* shell_quartet_tasks,
    unsigned int* task_count
) {
    const long long total = n_tile_pairs * n_tile_pairs;
    constexpr double kTiny = 1.0e-300;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        const long long tile_pair_p_idx = idx / n_tile_pairs;
        const long long tile_pair_q_idx = idx - tile_pair_p_idx * n_tile_pairs;

        const int tile_pair_p = tile_pair_ids[tile_pair_p_idx];
        const int tile_pair_q = tile_pair_ids[tile_pair_q_idx];
        const int tile_i = tile_pair_p / ntiles;
        const int tile_j = tile_pair_p % ntiles;
        const int tile_k = tile_pair_q / ntiles;
        const int tile_l = tile_pair_q % ntiles;
        const int tile_i_offset = tile_i * tile_size;
        const int tile_j_offset = tile_j * tile_size;
        const int tile_k_offset = tile_k * tile_size;
        const int tile_l_offset = tile_l * tile_size;

        for (int ii = 0; ii < tile_size; ++ii) {
            if (tile_shell_pad_mask[tile_i_offset + ii]) {
                continue;
            }
            const int ish = tile_shell_indices[tile_i_offset + ii];
            for (int jj = 0; jj < tile_size; ++jj) {
                if (tile_shell_pad_mask[tile_j_offset + jj]) {
                    continue;
                }
                const int jsh = tile_shell_indices[tile_j_offset + jj];
                if (tile_i == tile_j && ii < jj) {
                    continue;
                }
                int p_i = ish;
                int p_j = jsh;
                if (p_i < p_j) {
                    const int tmp = p_i;
                    p_i = p_j;
                    p_j = tmp;
                }
                const long long shell_pair_p = lower_pair_id(p_i, p_j);
                const double q_ij = shell_log_q_matrix[p_i * nshell + p_j];
                for (int kk = 0; kk < tile_size; ++kk) {
                    if (tile_shell_pad_mask[tile_k_offset + kk]) {
                        continue;
                    }
                    const int ksh = tile_shell_indices[tile_k_offset + kk];
                    for (int ll = 0; ll < tile_size; ++ll) {
                        if (tile_shell_pad_mask[tile_l_offset + ll]) {
                            continue;
                        }
                        const int lsh = tile_shell_indices[tile_l_offset + ll];
                        if (tile_k == tile_l && kk < ll) {
                            continue;
                        }
                        int q_i = ksh;
                        int q_j = lsh;
                        if (q_i < q_j) {
                            const int tmp = q_i;
                            q_i = q_j;
                            q_j = tmp;
                        }
                        const long long shell_pair_q = lower_pair_id(q_i, q_j);
                        if (shell_pair_p < shell_pair_q) {
                            continue;
                        }
                        const double q_kl = shell_log_q_matrix[q_i * nshell + q_j];
                        const double d_large = log(fmax(
                            shell_abs_max_density(shell_dm_cond, nshell, p_i, p_j, q_i, q_j),
                            kTiny));
                        if (q_ij + q_kl + d_large <= log_cutoff[0]) {
                            continue;
                        }
                        const unsigned int task_id = atomicAdd(task_count, 1u);
                        if (static_cast<long long>(task_id) < max_tasks) {
                            shell_quartet_tasks[task_id] = ShellQuartetTask{p_i, p_j, q_i, q_j};
                        }
                    }
                }
            }
        }
    }
}

__global__ void joltqc_shell_quartet_direct_jk_kernel(
    int joltqc_nao,
    const double* basis_data,
    const int* shell_l,
    const int* shell_nprims,
    const double* density,
    const int* shell_quartets,
    long long nquartets,
    double* j_mat,
    double* k_mat
) {
    for (long long task_id = blockIdx.x;
         task_id < nquartets;
         task_id += gridDim.x) {
        const int ish = shell_quartets[4 * task_id + 0];
        const int jsh = shell_quartets[4 * task_id + 1];
        const int ksh = shell_quartets[4 * task_id + 2];
        const int lsh = shell_quartets[4 * task_id + 3];
        const long long shell_pair_p = lower_pair_id(ish, jsh);
        const long long shell_pair_q = lower_pair_id(ksh, lsh);
        const int i0 = static_cast<int>(basis_data[static_cast<long long>(ish) * kJoltQCBasisStride + 3]);
        const int j0 = static_cast<int>(basis_data[static_cast<long long>(jsh) * kJoltQCBasisStride + 3]);
        const int k0 = static_cast<int>(basis_data[static_cast<long long>(ksh) * kJoltQCBasisStride + 3]);
        const int l0 = static_cast<int>(basis_data[static_cast<long long>(lsh) * kJoltQCBasisStride + 3]);
        const int ni = cart_ao_count_from_l(shell_l[ish]);
        const int nj = cart_ao_count_from_l(shell_l[jsh]);
        const int nk = cart_ao_count_from_l(shell_l[ksh]);
        const int nl = cart_ao_count_from_l(shell_l[lsh]);
        long long local_quartet = 0;

        for (int ia = 0; ia < ni; ++ia) {
            for (int ja = 0; ja < nj; ++ja) {
                if (ish == jsh && ja > ia) {
                    continue;
                }
                int i_base = i0 + ia;
                int j_base = j0 + ja;
                int ia_base = ia;
                int ja_base = ja;
                if (i_base < j_base) {
                    const int tmp_ao = i_base;
                    i_base = j_base;
                    j_base = tmp_ao;
                    const int tmp_local = ia_base;
                    ia_base = ja_base;
                    ja_base = tmp_local;
                }
                const long long pair_p_base = lower_pair_id(i_base, j_base);
                for (int ka = 0; ka < nk; ++ka) {
                    for (int la = 0; la < nl; ++la) {
                        if (ksh == lsh && la > ka) {
                            continue;
                        }
                        int k_base = k0 + ka;
                        int l_base = l0 + la;
                        int ka_base = ka;
                        int la_base = la;
                        if (k_base < l_base) {
                            const int tmp_ao = k_base;
                            k_base = l_base;
                            l_base = tmp_ao;
                            const int tmp_local = ka_base;
                            ka_base = la_base;
                            la_base = tmp_local;
                        }
                        long long pair_p = pair_p_base;
                        long long pair_q = lower_pair_id(k_base, l_base);
                        int i = i_base;
                        int j = j_base;
                        int k = k_base;
                        int l = l_base;
                        int i_local = ia_base;
                        int j_local = ja_base;
                        int k_local = ka_base;
                        int l_local = la_base;
                        int shell_i = ish;
                        int shell_j = jsh;
                        int shell_k = ksh;
                        int shell_l_id = lsh;
                        if (pair_p < pair_q) {
                            if (shell_pair_p == shell_pair_q) {
                                continue;
                            }
                            const long long tmp_pair = pair_p;
                            pair_p = pair_q;
                            pair_q = tmp_pair;
                            const int tmp_i = i;
                            const int tmp_j = j;
                            i = k;
                            j = l;
                            k = tmp_i;
                            l = tmp_j;
                            const int tmp_i_local = i_local;
                            const int tmp_j_local = j_local;
                            i_local = k_local;
                            j_local = l_local;
                            k_local = tmp_i_local;
                            l_local = tmp_j_local;
                            const int tmp_shell_i = shell_i;
                            const int tmp_shell_j = shell_j;
                            shell_i = shell_k;
                            shell_j = shell_l_id;
                            shell_k = tmp_shell_i;
                            shell_l_id = tmp_shell_j;
                        }
                        const long long quartet_id = local_quartet++;
                        if ((quartet_id % blockDim.x) != threadIdx.x) {
                            continue;
                        }

                        const double eri = joltqc_contracted_eri(
                            basis_data,
                            shell_l,
                            shell_nprims,
                            shell_i,
                            shell_j,
                            shell_k,
                            shell_l_id,
                            i_local,
                            j_local,
                            k_local,
                            l_local);
                        const double pair_p_mult = i == j ? 1.0 : 2.0;
                        const double pair_q_mult = k == l ? 1.0 : 2.0;

                        add_symmetric_j_pair(j_mat, joltqc_nao, i, j, pair_q_mult * eri * density[k * joltqc_nao + l]);
                        if (pair_p != pair_q) {
                            add_symmetric_j_pair(j_mat, joltqc_nao, k, l, pair_p_mult * eri * density[i * joltqc_nao + j]);
                        }

                        atomicAdd(k_mat + i * joltqc_nao + k, eri * density[j * joltqc_nao + l]);
                        if (i != j) {
                            atomicAdd(k_mat + j * joltqc_nao + k, eri * density[i * joltqc_nao + l]);
                        }
                        if (k != l) {
                            atomicAdd(k_mat + i * joltqc_nao + l, eri * density[j * joltqc_nao + k]);
                        }
                        if (i != j && k != l) {
                            atomicAdd(k_mat + j * joltqc_nao + l, eri * density[i * joltqc_nao + k]);
                        }
                        if (pair_p != pair_q) {
                            atomicAdd(k_mat + k * joltqc_nao + i, eri * density[l * joltqc_nao + j]);
                            if (k != l) {
                                atomicAdd(k_mat + l * joltqc_nao + i, eri * density[k * joltqc_nao + j]);
                            }
                            if (i != j) {
                                atomicAdd(k_mat + k * joltqc_nao + j, eri * density[l * joltqc_nao + i]);
                            }
                            if (i != j && k != l) {
                                atomicAdd(k_mat + l * joltqc_nao + j, eri * density[k * joltqc_nao + i]);
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void screened_shell_quartet_direct_jk_kernel(
    int nao,
    int max_pair_nprim,
    int max_shell_ao,
    const int* pair_nprims,
    const int* angulars,
    const double* centers,
    const double* pair_exponents,
    const double* pair_centers,
    const double* pair_prefactors,
    const double* density,
    const int* shell_ao_indices_padded,
    const int* shell_ao_sizes,
    const ShellQuartetTask* shell_quartet_tasks,
    const unsigned int* task_count,
    long long max_tasks,
    double* j_mat,
    double* k_mat
) {
    long long ntasks = static_cast<long long>(task_count[0]);
    if (ntasks > max_tasks) {
        ntasks = max_tasks;
    }
    for (long long task_id = blockIdx.x;
         task_id < ntasks;
         task_id += gridDim.x) {
        const ShellQuartetTask task = shell_quartet_tasks[task_id];
        const long long shell_pair_p = lower_pair_id(task.i, task.j);
        const long long shell_pair_q = lower_pair_id(task.k, task.l);
        const int* ao_i = shell_ao_indices_padded + task.i * max_shell_ao;
        const int* ao_j = shell_ao_indices_padded + task.j * max_shell_ao;
        const int* ao_k = shell_ao_indices_padded + task.k * max_shell_ao;
        const int* ao_l = shell_ao_indices_padded + task.l * max_shell_ao;
        const int ni = shell_ao_sizes[task.i];
        const int nj = shell_ao_sizes[task.j];
        const int nk = shell_ao_sizes[task.k];
        const int nl = shell_ao_sizes[task.l];
        long long local_quartet = 0;

        for (int ia = 0; ia < ni; ++ia) {
            const int i_raw = ao_i[ia];
            for (int ja = 0; ja < nj; ++ja) {
                if (task.i == task.j && ja > ia) {
                    continue;
                }
                int i_base = i_raw;
                int j_base = ao_j[ja];
                if (i_base < j_base) {
                    const int tmp = i_base;
                    i_base = j_base;
                    j_base = tmp;
                }
                const long long pair_p_base = lower_pair_id(i_base, j_base);
                for (int ka = 0; ka < nk; ++ka) {
                    const int k_raw = ao_k[ka];
                    for (int la = 0; la < nl; ++la) {
                        if (task.k == task.l && la > ka) {
                            continue;
                        }
                        int k_base = k_raw;
                        int l_base = ao_l[la];
                        if (k_base < l_base) {
                            const int tmp = k_base;
                            k_base = l_base;
                            l_base = tmp;
                        }
                        long long pair_p = pair_p_base;
                        long long pair_q = lower_pair_id(k_base, l_base);
                        int i = i_base;
                        int j = j_base;
                        int k = k_base;
                        int l = l_base;
                        if (pair_p < pair_q) {
                            if (shell_pair_p == shell_pair_q) {
                                continue;
                            }
                            const long long tmp_pair = pair_p;
                            pair_p = pair_q;
                            pair_q = tmp_pair;
                            const int tmp_i = i;
                            const int tmp_j = j;
                            i = k;
                            j = l;
                            k = tmp_i;
                            l = tmp_j;
                        }
                        const long long quartet_id = local_quartet++;
                        if ((quartet_id % blockDim.x) != threadIdx.x) {
                            continue;
                        }

                        const double eri = contracted_pair_eri(
                            max_pair_nprim,
                            pair_nprims,
                            angulars,
                            centers,
                            pair_exponents,
                            pair_centers,
                            pair_prefactors,
                            pair_p,
                            pair_q,
                            i,
                            j,
                            k,
                            l);
                        const double pair_p_mult = i == j ? 1.0 : 2.0;
                        const double pair_q_mult = k == l ? 1.0 : 2.0;

                        add_symmetric_j_pair(j_mat, nao, i, j, pair_q_mult * eri * density[k * nao + l]);
                        if (pair_p != pair_q) {
                            add_symmetric_j_pair(j_mat, nao, k, l, pair_p_mult * eri * density[i * nao + j]);
                        }

                        atomicAdd(k_mat + i * nao + k, eri * density[j * nao + l]);
                        if (i != j) {
                            atomicAdd(k_mat + j * nao + k, eri * density[i * nao + l]);
                        }
                        if (k != l) {
                            atomicAdd(k_mat + i * nao + l, eri * density[j * nao + k]);
                        }
                        if (i != j && k != l) {
                            atomicAdd(k_mat + j * nao + l, eri * density[i * nao + k]);
                        }
                        if (pair_p != pair_q) {
                            atomicAdd(k_mat + k * nao + i, eri * density[l * nao + j]);
                            if (k != l) {
                                atomicAdd(k_mat + l * nao + i, eri * density[k * nao + j]);
                            }
                            if (i != j) {
                                atomicAdd(k_mat + k * nao + j, eri * density[l * nao + i]);
                            }
                            if (i != j && k != l) {
                                atomicAdd(k_mat + l * nao + j, eri * density[k * nao + i]);
                            }
                        }
                    }
                }
            }
        }
    }
}

__global__ void symmetrize_kernel(int n, double* j_mat, double* k_mat) {
    const long long total = static_cast<long long>(n) * n;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int i = static_cast<int>(idx / n);
        const int j = static_cast<int>(idx % n);
        if (i < j) {
            const double jv = 0.5 * (j_mat[i * n + j] + j_mat[j * n + i]);
            const double kv = 0.5 * (k_mat[i * n + j] + k_mat[j * n + i]);
            j_mat[i * n + j] = jv;
            j_mat[j * n + i] = jv;
            k_mat[i * n + j] = kv;
            k_mat[j * n + i] = kv;
        }
    }
}

__device__ void reduce_block_sum(double* partials) {
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        }
        __syncthreads();
    }
}

__global__ void pair_matrix_j_kernel(
    int nao,
    const double* eri_pair_matrix,
    const double* density,
    const int* pair_rows,
    const int* pair_cols,
    double* j_mat
) {
    const long long npair = static_cast<long long>(nao) * (nao + 1) / 2;
    for (long long pair_p = blockIdx.x * blockDim.x + threadIdx.x;
         pair_p < npair;
         pair_p += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int i = pair_rows[pair_p];
        const int j = pair_cols[pair_p];
        double value = 0.0;
        for (int k = 0; k < nao; ++k) {
            const long long row_offset = pair_p * npair + (static_cast<long long>(k) * (k + 1)) / 2;
            for (int l = 0; l <= k; ++l) {
                const double multiplicity = k == l ? 1.0 : 2.0;
                value += eri_pair_matrix[row_offset + l] * multiplicity * density[k * nao + l];
            }
        }
        j_mat[i * nao + j] = value;
        if (i != j) {
            j_mat[j * nao + i] = value;
        }
    }
}

__global__ void pair_matrix_k_kernel(
    int nao,
    const double* eri_pair_matrix,
    const double* density,
    const int* pair_rows,
    const int* pair_cols,
    double* k_mat
) {
    const long long npair = static_cast<long long>(nao) * (nao + 1) / 2;
    for (long long pair_p = blockIdx.x * blockDim.x + threadIdx.x;
         pair_p < npair;
         pair_p += static_cast<long long>(blockDim.x) * gridDim.x) {
        const int p = pair_rows[pair_p];
        const int q = pair_cols[pair_p];
        double value = 0.0;
        for (int r = 0; r < nao; ++r) {
            const long long pair_pr = lower_pair_id(p, r);
            for (int s = 0; s < nao; ++s) {
                const long long pair_qs = lower_pair_id(q, s);
                value += eri_pair_matrix[pair_pr * npair + pair_qs] * density[r * nao + s];
            }
        }
        k_mat[p * nao + q] = value;
        if (p != q) {
            k_mat[q * nao + p] = value;
        }
    }
}

__global__ void pair_matrix_j_reduce_kernel(
    int nao,
    const double* eri_pair_matrix,
    const double* density,
    const int* pair_rows,
    const int* pair_cols,
    double* j_mat
) {
    extern __shared__ double partials[];
    const long long npair = static_cast<long long>(nao) * (nao + 1) / 2;
    const long long pair_p = static_cast<long long>(blockIdx.x);
    if (pair_p >= npair) {
        return;
    }
    double value = 0.0;
    const long long row_offset = pair_p * npair;
    for (long long pair_q = static_cast<long long>(threadIdx.x);
         pair_q < npair;
         pair_q += static_cast<long long>(blockDim.x)) {
        const int k = pair_rows[pair_q];
        const int l = pair_cols[pair_q];
        const double multiplicity = k == l ? 1.0 : 2.0;
        value += eri_pair_matrix[row_offset + pair_q] * multiplicity * density[k * nao + l];
    }
    partials[threadIdx.x] = value;
    reduce_block_sum(partials);
    if (threadIdx.x == 0) {
        const int i = pair_rows[pair_p];
        const int j = pair_cols[pair_p];
        j_mat[i * nao + j] = partials[0];
        if (i != j) {
            j_mat[j * nao + i] = partials[0];
        }
    }
}

__global__ void pair_matrix_k_reduce_kernel(
    int nao,
    const double* eri_pair_matrix,
    const double* density,
    const int* pair_rows,
    const int* pair_cols,
    double* k_mat
) {
    extern __shared__ double partials[];
    const long long npair = static_cast<long long>(nao) * (nao + 1) / 2;
    const long long pair_p = static_cast<long long>(blockIdx.x);
    if (pair_p >= npair) {
        return;
    }
    const int p = pair_rows[pair_p];
    const int q = pair_cols[pair_p];
    const long long total_density = static_cast<long long>(nao) * static_cast<long long>(nao);
    double value = 0.0;
    for (long long flat = static_cast<long long>(threadIdx.x);
         flat < total_density;
         flat += static_cast<long long>(blockDim.x)) {
        const int r = static_cast<int>(flat / nao);
        const int s = static_cast<int>(flat - static_cast<long long>(r) * nao);
        const long long pair_pr = lower_pair_id(p, r);
        const long long pair_qs = lower_pair_id(q, s);
        value += eri_pair_matrix[pair_pr * npair + pair_qs] * density[r * nao + s];
    }
    partials[threadIdx.x] = value;
    reduce_block_sum(partials);
    if (threadIdx.x == 0) {
        k_mat[p * nao + q] = partials[0];
        if (p != q) {
            k_mat[q * nao + p] = partials[0];
        }
    }
}

__device__ void set_eri_value(double* eri, int nao, int i, int j, int k, int l, double value) {
    const long long idx =
        (((static_cast<long long>(i) * nao + j) * nao + k) * nao + l);
    eri[idx] = value;
}

__global__ void unique_pair_quartet_eri_tensor_kernel(
    int nao,
    int max_pair_nprim,
    const int* pair_nprims,
    const int* angulars,
    const double* centers,
    const double* pair_exponents,
    const double* pair_centers,
    const double* pair_prefactors,
    const int* pair_rows,
    const int* pair_cols,
    double* eri_tensor
) {
    const long long npair = static_cast<long long>(nao) * (nao + 1) / 2;
    const long long total = npair * (npair + 1) / 2;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long pair_p;
        long long pair_q;
        decode_pair_quartet(idx, &pair_p, &pair_q);

        const int i = pair_rows[pair_p];
        const int j = pair_cols[pair_p];
        const int k = pair_rows[pair_q];
        const int l = pair_cols[pair_q];

        const double eri = contracted_pair_eri(
            max_pair_nprim,
            pair_nprims,
            angulars,
            centers,
            pair_exponents,
            pair_centers,
            pair_prefactors,
            pair_p,
            pair_q,
            i,
            j,
            k,
            l);

        set_eri_value(eri_tensor, nao, i, j, k, l, eri);
        set_eri_value(eri_tensor, nao, j, i, k, l, eri);
        set_eri_value(eri_tensor, nao, i, j, l, k, eri);
        set_eri_value(eri_tensor, nao, j, i, l, k, eri);
        set_eri_value(eri_tensor, nao, k, l, i, j, eri);
        set_eri_value(eri_tensor, nao, l, k, i, j, eri);
        set_eri_value(eri_tensor, nao, k, l, j, i, eri);
        set_eri_value(eri_tensor, nao, l, k, j, i, eri);
    }
}

__global__ void unique_pair_quartet_eri_pair_matrix_kernel(
    int nao,
    int max_pair_nprim,
    const int* pair_nprims,
    const int* angulars,
    const double* centers,
    const double* pair_exponents,
    const double* pair_centers,
    const double* pair_prefactors,
    const int* pair_rows,
    const int* pair_cols,
    const double* pair_schwarz,
    const double* eri_cutoff,
    double* eri_pair_matrix
) {
    const long long npair = static_cast<long long>(nao) * (nao + 1) / 2;
    const long long total = npair * (npair + 1) / 2;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x) {
        long long pair_p;
        long long pair_q;
        decode_pair_quartet(idx, &pair_p, &pair_q);

        const int i = pair_rows[pair_p];
        const int j = pair_cols[pair_p];
        const int k = pair_rows[pair_q];
        const int l = pair_cols[pair_q];

        if (
            eri_cutoff[0] > 0.0 &&
            pair_schwarz[pair_p] * pair_schwarz[pair_q] < eri_cutoff[0]
        ) {
            continue;
        }

        const double eri = contracted_pair_eri(
            max_pair_nprim,
            pair_nprims,
            angulars,
            centers,
            pair_exponents,
            pair_centers,
            pair_prefactors,
            pair_p,
            pair_q,
            i,
            j,
            k,
            l);

        eri_pair_matrix[pair_p * npair + pair_q] = eri;
        eri_pair_matrix[pair_q * npair + pair_p] = eri;
    }
}

ffi::Error CudaJoltQCDirectJkDispatch(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64, 2> density,
    ffi::Buffer<ffi::F64, 2> basis_data,
    ffi::Buffer<ffi::S32, 1> shell_l,
    ffi::Buffer<ffi::S32, 1> shell_nprims,
    ffi::Buffer<ffi::S32, 1> ao_to_parent_ao,
    ffi::Buffer<ffi::S32, 2> group_keys,
    ffi::Buffer<ffi::S32, 2> group_quartet_keys,
    ffi::Buffer<ffi::S32, 1> group_quartet_offsets,
    ffi::Buffer<ffi::S32, 2> shell_quartets,
    ffi::Result<ffi::Buffer<ffi::F64, 2>> j_out,
    ffi::Result<ffi::Buffer<ffi::F64, 2>> k_out) {
    const int64_t nao64 = density.dimensions()[0];
    if (density.dimensions()[1] != nao64) {
        return ffi::Error::InvalidArgument("density must be square");
    }
    if (basis_data.dimensions()[1] != kJoltQCBasisStride) {
        return ffi::Error::InvalidArgument("basis_data must have shape (nbas, 12)");
    }
    const int64_t nbas64 = basis_data.dimensions()[0];
    if (shell_l.dimensions()[0] != nbas64 || shell_nprims.dimensions()[0] != nbas64) {
        return ffi::Error::InvalidArgument("shell metadata must have shape (nbas,)");
    }
    if (group_keys.dimensions()[1] != 2) {
        return ffi::Error::InvalidArgument("group_keys must have shape (ngroups, 2)");
    }
    if (group_quartet_keys.dimensions()[1] != 4) {
        return ffi::Error::InvalidArgument("group_quartet_keys must have shape (ngroups, 4)");
    }
    if (group_quartet_offsets.dimensions()[0] != group_quartet_keys.dimensions()[0] + 1) {
        return ffi::Error::InvalidArgument("group_quartet_offsets must have shape (ngroups + 1,)");
    }
    if (shell_quartets.dimensions()[1] != 4) {
        return ffi::Error::InvalidArgument("shell_quartets must have shape (nquartets, 4)");
    }
    if (j_out->dimensions()[0] != nao64 || j_out->dimensions()[1] != nao64 ||
        k_out->dimensions()[0] != nao64 || k_out->dimensions()[1] != nao64) {
        return ffi::Error::InvalidArgument("J/K outputs must match density shape");
    }

    const int nao = static_cast<int>(nao64);
    const int joltqc_nao = static_cast<int>(ao_to_parent_ao.dimensions()[0]);
    const long long nquartets = static_cast<long long>(shell_quartets.dimensions()[0]);
    const size_t out_matrix_bytes = static_cast<size_t>(nao) * static_cast<size_t>(nao) * sizeof(double);
    const size_t internal_matrix_bytes =
        static_cast<size_t>(joltqc_nao) * static_cast<size_t>(joltqc_nao) * sizeof(double);

    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    cudaMemsetAsync(j_out->typed_data(), 0, out_matrix_bytes, stream);
    cudaMemsetAsync(k_out->typed_data(), 0, out_matrix_bytes, stream);
    if (joltqc_nao <= 0 || nquartets <= 0) {
        return ffi::Error::Success();
    }

    double* joltqc_density = nullptr;
    double* joltqc_j = nullptr;
    double* joltqc_k = nullptr;
    if (cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(&joltqc_density), internal_matrix_bytes, stream); err != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    if (cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(&joltqc_j), internal_matrix_bytes, stream); err != cudaSuccess) {
        cudaFreeAsync(joltqc_density, stream);
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    if (cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(&joltqc_k), internal_matrix_bytes, stream); err != cudaSuccess) {
        cudaFreeAsync(joltqc_j, stream);
        cudaFreeAsync(joltqc_density, stream);
        return ffi::Error::Internal(cudaGetErrorString(err));
    }

    const int block = 128;
    const int internal_grid =
        static_cast<int>((static_cast<long long>(joltqc_nao) * joltqc_nao + block - 1) / block);
    expand_joltqc_density_kernel<<<internal_grid, block, 0, stream>>>(
        nao,
        joltqc_nao,
        ao_to_parent_ao.typed_data(),
        density.typed_data(),
        joltqc_density);
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        cudaFreeAsync(joltqc_k, stream);
        cudaFreeAsync(joltqc_j, stream);
        cudaFreeAsync(joltqc_density, stream);
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    cudaMemsetAsync(joltqc_j, 0, internal_matrix_bytes, stream);
    cudaMemsetAsync(joltqc_k, 0, internal_matrix_bytes, stream);

    cudaError_t fast_err = TdGraddftLaunchJoltQC1qnt(
        stream,
        joltqc_nao,
        basis_data.typed_data(),
        joltqc_density,
        joltqc_j,
        joltqc_k,
        group_keys.typed_data(),
        static_cast<int>(group_keys.dimensions()[0]),
        group_quartet_keys.typed_data(),
        group_quartet_offsets.typed_data(),
        shell_quartets.typed_data(),
        static_cast<int>(shell_quartets.dimensions()[0]),
        static_cast<int>(group_quartet_keys.dimensions()[0]));
    if (fast_err != cudaSuccess && fast_err != cudaErrorNotSupported) {
        cudaFreeAsync(joltqc_k, stream);
        cudaFreeAsync(joltqc_j, stream);
        cudaFreeAsync(joltqc_density, stream);
        return ffi::Error::Internal(cudaGetErrorString(fast_err));
    }
    if (fast_err == cudaErrorNotSupported) {
        cudaGetLastError();
        const int quartet_grid = nquartets > 65535 ? 65535 : static_cast<int>(nquartets);
        joltqc_shell_quartet_direct_jk_kernel<<<quartet_grid, block, 0, stream>>>(
            joltqc_nao,
            basis_data.typed_data(),
            shell_l.typed_data(),
            shell_nprims.typed_data(),
            joltqc_density,
            shell_quartets.typed_data(),
            nquartets,
            joltqc_j,
            joltqc_k);
        if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
            cudaFreeAsync(joltqc_k, stream);
            cudaFreeAsync(joltqc_j, stream);
            cudaFreeAsync(joltqc_density, stream);
            return ffi::Error::Internal(cudaGetErrorString(err));
        }
    }
    if (fast_err == cudaSuccess) {
        const int finalize_grid =
            static_cast<int>((static_cast<long long>(joltqc_nao) * joltqc_nao + block - 1) / block);
        finalize_joltqc_potential_kernel<<<finalize_grid, block, 0, stream>>>(
            joltqc_nao,
            joltqc_j,
            joltqc_k);
        if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
            cudaFreeAsync(joltqc_k, stream);
            cudaFreeAsync(joltqc_j, stream);
            cudaFreeAsync(joltqc_density, stream);
            return ffi::Error::Internal(cudaGetErrorString(err));
        }
    }

    contract_joltqc_potential_kernel<<<internal_grid, block, 0, stream>>>(
        nao,
        joltqc_nao,
        ao_to_parent_ao.typed_data(),
        joltqc_j,
        joltqc_k,
        j_out->typed_data(),
        k_out->typed_data());
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        cudaFreeAsync(joltqc_k, stream);
        cudaFreeAsync(joltqc_j, stream);
        cudaFreeAsync(joltqc_density, stream);
        return ffi::Error::Internal(cudaGetErrorString(err));
    }

    if (fast_err == cudaErrorNotSupported) {
        const int sym_grid = static_cast<int>((static_cast<long long>(nao) * nao + block - 1) / block);
        symmetrize_kernel<<<sym_grid, block, 0, stream>>>(nao, j_out->typed_data(), k_out->typed_data());
        if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
            cudaFreeAsync(joltqc_k, stream);
            cudaFreeAsync(joltqc_j, stream);
            cudaFreeAsync(joltqc_density, stream);
            return ffi::Error::Internal(cudaGetErrorString(err));
        }
    }

    cudaFreeAsync(joltqc_k, stream);
    cudaFreeAsync(joltqc_j, stream);
    cudaFreeAsync(joltqc_density, stream);
    return ffi::Error::Success();
}

ffi::Error CudaDirectJkDispatch(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64, 2> density,
    ffi::Buffer<ffi::F64, 2> centers,
    ffi::Buffer<ffi::S32, 2> angulars,
    ffi::Buffer<ffi::F64, 2> pair_exponents,
    ffi::Buffer<ffi::F64, 3> pair_centers,
    ffi::Buffer<ffi::F64, 2> pair_prefactors,
    ffi::Buffer<ffi::S32, 1> pair_nprims,
    ffi::Buffer<ffi::S32, 1> pair_rows,
    ffi::Buffer<ffi::S32, 1> pair_cols,
    ffi::Buffer<ffi::F64, 1> pair_schwarz,
    ffi::Buffer<ffi::F64, 1> density_cutoff,
    ffi::Result<ffi::Buffer<ffi::F64, 2>> j_out,
    ffi::Result<ffi::Buffer<ffi::F64, 2>> k_out) {
    const int64_t nao64 = density.dimensions()[0];
    if (density.dimensions()[1] != nao64) {
        return ffi::Error::InvalidArgument("density must be square");
    }
    if (centers.dimensions()[0] != nao64 || centers.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("centers must have shape (nao, 3)");
    }
    if (angulars.dimensions()[0] != nao64 || angulars.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("angulars must have shape (nao, 3)");
    }
    const int64_t npair64 = nao64 * (nao64 + 1) / 2;
    if (pair_exponents.dimensions()[0] != npair64 || pair_prefactors.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair primitive arrays must have leading dimension nao*(nao+1)/2");
    }
    if (pair_exponents.dimensions()[1] != pair_prefactors.dimensions()[1]) {
        return ffi::Error::InvalidArgument("pair exponents and prefactors must share max_pair_nprim");
    }
    if (
        pair_centers.dimensions()[0] != npair64 ||
        pair_centers.dimensions()[1] != pair_exponents.dimensions()[1] ||
        pair_centers.dimensions()[2] != 3
    ) {
        return ffi::Error::InvalidArgument("pair_centers must have shape (npair, max_pair_nprim, 3)");
    }
    if (pair_nprims.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair_nprims must have shape (npair,)");
    }
    if (pair_rows.dimensions()[0] != npair64 || pair_cols.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair rows/cols must have shape (npair,)");
    }
    if (pair_schwarz.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair_schwarz must have shape (npair,)");
    }
    if (density_cutoff.dimensions()[0] != 1) {
        return ffi::Error::InvalidArgument("density_cutoff must have shape (1,)");
    }

    const int nao = static_cast<int>(nao64);
    const int max_pair_nprim = static_cast<int>(pair_exponents.dimensions()[1]);
    const size_t matrix_bytes = static_cast<size_t>(nao) * static_cast<size_t>(nao) * sizeof(double);

    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    cudaMemsetAsync(j_out->typed_data(), 0, matrix_bytes, stream);
    cudaMemsetAsync(k_out->typed_data(), 0, matrix_bytes, stream);

    const long long npair = static_cast<long long>(nao) * (nao + 1) / 2;
    const long long total = npair * (npair + 1) / 2;
    const int block = 128;
    const int grid = total > 0 ? static_cast<int>((total + block - 1) / block) : 1;
    unique_pair_quartet_direct_jk_kernel<<<grid, block, 0, stream>>>(
        nao,
        max_pair_nprim,
        pair_nprims.typed_data(),
        angulars.typed_data(),
        centers.typed_data(),
        pair_exponents.typed_data(),
        pair_centers.typed_data(),
        pair_prefactors.typed_data(),
        density.typed_data(),
        pair_rows.typed_data(),
        pair_cols.typed_data(),
        pair_schwarz.typed_data(),
        density_cutoff.typed_data(),
        j_out->typed_data(),
        k_out->typed_data());
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(err));
    }

    const int sym_grid = static_cast<int>((static_cast<long long>(nao) * nao + block - 1) / block);
    symmetrize_kernel<<<sym_grid, block, 0, stream>>>(nao, j_out->typed_data(), k_out->typed_data());
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

ffi::Error CudaScreenedDirectJkDispatch(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64, 2> density,
    ffi::Buffer<ffi::F64, 2> centers,
    ffi::Buffer<ffi::S32, 2> angulars,
    ffi::Buffer<ffi::F64, 2> pair_exponents,
    ffi::Buffer<ffi::F64, 3> pair_centers,
    ffi::Buffer<ffi::F64, 2> pair_prefactors,
    ffi::Buffer<ffi::S32, 1> pair_nprims,
    ffi::Buffer<ffi::S32, 1> pair_rows,
    ffi::Buffer<ffi::S32, 1> pair_cols,
    ffi::Buffer<ffi::F64, 1> pair_schwarz,
    ffi::Buffer<ffi::F64, 1> density_cutoff,
    ffi::Buffer<ffi::F64, 2> shell_log_q_matrix,
    ffi::Buffer<ffi::F64, 2> shell_dm_cond,
    ffi::Buffer<ffi::S32, 2> shell_ao_indices_padded,
    ffi::Buffer<ffi::S32, 1> shell_ao_sizes,
    ffi::Buffer<ffi::S32, 2> tile_shell_indices,
    ffi::Buffer<ffi::S32, 2> tile_shell_pad_mask,
    ffi::Buffer<ffi::S32, 1> tile_pair_ids,
    ffi::Result<ffi::Buffer<ffi::F64, 2>> j_out,
    ffi::Result<ffi::Buffer<ffi::F64, 2>> k_out) {
    const int64_t nao64 = density.dimensions()[0];
    if (density.dimensions()[1] != nao64) {
        return ffi::Error::InvalidArgument("density must be square");
    }
    if (centers.dimensions()[0] != nao64 || centers.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("centers must have shape (nao, 3)");
    }
    if (angulars.dimensions()[0] != nao64 || angulars.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("angulars must have shape (nao, 3)");
    }
    const int64_t npair64 = nao64 * (nao64 + 1) / 2;
    if (pair_exponents.dimensions()[0] != npair64 || pair_prefactors.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair primitive arrays must have leading dimension nao*(nao+1)/2");
    }
    if (pair_exponents.dimensions()[1] != pair_prefactors.dimensions()[1]) {
        return ffi::Error::InvalidArgument("pair exponents and prefactors must share max_pair_nprim");
    }
    if (
        pair_centers.dimensions()[0] != npair64 ||
        pair_centers.dimensions()[1] != pair_exponents.dimensions()[1] ||
        pair_centers.dimensions()[2] != 3
    ) {
        return ffi::Error::InvalidArgument("pair_centers must have shape (npair, max_pair_nprim, 3)");
    }
    if (pair_nprims.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair_nprims must have shape (npair,)");
    }
    if (pair_rows.dimensions()[0] != npair64 || pair_cols.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair rows/cols must have shape (npair,)");
    }
    if (pair_schwarz.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair_schwarz must have shape (npair,)");
    }
    if (density_cutoff.dimensions()[0] != 1) {
        return ffi::Error::InvalidArgument("density_cutoff must have shape (1,)");
    }
    const int64_t nshell64 = shell_ao_sizes.dimensions()[0];
    if (shell_log_q_matrix.dimensions()[0] != nshell64 || shell_log_q_matrix.dimensions()[1] != nshell64) {
        return ffi::Error::InvalidArgument("shell_log_q_matrix must have shape (nshell, nshell)");
    }
    if (shell_dm_cond.dimensions()[0] != nshell64 || shell_dm_cond.dimensions()[1] != nshell64) {
        return ffi::Error::InvalidArgument("shell_dm_cond must have shape (nshell, nshell)");
    }
    if (shell_ao_indices_padded.dimensions()[0] != nshell64) {
        return ffi::Error::InvalidArgument("shell_ao_indices_padded must have leading dimension nshell");
    }
    if (tile_shell_indices.dimensions()[0] != tile_shell_pad_mask.dimensions()[0] ||
        tile_shell_indices.dimensions()[1] != tile_shell_pad_mask.dimensions()[1]) {
        return ffi::Error::InvalidArgument("tile shell arrays must share the same shape");
    }

    const int nao = static_cast<int>(nao64);
    const int nshell = static_cast<int>(nshell64);
    const int max_pair_nprim = static_cast<int>(pair_exponents.dimensions()[1]);
    const int max_shell_ao = static_cast<int>(shell_ao_indices_padded.dimensions()[1]);
    const int ntiles = static_cast<int>(tile_shell_indices.dimensions()[0]);
    const int tile_size = static_cast<int>(tile_shell_indices.dimensions()[1]);
    const size_t matrix_bytes = static_cast<size_t>(nao) * static_cast<size_t>(nao) * sizeof(double);

    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    cudaMemsetAsync(j_out->typed_data(), 0, matrix_bytes, stream);
    cudaMemsetAsync(k_out->typed_data(), 0, matrix_bytes, stream);

    const int block = 128;
    const long long n_tile_pairs = tile_pair_ids.dimensions()[0];
    if (n_tile_pairs > 0 && nshell > 0) {
        const long long nshell_pair = static_cast<long long>(nshell) * (nshell + 1) / 2;
        const long long max_tasks = nshell_pair * (nshell_pair + 1) / 2;
        ShellQuartetTask* shell_quartet_tasks = nullptr;
        unsigned int* task_count = nullptr;
        const size_t task_bytes = static_cast<size_t>(max_tasks) * sizeof(ShellQuartetTask);
        if (cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(&shell_quartet_tasks), task_bytes, stream); err != cudaSuccess) {
            return ffi::Error::Internal(cudaGetErrorString(err));
        }
        if (cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(&task_count), sizeof(unsigned int), stream); err != cudaSuccess) {
            cudaFreeAsync(shell_quartet_tasks, stream);
            return ffi::Error::Internal(cudaGetErrorString(err));
        }
        cudaMemsetAsync(task_count, 0, sizeof(unsigned int), stream);

        const long long screen_total = n_tile_pairs * n_tile_pairs;
        const long long screen_grid64 = screen_total > 0 ? (screen_total + block - 1) / block : 1;
        const int screen_grid = screen_grid64 > 65535 ? 65535 : static_cast<int>(screen_grid64);
        screen_shell_quartet_tasks<<<screen_grid, block, 0, stream>>>(
            nshell,
            ntiles,
            tile_size,
            tile_pair_ids.typed_data(),
            n_tile_pairs,
            tile_shell_indices.typed_data(),
            tile_shell_pad_mask.typed_data(),
            shell_log_q_matrix.typed_data(),
            shell_dm_cond.typed_data(),
            density_cutoff.typed_data(),
            max_tasks,
            shell_quartet_tasks,
            task_count);
        if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
            cudaFreeAsync(task_count, stream);
            cudaFreeAsync(shell_quartet_tasks, stream);
            return ffi::Error::Internal(cudaGetErrorString(err));
        }

        const long long task_grid64 = max_tasks > 0 ? max_tasks : 1;
        const int task_grid = task_grid64 > 65535 ? 65535 : static_cast<int>(task_grid64);
        screened_shell_quartet_direct_jk_kernel<<<task_grid, block, 0, stream>>>(
            nao,
            max_pair_nprim,
            max_shell_ao,
            pair_nprims.typed_data(),
            angulars.typed_data(),
            centers.typed_data(),
            pair_exponents.typed_data(),
            pair_centers.typed_data(),
            pair_prefactors.typed_data(),
            density.typed_data(),
            shell_ao_indices_padded.typed_data(),
            shell_ao_sizes.typed_data(),
            shell_quartet_tasks,
            task_count,
            max_tasks,
            j_out->typed_data(),
            k_out->typed_data());
        if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
            cudaFreeAsync(task_count, stream);
            cudaFreeAsync(shell_quartet_tasks, stream);
            return ffi::Error::Internal(cudaGetErrorString(err));
        }
        cudaFreeAsync(task_count, stream);
        cudaFreeAsync(shell_quartet_tasks, stream);
    }

    const int sym_grid = static_cast<int>((static_cast<long long>(nao) * nao + block - 1) / block);
    symmetrize_kernel<<<sym_grid, block, 0, stream>>>(nao, j_out->typed_data(), k_out->typed_data());
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

#ifdef TD_GRADDFT_ENABLE_GPU4PYSCF_RYS
ffi::Error CudaGpu4PyScfRysDirectJkDispatch(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64, 2> density,
    ffi::Buffer<ffi::S32, 2> rys_atm,
    ffi::Buffer<ffi::S32, 2> rys_bas,
    ffi::Buffer<ffi::F64, 1> rys_env,
    ffi::Buffer<ffi::S32, 1> rys_ao_loc,
    ffi::Buffer<ffi::S32, 1> ao_to_parent_ao,
    ffi::Buffer<ffi::S32, 1> group_offsets,
    ffi::Buffer<ffi::F32, 2> q_cond,
    ffi::Buffer<ffi::F32, 2> dm_cond,
    ffi::Buffer<ffi::S32, 1> pair_offsets,
    ffi::Buffer<ffi::S32, 1> pair_ids,
    ffi::Buffer<ffi::F32, 1> log_cutoff,
    ffi::Result<ffi::Buffer<ffi::F64, 2>> j_out,
    ffi::Result<ffi::Buffer<ffi::F64, 2>> k_out) {
    const int nao = static_cast<int>(density.dimensions()[0]);
    if (density.dimensions()[1] != nao) {
        return ffi::Error::InvalidArgument("density must be square");
    }
    if (j_out->dimensions()[0] != nao || j_out->dimensions()[1] != nao ||
        k_out->dimensions()[0] != nao || k_out->dimensions()[1] != nao) {
        return ffi::Error::InvalidArgument("J/K outputs must match density shape");
    }
    if (rys_atm.dimensions()[1] != ATM_SLOTS) {
        return ffi::Error::InvalidArgument("rys_atm must have shape (natm, 6)");
    }
    if (rys_bas.dimensions()[1] != BAS_SLOTS) {
        return ffi::Error::InvalidArgument("rys_bas must have shape (nbas, 8)");
    }
    const int natm = static_cast<int>(rys_atm.dimensions()[0]);
    const int nbas = static_cast<int>(rys_bas.dimensions()[0]);
    if (rys_ao_loc.dimensions()[0] != nbas + 1) {
        return ffi::Error::InvalidArgument("rys_ao_loc must have shape (nbas + 1,)");
    }
    const int sorted_nao = static_cast<int>(ao_to_parent_ao.dimensions()[0]);
    if (q_cond.dimensions()[0] != nbas || q_cond.dimensions()[1] != nbas ||
        dm_cond.dimensions()[0] != nbas || dm_cond.dimensions()[1] != nbas) {
        return ffi::Error::InvalidArgument("q_cond/dm_cond must have shape (nbas, nbas)");
    }
    if (group_offsets.dimensions()[0] < 1) {
        return ffi::Error::InvalidArgument("group_offsets must be nonempty");
    }
    if (log_cutoff.dimensions()[0] != 1) {
        return ffi::Error::InvalidArgument("log_cutoff must have shape (1,)");
    }

    const int n_groups = static_cast<int>(group_offsets.dimensions()[0] - 1);
    const int n_group_pairs = n_groups * (n_groups + 1) / 2;
    if (pair_offsets.dimensions()[0] != n_group_pairs + 1) {
        return ffi::Error::InvalidArgument("pair_offsets must cover all lower-triangle group pairs");
    }

    std::vector<int> h_group_offsets(static_cast<size_t>(group_offsets.dimensions()[0]));
    std::vector<int> h_pair_offsets(static_cast<size_t>(pair_offsets.dimensions()[0]));
    std::vector<int> h_atm(static_cast<size_t>(natm) * ATM_SLOTS);
    std::vector<int> h_bas(static_cast<size_t>(nbas) * BAS_SLOTS);
    std::vector<double> h_env(static_cast<size_t>(rys_env.dimensions()[0]));
    std::vector<float> h_cutoff(1);

    cudaError_t err = cudaMemcpyAsync(
        h_group_offsets.data(), group_offsets.typed_data(),
        h_group_offsets.size() * sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    err = cudaMemcpyAsync(
        h_pair_offsets.data(), pair_offsets.typed_data(),
        h_pair_offsets.size() * sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    err = cudaMemcpyAsync(
        h_atm.data(), rys_atm.typed_data(),
        h_atm.size() * sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    err = cudaMemcpyAsync(
        h_bas.data(), rys_bas.typed_data(),
        h_bas.size() * sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    err = cudaMemcpyAsync(
        h_env.data(), rys_env.typed_data(),
        h_env.size() * sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    err = cudaMemcpyAsync(
        h_cutoff.data(), log_cutoff.typed_data(), sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));

    if (nbas > 0 && h_group_offsets.back() != nbas) {
        return ffi::Error::InvalidArgument("group_offsets must end at nbas");
    }

    const size_t out_matrix_bytes = static_cast<size_t>(nao) * nao * sizeof(double);
    const size_t sorted_matrix_bytes =
        static_cast<size_t>(sorted_nao) * sorted_nao * sizeof(double);
    cudaMemsetAsync(j_out->typed_data(), 0, out_matrix_bytes, stream);
    cudaMemsetAsync(k_out->typed_data(), 0, out_matrix_bytes, stream);
    if (sorted_nao == 0 || nbas == 0) {
        return ffi::Error::Success();
    }

    double* dm_sorted = nullptr;
    double* vj_sorted = nullptr;
    double* vk_sorted = nullptr;
    int* pool = nullptr;
    err = cudaMallocAsync(reinterpret_cast<void**>(&dm_sorted), sorted_matrix_bytes, stream);
    if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
    err = cudaMallocAsync(reinterpret_cast<void**>(&vj_sorted), sorted_matrix_bytes, stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(dm_sorted, stream);
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    err = cudaMallocAsync(reinterpret_cast<void**>(&vk_sorted), sorted_matrix_bytes, stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(vj_sorted, stream);
        cudaFreeAsync(dm_sorted, stream);
        return ffi::Error::Internal(cudaGetErrorString(err));
    }

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    const int workers = prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
    constexpr int kRysQueueDepth = 262144;
    constexpr int kRysShmSize = 48 * 1024 - 1024;
    err = cudaMallocAsync(
        reinterpret_cast<void**>(&pool),
        static_cast<size_t>(workers) * kRysQueueDepth * sizeof(int) + sizeof(int),
        stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(vk_sorted, stream);
        cudaFreeAsync(vj_sorted, stream);
        cudaFreeAsync(dm_sorted, stream);
        return ffi::Error::Internal(cudaGetErrorString(err));
    }

    const int block = 256;
    const int sorted_grid =
        static_cast<int>((static_cast<long long>(sorted_nao) * sorted_nao + block - 1) / block);
    expand_joltqc_density_kernel<<<sorted_grid, block, 0, stream>>>(
        nao, sorted_nao, ao_to_parent_ao.typed_data(), density.typed_data(), dm_sorted);
    cudaMemsetAsync(vj_sorted, 0, sorted_matrix_bytes, stream);
    cudaMemsetAsync(vk_sorted, 0, sorted_matrix_bytes, stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFreeAsync(pool, stream);
        cudaFreeAsync(vk_sorted, stream);
        cudaFreeAsync(vj_sorted, stream);
        cudaFreeAsync(dm_sorted, stream);
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        cudaFreeAsync(pool, stream);
        cudaFreeAsync(vk_sorted, stream);
        cudaFreeAsync(vj_sorted, stream);
        cudaFreeAsync(dm_sorted, stream);
        return ffi::Error::Internal(cudaGetErrorString(err));
    }

    static bool rys_initialized = false;
    if (!rys_initialized) {
        if (RYS_build_jk_init(kRysShmSize) != 0) {
            cudaFreeAsync(pool, stream);
            cudaFreeAsync(vk_sorted, stream);
            cudaFreeAsync(vj_sorted, stream);
            cudaFreeAsync(dm_sorted, stream);
            return ffi::Error::Internal("RYS_build_jk_init failed");
        }
        rys_initialized = true;
    }

    RysIntEnvVars envs = {
        natm,
        nbas,
        const_cast<int*>(rys_atm.typed_data()),
        const_cast<int*>(rys_bas.typed_data()),
        const_cast<double*>(rys_env.typed_data()),
        const_cast<int*>(rys_ao_loc.typed_data()),
    };
    uint32_t* pair_mapping = reinterpret_cast<uint32_t*>(
        const_cast<int*>(pair_ids.typed_data()));
    float* q_ptr = const_cast<float*>(q_cond.typed_data());
    float* dm_ptr = const_cast<float*>(dm_cond.typed_data());

    for (int i = 0; i < n_groups; ++i) {
        for (int j = 0; j <= i; ++j) {
            const int ij_slot = i * (i + 1) / 2 + j;
            const int ij0 = h_pair_offsets[ij_slot];
            const int ij1 = h_pair_offsets[ij_slot + 1];
            const int npairs_ij = ij1 - ij0;
            if (npairs_ij <= 0) {
                continue;
            }
            for (int k = 0; k <= i; ++k) {
                for (int l = 0; l <= k; ++l) {
                    const int kl_slot = k * (k + 1) / 2 + l;
                    const int kl0 = h_pair_offsets[kl_slot];
                    const int kl1 = h_pair_offsets[kl_slot + 1];
                    const int npairs_kl = kl1 - kl0;
                    if (npairs_kl <= 0) {
                        continue;
                    }
                    int shls_slice[8] = {
                        h_group_offsets[i], h_group_offsets[i + 1],
                        h_group_offsets[j], h_group_offsets[j + 1],
                        h_group_offsets[k], h_group_offsets[k + 1],
                        h_group_offsets[l], h_group_offsets[l + 1],
                    };
                    const int status = RYS_build_jk(
                        vj_sorted, vk_sorted, dm_sorted, 1, sorted_nao,
                        &envs, shls_slice, kRysShmSize,
                        npairs_ij, npairs_kl,
                        pair_mapping + ij0, pair_mapping + kl0,
                        q_ptr, nullptr, dm_ptr, h_cutoff[0],
                        pool, h_atm.data(), natm, h_bas.data(), nbas, h_env.data());
                    if (status != 0) {
                        cudaFreeAsync(pool, stream);
                        cudaFreeAsync(vk_sorted, stream);
                        cudaFreeAsync(vj_sorted, stream);
                        cudaFreeAsync(dm_sorted, stream);
                        return ffi::Error::Internal("RYS_build_jk failed");
                    }
                }
            }
        }
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFreeAsync(pool, stream);
        cudaFreeAsync(vk_sorted, stream);
        cudaFreeAsync(vj_sorted, stream);
        cudaFreeAsync(dm_sorted, stream);
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    finalize_joltqc_potential_kernel<<<sorted_grid, block, 0, stream>>>(
        sorted_nao, vj_sorted, vk_sorted);
    contract_joltqc_potential_kernel<<<sorted_grid, block, 0, stream>>>(
        nao, sorted_nao, ao_to_parent_ao.typed_data(),
        vj_sorted, vk_sorted, j_out->typed_data(), k_out->typed_data());
    err = cudaGetLastError();
    cudaFreeAsync(pool, stream);
    cudaFreeAsync(vk_sorted, stream);
    cudaFreeAsync(vj_sorted, stream);
    cudaFreeAsync(dm_sorted, stream);
    if (err != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}
#endif

ffi::Error CudaPairSchwarzDispatch(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64, 2> centers,
    ffi::Buffer<ffi::S32, 2> angulars,
    ffi::Buffer<ffi::F64, 2> pair_exponents,
    ffi::Buffer<ffi::F64, 3> pair_centers,
    ffi::Buffer<ffi::F64, 2> pair_prefactors,
    ffi::Buffer<ffi::S32, 1> pair_nprims,
    ffi::Buffer<ffi::S32, 1> pair_rows,
    ffi::Buffer<ffi::S32, 1> pair_cols,
    ffi::Result<ffi::Buffer<ffi::F64, 1>> pair_schwarz_out) {
    const int64_t nao64 = centers.dimensions()[0];
    if (centers.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("centers must have shape (nao, 3)");
    }
    if (angulars.dimensions()[0] != nao64 || angulars.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("angulars must have shape (nao, 3)");
    }
    const int64_t npair64 = nao64 * (nao64 + 1) / 2;
    if (pair_schwarz_out->dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair_schwarz output must have shape (npair,)");
    }
    if (pair_exponents.dimensions()[0] != npair64 || pair_prefactors.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair primitive arrays must have leading dimension nao*(nao+1)/2");
    }
    if (pair_exponents.dimensions()[1] != pair_prefactors.dimensions()[1]) {
        return ffi::Error::InvalidArgument("pair exponents and prefactors must share max_pair_nprim");
    }
    if (
        pair_centers.dimensions()[0] != npair64 ||
        pair_centers.dimensions()[1] != pair_exponents.dimensions()[1] ||
        pair_centers.dimensions()[2] != 3
    ) {
        return ffi::Error::InvalidArgument("pair_centers must have shape (npair, max_pair_nprim, 3)");
    }
    if (pair_nprims.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair_nprims must have shape (npair,)");
    }
    if (pair_rows.dimensions()[0] != npair64 || pair_cols.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair rows/cols must have shape (npair,)");
    }

    const int nao = static_cast<int>(nao64);
    const int max_pair_nprim = static_cast<int>(pair_exponents.dimensions()[1]);

    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    const int block = 128;
    const int grid = npair64 > 0 ? static_cast<int>((npair64 + block - 1) / block) : 1;
    pair_schwarz_kernel<<<grid, block, 0, stream>>>(
        nao,
        max_pair_nprim,
        pair_nprims.typed_data(),
        angulars.typed_data(),
        centers.typed_data(),
        pair_exponents.typed_data(),
        pair_centers.typed_data(),
        pair_prefactors.typed_data(),
        pair_rows.typed_data(),
        pair_cols.typed_data(),
        pair_schwarz_out->typed_data());
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

ffi::Error CudaEriPairMatrixDispatch(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64, 2> centers,
    ffi::Buffer<ffi::S32, 2> angulars,
    ffi::Buffer<ffi::F64, 2> pair_exponents,
    ffi::Buffer<ffi::F64, 3> pair_centers,
    ffi::Buffer<ffi::F64, 2> pair_prefactors,
    ffi::Buffer<ffi::S32, 1> pair_nprims,
    ffi::Buffer<ffi::S32, 1> pair_rows,
    ffi::Buffer<ffi::S32, 1> pair_cols,
    ffi::Buffer<ffi::F64, 1> pair_schwarz,
    ffi::Buffer<ffi::F64, 1> eri_cutoff,
    ffi::Result<ffi::Buffer<ffi::F64, 2>> eri_pair_out) {
    const int64_t nao64 = centers.dimensions()[0];
    if (centers.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("centers must have shape (nao, 3)");
    }
    if (angulars.dimensions()[0] != nao64 || angulars.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("angulars must have shape (nao, 3)");
    }
    const int64_t npair64 = nao64 * (nao64 + 1) / 2;
    if (eri_pair_out->dimensions()[0] != npair64 || eri_pair_out->dimensions()[1] != npair64) {
        return ffi::Error::InvalidArgument("eri pair output must have shape (npair, npair)");
    }
    if (pair_exponents.dimensions()[0] != npair64 || pair_prefactors.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair primitive arrays must have leading dimension nao*(nao+1)/2");
    }
    if (pair_exponents.dimensions()[1] != pair_prefactors.dimensions()[1]) {
        return ffi::Error::InvalidArgument("pair exponents and prefactors must share max_pair_nprim");
    }
    if (
        pair_centers.dimensions()[0] != npair64 ||
        pair_centers.dimensions()[1] != pair_exponents.dimensions()[1] ||
        pair_centers.dimensions()[2] != 3
    ) {
        return ffi::Error::InvalidArgument("pair_centers must have shape (npair, max_pair_nprim, 3)");
    }
    if (pair_nprims.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair_nprims must have shape (npair,)");
    }
    if (pair_rows.dimensions()[0] != npair64 || pair_cols.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair rows/cols must have shape (npair,)");
    }
    if (pair_schwarz.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair_schwarz must have shape (npair,)");
    }
    if (eri_cutoff.dimensions()[0] != 1) {
        return ffi::Error::InvalidArgument("eri_cutoff must have shape (1,)");
    }

    const int nao = static_cast<int>(nao64);
    const int max_pair_nprim = static_cast<int>(pair_exponents.dimensions()[1]);

    const size_t pair_bytes = static_cast<size_t>(npair64) * static_cast<size_t>(npair64) * sizeof(double);

    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    cudaMemsetAsync(eri_pair_out->typed_data(), 0, pair_bytes, stream);
    const long long npair = static_cast<long long>(nao) * (nao + 1) / 2;
    const long long total = npair * (npair + 1) / 2;
    const int eri_pair_block = 256;
    const int grid = total > 0 ? static_cast<int>((total + eri_pair_block - 1) / eri_pair_block) : 1;
    unique_pair_quartet_eri_pair_matrix_kernel<<<grid, eri_pair_block, 0, stream>>>(
        nao,
        max_pair_nprim,
        pair_nprims.typed_data(),
        angulars.typed_data(),
        centers.typed_data(),
        pair_exponents.typed_data(),
        pair_centers.typed_data(),
        pair_prefactors.typed_data(),
        pair_rows.typed_data(),
        pair_cols.typed_data(),
        pair_schwarz.typed_data(),
        eri_cutoff.typed_data(),
        eri_pair_out->typed_data());
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

ffi::Error CudaPairMatrixJkDispatch(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64, 2> eri_pair_matrix,
    ffi::Buffer<ffi::F64, 2> density,
    ffi::Buffer<ffi::S32, 1> pair_rows,
    ffi::Buffer<ffi::S32, 1> pair_cols,
    ffi::Result<ffi::Buffer<ffi::F64, 2>> j_out,
    ffi::Result<ffi::Buffer<ffi::F64, 2>> k_out) {
    const int64_t nao64 = density.dimensions()[0];
    if (density.dimensions()[1] != nao64) {
        return ffi::Error::InvalidArgument("density must be square");
    }
    const int64_t npair64 = nao64 * (nao64 + 1) / 2;
    if (eri_pair_matrix.dimensions()[0] != npair64 || eri_pair_matrix.dimensions()[1] != npair64) {
        return ffi::Error::InvalidArgument("eri_pair_matrix must have shape (npair, npair)");
    }
    if (pair_rows.dimensions()[0] != npair64 || pair_cols.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair rows/cols must have shape (npair,)");
    }
    const int nao = static_cast<int>(nao64);
    const size_t matrix_bytes = static_cast<size_t>(nao) * static_cast<size_t>(nao) * sizeof(double);
    cudaMemsetAsync(j_out->typed_data(), 0, matrix_bytes, stream);
    cudaMemsetAsync(k_out->typed_data(), 0, matrix_bytes, stream);

    const int block = nao >= kPairMatrixReductionMinNao ? 256 : 128;
    if (nao >= kPairMatrixReductionMinNao) {
        const int j_grid = static_cast<int>(npair64);
        pair_matrix_j_reduce_kernel<<<j_grid, block, block * sizeof(double), stream>>>(
            nao,
            eri_pair_matrix.typed_data(),
            density.typed_data(),
            pair_rows.typed_data(),
            pair_cols.typed_data(),
            j_out->typed_data());
    } else {
        const int j_grid = static_cast<int>((npair64 + block - 1) / block);
        pair_matrix_j_kernel<<<j_grid, block, 0, stream>>>(
            nao,
            eri_pair_matrix.typed_data(),
            density.typed_data(),
            pair_rows.typed_data(),
            pair_cols.typed_data(),
            j_out->typed_data());
    }
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(err));
    }

    if (nao >= kPairMatrixReductionMinNao) {
        const int k_grid = static_cast<int>(npair64);
        pair_matrix_k_reduce_kernel<<<k_grid, block, block * sizeof(double), stream>>>(
            nao,
            eri_pair_matrix.typed_data(),
            density.typed_data(),
            pair_rows.typed_data(),
            pair_cols.typed_data(),
            k_out->typed_data());
    } else {
        const int k_grid = static_cast<int>((npair64 + block - 1) / block);
        pair_matrix_k_kernel<<<k_grid, block, 0, stream>>>(
            nao,
            eri_pair_matrix.typed_data(),
            density.typed_data(),
            pair_rows.typed_data(),
            pair_cols.typed_data(),
            k_out->typed_data());
    }
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

ffi::Error CudaEriTensorDispatch(
    cudaStream_t stream,
    ffi::Buffer<ffi::F64, 2> centers,
    ffi::Buffer<ffi::S32, 2> angulars,
    ffi::Buffer<ffi::F64, 2> pair_exponents,
    ffi::Buffer<ffi::F64, 3> pair_centers,
    ffi::Buffer<ffi::F64, 2> pair_prefactors,
    ffi::Buffer<ffi::S32, 1> pair_nprims,
    ffi::Buffer<ffi::S32, 1> pair_rows,
    ffi::Buffer<ffi::S32, 1> pair_cols,
    ffi::Result<ffi::Buffer<ffi::F64, 4>> eri_out) {
    const int64_t nao64 = centers.dimensions()[0];
    if (centers.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("centers must have shape (nao, 3)");
    }
    if (angulars.dimensions()[0] != nao64 || angulars.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("angulars must have shape (nao, 3)");
    }
    const int64_t npair64 = nao64 * (nao64 + 1) / 2;
    if (pair_exponents.dimensions()[0] != npair64 || pair_prefactors.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair primitive arrays must have leading dimension nao*(nao+1)/2");
    }
    if (pair_exponents.dimensions()[1] != pair_prefactors.dimensions()[1]) {
        return ffi::Error::InvalidArgument("pair exponents and prefactors must share max_pair_nprim");
    }
    if (
        pair_centers.dimensions()[0] != npair64 ||
        pair_centers.dimensions()[1] != pair_exponents.dimensions()[1] ||
        pair_centers.dimensions()[2] != 3
    ) {
        return ffi::Error::InvalidArgument("pair_centers must have shape (npair, max_pair_nprim, 3)");
    }
    if (pair_nprims.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair_nprims must have shape (npair,)");
    }
    if (pair_rows.dimensions()[0] != npair64 || pair_cols.dimensions()[0] != npair64) {
        return ffi::Error::InvalidArgument("pair rows/cols must have shape (npair,)");
    }
    if (
        eri_out->dimensions()[0] != nao64 ||
        eri_out->dimensions()[1] != nao64 ||
        eri_out->dimensions()[2] != nao64 ||
        eri_out->dimensions()[3] != nao64
    ) {
        return ffi::Error::InvalidArgument("eri output must have shape (nao, nao, nao, nao)");
    }

    const int nao = static_cast<int>(nao64);
    const int max_pair_nprim = static_cast<int>(pair_exponents.dimensions()[1]);
    const size_t eri_bytes =
        static_cast<size_t>(nao) * static_cast<size_t>(nao) *
        static_cast<size_t>(nao) * static_cast<size_t>(nao) * sizeof(double);

    cudaDeviceSetLimit(cudaLimitStackSize, 16384);
    cudaMemsetAsync(eri_out->typed_data(), 0, eri_bytes, stream);

    const long long npair = static_cast<long long>(nao) * (nao + 1) / 2;
    const long long total = npair * (npair + 1) / 2;
    const int block = 128;
    const int grid = total > 0 ? static_cast<int>((total + block - 1) / block) : 1;
    unique_pair_quartet_eri_tensor_kernel<<<grid, block, 0, stream>>>(
        nao,
        max_pair_nprim,
        pair_nprims.typed_data(),
        angulars.typed_data(),
        centers.typed_data(),
        pair_exponents.typed_data(),
        pair_centers.typed_data(),
        pair_prefactors.typed_data(),
        pair_rows.typed_data(),
        pair_cols.typed_data(),
        eri_out->typed_data());
    if (cudaError_t err = cudaGetLastError(); err != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}

using StreamCtx = ffi::PlatformStream<cudaStream_t>;
using F64Buffer2 = ffi::Buffer<ffi::F64, 2>;
using F64Buffer3 = ffi::Buffer<ffi::F64, 3>;
using F64Buffer4 = ffi::Buffer<ffi::F64, 4>;
using F64Buffer1 = ffi::Buffer<ffi::F64, 1>;
using F32Buffer2 = ffi::Buffer<ffi::F32, 2>;
using F32Buffer1 = ffi::Buffer<ffi::F32, 1>;
using S32Buffer2 = ffi::Buffer<ffi::S32, 2>;
using S32Buffer1 = ffi::Buffer<ffi::S32, 1>;

auto CudaJoltQCDirectJkBinding() {
    return ffi::Ffi::Bind()
        .Ctx<StreamCtx>()
        .Arg<F64Buffer2>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer2>()
        .Arg<S32Buffer2>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer2>()
        .Ret<F64Buffer2>()
        .Ret<F64Buffer2>();
}

auto CudaDirectJkBinding() {
    return ffi::Ffi::Bind()
        .Ctx<StreamCtx>()
        .Arg<F64Buffer2>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer2>()
        .Arg<F64Buffer2>()
        .Arg<F64Buffer3>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<F64Buffer1>()
        .Arg<F64Buffer1>()
        .Ret<F64Buffer2>()
        .Ret<F64Buffer2>();
}

auto CudaScreenedDirectJkBinding() {
    return ffi::Ffi::Bind()
        .Ctx<StreamCtx>()
        .Arg<F64Buffer2>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer2>()
        .Arg<F64Buffer2>()
        .Arg<F64Buffer3>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<F64Buffer1>()
        .Arg<F64Buffer1>()
        .Arg<F64Buffer2>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer2>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer2>()
        .Arg<S32Buffer2>()
        .Arg<S32Buffer1>()
        .Ret<F64Buffer2>()
        .Ret<F64Buffer2>();
}

#ifdef TD_GRADDFT_ENABLE_GPU4PYSCF_RYS
auto CudaGpu4PyScfRysDirectJkBinding() {
    return ffi::Ffi::Bind()
        .Ctx<StreamCtx>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer2>()
        .Arg<S32Buffer2>()
        .Arg<F64Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<F32Buffer2>()
        .Arg<F32Buffer2>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<F32Buffer1>()
        .Ret<F64Buffer2>()
        .Ret<F64Buffer2>();
}
#endif

auto CudaPairSchwarzBinding() {
    return ffi::Ffi::Bind()
        .Ctx<StreamCtx>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer2>()
        .Arg<F64Buffer2>()
        .Arg<F64Buffer3>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Ret<F64Buffer1>();
}

auto CudaEriTensorBinding() {
    return ffi::Ffi::Bind()
        .Ctx<StreamCtx>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer2>()
        .Arg<F64Buffer2>()
        .Arg<F64Buffer3>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Ret<F64Buffer4>();
}

auto CudaEriPairMatrixBinding() {
    return ffi::Ffi::Bind()
        .Ctx<StreamCtx>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer2>()
        .Arg<F64Buffer2>()
        .Arg<F64Buffer3>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Arg<F64Buffer1>()
        .Arg<F64Buffer1>()
        .Ret<F64Buffer2>();
}

auto CudaPairMatrixJkBinding() {
    return ffi::Ffi::Bind()
        .Ctx<StreamCtx>()
        .Arg<F64Buffer2>()
        .Arg<F64Buffer2>()
        .Arg<S32Buffer1>()
        .Arg<S32Buffer1>()
        .Ret<F64Buffer2>()
        .Ret<F64Buffer2>();
}

}  // namespace

extern "C" __attribute__((weak)) cudaError_t TdGraddftLaunchJoltQC1qnt(
    cudaStream_t stream,
    int nao,
    const double* basis_data,
    double* dm,
    double* vj,
    double* vk,
    const int* group_keys,
    int n_groups,
    const int* group_quartet_keys,
    const int* group_quartet_offsets,
    const int* shell_quartets,
    int n_shell_quartets,
    int n_group_quartets
) {
    return cudaErrorNotSupported;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TdGraddftCudaDirectJkFfi,
    CudaDirectJkDispatch,
    CudaDirectJkBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TdGraddftCudaJoltQCDirectJkFfi,
    CudaJoltQCDirectJkDispatch,
    CudaJoltQCDirectJkBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TdGraddftCudaScreenedDirectJkFfi,
    CudaScreenedDirectJkDispatch,
    CudaScreenedDirectJkBinding());

#ifdef TD_GRADDFT_ENABLE_GPU4PYSCF_RYS
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TdGraddftCudaGpu4PyScfRysDirectJkFfi,
    CudaGpu4PyScfRysDirectJkDispatch,
    CudaGpu4PyScfRysDirectJkBinding());
#endif

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TdGraddftCudaPairSchwarzFfi,
    CudaPairSchwarzDispatch,
    CudaPairSchwarzBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TdGraddftCudaEriTensorFfi,
    CudaEriTensorDispatch,
    CudaEriTensorBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TdGraddftCudaEriPairMatrixFfi,
    CudaEriPairMatrixDispatch,
    CudaEriPairMatrixBinding());

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TdGraddftCudaPairMatrixJkFfi,
    CudaPairMatrixJkDispatch,
    CudaPairMatrixJkBinding());
