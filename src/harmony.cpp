// harmonypy - C++ backend matching R harmony2 package.
// Copyright (C) 2018  Ilya Korsunsky
//               2019  Kamil Slowikowski <kslowikowski@gmail.com>
//
// Uses custom scatter/gather kernels on a batch_id vector instead of
// sparse Phi matrices. BLAS threading (Accelerate/OpenBLAS) is controlled
// by the ncores parameter via environment variables in the Python layer.

#include "harmony.hpp"
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace harmony {

bool objective_converged(float obj_old, float obj_new, float epsilon) {
    if (!std::isfinite(obj_old) || !std::isfinite(obj_new) || !std::isfinite(epsilon))
        return false;
    if (obj_old == 0.0f) return obj_new == 0.0f;

    float delta = (obj_old - obj_new) / std::abs(obj_old);
    return delta >= 0.0f && delta < epsilon;
}

MATTYPE assignment_logits(
    const MATTYPE& distances,
    const VECTYPE& sigma,
    const MATTYPE& E,
    const MATTYPE& O,
    const VECTYPE& theta,
    const arma::Mat<arma::uword>& batch_ids
) {
    MATTYPE logits = -distances;
    logits.each_col() /= sigma;

    MATTYPE log_diversity = arma::log((2 * E) + 1) - arma::log(O + E + 1);
    log_diversity.each_row() %= theta.t();
    for (arma::uword c = 0; c < batch_ids.n_rows; ++c) {
        for (arma::uword j = 0; j < batch_ids.n_cols; ++j) {
            logits.col(j) += log_diversity.col(batch_ids(c, j));
        }
    }
    return logits;
}

ROWTYPE exponentiate_shifted_logits(MATTYPE& logits) {
    logits.each_row() -= arma::max(logits, 0);
    logits = arma::exp(logits);
    return arma::sum(logits, 0);
}

[[noreturn]] void Harmony::numerical_error(const char* stage, const char* invariant) const {
    std::ostringstream oss;
    oss << "Harmony numerical error during " << stage << ": " << invariant
        << " (sigma=[";
    for (arma::uword i = 0; i < sigma.n_elem; ++i) {
        if (i > 0) oss << ", ";
        oss << sigma(i);
    }
    oss << "], theta=[";
    for (arma::uword i = 0; i < theta.n_elem; ++i) {
        if (i > 0) oss << ", ";
        oss << theta(i);
    }
    oss << "], block_size=" << block_size << ", K=" << K << ", N=" << N << ")";
    throw std::runtime_error(oss.str());
}

void Harmony::check_assignment_normalizers(const ROWTYPE& normalizers, const char* stage) const {
    if (!normalizers.is_finite() || normalizers.min() <= 0.0f)
        numerical_error(stage, "assignment normalizers must be finite and positive");
}

void Harmony::normalize_log_assignments(MATTYPE& logits, const char* stage) const {
    ROWTYPE normalizers = exponentiate_shifted_logits(logits);
    check_assignment_normalizers(normalizers, stage);
    logits.each_row() /= normalizers;
}

void Harmony::check_state(const char* stage) const {
    if (!R.is_finite()) numerical_error(stage, "assignments must be finite");
    if (R.min() < 0.0f) numerical_error(stage, "assignments must be nonnegative");

    ROWTYPE assignment_sums = arma::sum(R, 0);
    if (!assignment_sums.is_finite() || arma::abs(assignment_sums - 1.0f).max() > 1e-4f)
        numerical_error(stage, "assignment columns must sum to one");

    auto all_finite = [](const std::vector<float>& values) {
        return std::all_of(values.begin(), values.end(), [](float value) {
            return std::isfinite(value);
        });
    };
    if (!all_finite(objective_harmony) || !all_finite(objective_kmeans) ||
        !all_finite(objective_kmeans_dist) || !all_finite(objective_kmeans_entropy) ||
        !all_finite(objective_kmeans_cross))
        numerical_error(stage, "objectives must be finite");

    if (!Z_corr.is_finite()) numerical_error(stage, "corrected coordinates must be finite");
}

// =========================================================================
// Custom kernels
// =========================================================================

// Scatter-add R columns into O for each covariate's batch assignment.
// ids is n_cov x n_cells (matching batch_ids layout).
void Harmony::scatter_add_O(const MATTYPE& Rsub, const arma::Mat<arma::uword>& ids, float sign) {
    const int n = Rsub.n_cols;
    const int k = Rsub.n_rows;
    const int nc = ids.n_rows;
    for (int c = 0; c < nc; ++c) {
        for (int j = 0; j < n; ++j) {
            unsigned b = ids(c, j);
            const float* col = Rsub.colptr(j);
            float* dst = O.colptr(b);
            if (sign > 0) {
                for (int i = 0; i < k; ++i) dst[i] += col[i];
            } else {
                for (int i = 0; i < k; ++i) dst[i] -= col[i];
            }
        }
    }
}

// =========================================================================
// K-means initialization (matches R harmony2)
// =========================================================================

MATTYPE kmeans_init(const MATTYPE& X, int K, std::mt19937& rng) {
    int N = X.n_cols;

    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);
    MATTYPE Y(X.n_rows, K);
    for (int i = 0; i < K; ++i) {
        int idx = static_cast<int>(std::round(uniform01(rng) * N));
        if (idx >= N) idx = N - 1;
        Y.col(i) = X.col(idx);
    }

    std::set<unsigned> chosen;
    for (int i = 0; i < K; ++i) {
        VECTYPE distances = arma::abs((2.0f * (1.0f - Y.col(i).t() * X)).as_col());
        VECTYPE random_numbers(N, arma::fill::none);
        for (int j = 0; j < N; ++j) random_numbers(j) = uniform01(rng);
        VECTYPE prob = -arma::log(random_numbers) / (distances + 1e-10f);

        for (auto idx : chosen) prob(idx) = prob.max();
        unsigned best = prob.index_min();
        while (chosen.count(best)) {
            prob(best) = prob.max();
            best = prob.index_min();
        }
        chosen.insert(best);
        Y.col(i) = X.col(best);
    }

    for (int i = 0; i < 10; ++i) {
        arma::kmeans(Y, X, K, arma::keep_existing, 1, false);
    }

    return Y;
}

// =========================================================================
// Constructor
// =========================================================================

Harmony::Harmony(
    const arma::mat& Z,
    const arma::Mat<int64_t>& batch_of_cell,
    const arma::vec& Pr_b_in,
    const arma::vec& sigma_in,
    const arma::vec& theta_in,
    const arma::vec& lambda_in,
    double alpha_in,
    int max_iter_harmony,
    int max_iter_kmeans,
    double epsilon_kmeans,
    double epsilon_harmony,
    int K,
    double block_size,
    const std::vector<int>& B_vec_in,
    double batch_proportion_cutoff,
    bool verbose,
    int random_state,
    std::function<void(const std::string&)> log_fn_in
) : max_iter_harmony(max_iter_harmony),
    max_iter_kmeans(max_iter_kmeans),
    epsilon_kmeans(static_cast<float>(epsilon_kmeans)),
    epsilon_harmony(static_cast<float>(epsilon_harmony)),
    K(K),
    block_size(static_cast<float>(block_size)),
    verbose(verbose),
    window_size(3),
    alpha(alpha_in),
    batch_proportion_cutoff(static_cast<float>(batch_proportion_cutoff)),
    B_vec(B_vec_in),
    log_fn(std::move(log_fn_in)),
    rng(random_state)
{
    Z_orig = arma::conv_to<MATTYPE>::from(Z);
    Z_corr = arma::normalise(Z_orig, 2, 0);

    Pr_b = arma::conv_to<VECTYPE>::from(Pr_b_in);
    N = Z.n_cols;
    d = Z.n_rows;
    B = 0;
    for (auto v : B_vec) B += v;

    sigma = arma::conv_to<VECTYPE>::from(sigma_in);
    theta = arma::conv_to<VECTYPE>::from(theta_in);

    if (lambda_in(0) < 0) {
        lambda_estimation = true;
        lambda.zeros(B + 1);
    } else {
        lambda_estimation = false;
        lambda = lambda_in;
    }

    if (B_vec.size() > 1) {
        covariate_bounds.resize(B_vec.size());
        std::partial_sum(B_vec.begin(), B_vec.end(), covariate_bounds.begin());
    } else {
        covariate_bounds.push_back(B_vec.front());
    }

    build_batch_structures(batch_of_cell);
    allocate_buffers();

    if (verbose && log_fn) log_fn("Computing initial centroids...");
    init_cluster();
    check_state("initialization");
    if (verbose && log_fn) log_fn("Initialization complete.");
    harmonize(max_iter_harmony, verbose);
    check_state("return");
}

void Harmony::build_batch_structures(const arma::Mat<int64_t>& batch_of_cell) {
    // batch_of_cell is n_cov x N (int64). Each row c contains the batch
    // index for covariate c, with values in [offset_c, offset_c + n_levels_c).
    n_covariates = batch_of_cell.n_rows;
    batch_ids.set_size(n_covariates, N);
    for (int c = 0; c < n_covariates; ++c) {
        for (int j = 0; j < N; ++j) {
            batch_ids(c, j) = static_cast<arma::uword>(batch_of_cell(c, j));
        }
    }

    // Compute batch sizes (count cells per batch across all covariates)
    batch_sizes.zeros(B);
    for (int c = 0; c < n_covariates; ++c) {
        for (int j = 0; j < N; ++j) {
            batch_sizes(batch_ids(c, j)) += 1.0f;
        }
    }
    // Each cell is counted n_covariates times; normalize
    batch_sizes /= static_cast<float>(n_covariates);

    // Build per-batch cell index lists (using first covariate for indexing)
    // For ridge correction, batch_index[b] lists cells belonging to batch b.
    // With multiple covariates, a cell's "primary" batch is determined by
    // which covariate each batch belongs to (tracked via covariate_bounds).
    batch_index.resize(B);

    // Determine which covariate owns each batch
    std::vector<int> cov_of_batch(B);
    int idx = 0;
    for (size_t c = 0; c < B_vec.size(); ++c) {
        for (int i = 0; i < B_vec[c]; ++i) {
            cov_of_batch[idx++] = c;
        }
    }

    // Count cells per batch
    std::vector<unsigned> counts(B, 0);
    for (int b = 0; b < B; ++b) {
        int c = cov_of_batch[b];
        for (int j = 0; j < N; ++j) {
            if (batch_ids(c, j) == static_cast<arma::uword>(b)) counts[b]++;
        }
    }
    for (int b = 0; b < B; ++b) {
        batch_index[b].set_size(counts[b]);
    }

    // Fill batch_index
    std::vector<unsigned> counters(B, 0);
    for (int b = 0; b < B; ++b) {
        int c = cov_of_batch[b];
        for (int j = 0; j < N; ++j) {
            if (batch_ids(c, j) == static_cast<arma::uword>(b)) {
                batch_index[b](counters[b]++) = j;
            }
        }
    }
}

void Harmony::allocate_buffers() {
    dist_mat.zeros(K, N);
    O.zeros(K, B);
    E.zeros(K, B);
    W.zeros(B + 1, d);
    R.zeros(K, N);
    Y.zeros(d, K);
}

// =========================================================================
// init_cluster
// =========================================================================

void Harmony::init_cluster() {
    Y = kmeans_init(Z_corr, K, rng);
    Y = arma::normalise(Y, 2, 0);

    dist_mat = 2.0f * (1.0f - Y.t() * Z_corr);

    R = -dist_mat;
    R.each_col() /= sigma;
    normalize_log_assignments(R, "initialization");

    E = arma::sum(R, 1) * Pr_b.t();
    O.zeros();
    scatter_add_O(R, batch_ids, 1.0f);

    compute_objective();
    objective_harmony.push_back(objective_kmeans.back());
}

// =========================================================================
// compute_objective
// =========================================================================

void Harmony::compute_objective() {
    const float norm_const = 2000.0f / static_cast<float>(N);

    float kmeans_error = arma::accu(R % dist_mat);

    MATTYPE log_R = R;
    log_R.transform([](float val) { return val > 0 ? val * std::log(val) : 0.0f; });
    float _entropy = arma::as_scalar(arma::accu(log_R.each_col() % sigma));

    MATTYPE ratio = (O + E + 1) / (2 * E + 1);
    ratio.transform([](float val) { return std::log(val); });
    ratio.each_row() %= theta.t();
    ratio.each_col() %= sigma;
    ratio %= O;
    float _cross_entropy = arma::accu(ratio);

    objective_kmeans.push_back((kmeans_error + _entropy + _cross_entropy) * norm_const);
    objective_kmeans_dist.push_back(kmeans_error * norm_const);
    objective_kmeans_entropy.push_back(_entropy * norm_const);
    objective_kmeans_cross.push_back(_cross_entropy * norm_const);
}

// =========================================================================
// harmonize / cluster
// =========================================================================

void Harmony::harmonize(int iter_harmony, bool verbose_flag) {
    bool converged = false;
    for (int i = 1; i <= iter_harmony; ++i) {
        if (verbose_flag && log_fn) {
            std::ostringstream oss;
            oss << "Iteration " << i << " of " << iter_harmony;
            log_fn(oss.str());
        }

        cluster();
        moe_correct_ridge();

        converged = check_convergence(1);
        if (converged) {
            if (verbose_flag && log_fn) {
                std::ostringstream oss;
                oss << "Converged after " << i << " iteration"
                    << (i > 1 ? "s" : "");
                log_fn(oss.str());
            }
            break;
        }
    }
    if (verbose_flag && !converged && log_fn)
        log_fn("Stopped before convergence");
}

void Harmony::cluster() {
    if (objective_harmony.size() > 1) {
        Z_corr = arma::normalise(Z_corr, 2, 0);
        dist_mat = 2.0f * (1.0f - Y.t() * Z_corr);
        R = -dist_mat;
        R.each_col() /= sigma;
        normalize_log_assignments(R, "cluster initialization");
        E = arma::sum(R, 1) * Pr_b.t();
        O.zeros();
        scatter_add_O(R, batch_ids, 1.0f);
    }

    int rounds = 0;
    for (int i = 0; i < max_iter_kmeans; ++i) {
        update_R();
        compute_objective();
        check_state("assignment update");

        if (i > window_size) {
            if (check_convergence(0)) {
                rounds = i + 1;
                break;
            }
        }
        rounds = i + 1;
    }

    kmeans_rounds.push_back(rounds);
    objective_harmony.push_back(objective_kmeans.back());
}

// =========================================================================
// update_R
// =========================================================================

void Harmony::update_R() {
    std::vector<unsigned> indices_vec(N);
    std::iota(indices_vec.begin(), indices_vec.end(), 0);
    std::shuffle(indices_vec.begin(), indices_vec.end(), rng);
    arma::uvec update_order(N);
    for (int i = 0; i < N; ++i) update_order(i) = indices_vec[i];

    arma::uvec indices = arma::linspace<arma::uvec>(0, N - 1, N);
    arma::uvec reverse_index(N, arma::fill::zeros);
    reverse_index.rows(update_order) = indices;

    unsigned n_blocks = static_cast<unsigned>(std::ceil(1.0 / block_size));
    unsigned cells_per_block = std::max(1u, static_cast<unsigned>(N * block_size));

    R = R.cols(update_order);
    dist_mat = dist_mat.cols(update_order);
    // Shuffle batch_ids (n_cov x N) by column order
    arma::Mat<arma::uword> batch_ids_shuf = batch_ids.cols(update_order);

    for (unsigned i = 0; i < n_blocks; ++i) {
        unsigned idx_min = i * cells_per_block;
        unsigned idx_max = ((i + 1) * cells_per_block) - 1;
        if (i == n_blocks - 1) idx_max = N - 1;
        if (idx_min >= static_cast<unsigned>(N)) break;
        auto Rcells = R.submat(0, idx_min, R.n_rows - 1, idx_max);
        auto dist_matcells = dist_mat.submat(0, idx_min, dist_mat.n_rows - 1, idx_max);
        arma::Mat<arma::uword> block_ids = batch_ids_shuf.cols(idx_min, idx_max);

        E -= arma::sum(Rcells, 1) * Pr_b.t();
        scatter_add_O(Rcells, block_ids, -1.0f);

        // E and O stay frozen while every cell in this block is reassigned.
        MATTYPE logits = assignment_logits(dist_matcells, sigma, E, O, theta, block_ids);
        normalize_log_assignments(logits, "assignment update");
        Rcells = logits;

        E += arma::sum(Rcells, 1) * Pr_b.t();
        scatter_add_O(Rcells, block_ids, 1.0f);
    }

    R = R.cols(reverse_index);
    dist_mat = dist_mat.cols(reverse_index);
}

// =========================================================================
// check_convergence
// =========================================================================

bool Harmony::check_convergence(int i_type) {
    if (i_type == 0) {
        if (objective_kmeans.size() <= static_cast<size_t>(window_size + 1))
            return false;

        float obj_old = 0.0f, obj_new = 0.0f;
        size_t n = objective_kmeans.size();
        for (int i = 0; i < window_size; ++i) {
            obj_old += objective_kmeans[n - 2 - i];
            obj_new += objective_kmeans[n - 1 - i];
        }
        return std::abs(obj_old - obj_new) / std::abs(obj_old) < epsilon_kmeans;
    }

    if (i_type == 1) {
        if (objective_harmony.size() < 2) return false;
        float obj_old = objective_harmony[objective_harmony.size() - 2];
        float obj_new = objective_harmony[objective_harmony.size() - 1];
        return objective_converged(obj_old, obj_new, epsilon_harmony);
    }
    return true;
}

// =========================================================================
// moe_correct_ridge
// =========================================================================

void Harmony::moe_correct_ridge() {
    Z_corr = Z_orig;
    double coordinate_error_bound = 0.0;
    struct RidgeFit {
        std::vector<unsigned> batch_row;
        arma::mat coefficients;
        int cluster;
    };
    std::vector<RidgeFit> fits;
    fits.reserve(K);
    arma::rowvec max_abs_input = arma::conv_to<arma::rowvec>::from(
        arma::max(arma::abs(Z_orig), 1).t());

    for (int k = 0; k < K; ++k) {
        VECTYPE avg_R = O.row(k).t() / batch_sizes;

        std::vector<unsigned> keep;
        std::vector<unsigned> cov_levels(B_vec.size(), 0);

        for (unsigned b = 0, current_cov = 0; b < static_cast<unsigned>(B); ++b) {
            if (current_cov < covariate_bounds.size() && !(b < covariate_bounds[current_cov]))
                current_cov++;
            if (arma::as_scalar(avg_R.row(b)) > batch_proportion_cutoff)
                cov_levels[current_cov]++;
        }

        unsigned active_covariates = 0;
        for (auto const& l : cov_levels) {
            if (l > 1) active_covariates++;
        }

        for (unsigned b = 0, current_cov = 0; b < static_cast<unsigned>(B); ++b) {
            if (current_cov < covariate_bounds.size() && !(b < covariate_bounds[current_cov]))
                current_cov++;
            if (arma::as_scalar(avg_R.row(b)) > batch_proportion_cutoff && cov_levels[current_cov] > 1)
                keep.push_back(b);
        }

        if (active_covariates == 0) continue;

        unsigned n_keep = keep.size();
        arma::vec lamb_vec(n_keep + 1, arma::fill::zeros);
        if (lambda_estimation) {
            double mass = 0.0, compensation = 0.0;
            for (int j = 0; j < N; ++j) {
                double corrected = static_cast<double>(R(k, j)) - compensation;
                double next = mass + corrected;
                compensation = (next - mass) - corrected;
                mass = next;
            }
            for (unsigned i = 0; i < n_keep; ++i)
                lamb_vec(i + 1) = alpha * mass * batch_index[keep[i]].n_elem / N;
        } else {
            for (unsigned i = 0; i < n_keep; ++i)
                lamb_vec(i + 1) = lambda(keep[i] + 1);
        }

        // Fit with the original float32 coordinates and assignments, but
        // accumulate and solve in float64. Forming an inverse in float32 loses
        // substantial direct-coordinate accuracy when the penalty is weak.
        std::vector<unsigned> batch_row(B, 0);
        for (unsigned i = 0; i < n_keep; ++i) batch_row[keep[i]] = i + 1;
        arma::mat gram(n_keep + 1, n_keep + 1, arma::fill::zeros);
        arma::mat rhs(n_keep + 1, d, arma::fill::zeros);
        arma::mat rhs_abs(rhs.n_rows, rhs.n_cols, arma::fill::zeros);
        auto compensated_add = [](double& sum, double& compensation, double value) {
            double corrected = value - compensation;
            double next = sum + corrected;
            compensation = (next - sum) - corrected;
            sum = next;
        };
        auto assemble = [&](bool compensated) {
            gram.zeros();
            rhs.zeros();
            rhs_abs.zeros();
            if (!compensated && n_covariates == 1) {
                // One covariate has disjoint groups. Use a double-precision
                // matrix-vector product for each group and sum those rows for
                // the intercept; this avoids a cell-by-cell coordinate loop.
                for (unsigned i = 0; i < n_keep; ++i) {
                    unsigned row = i + 1;
                    const arma::uvec& idx = batch_index[keep[i]];
                    arma::vec weights(idx.n_elem);
                    for (arma::uword pos = 0; pos < idx.n_elem; ++pos)
                        weights(pos) = R(k, idx(pos));
                    double mass = arma::accu(weights);
                    gram(0, 0) += mass;
                    gram(0, row) = mass;
                    gram(row, 0) = mass;
                    gram(row, row) = mass;
                    arma::mat coordinates = arma::conv_to<arma::mat>::from(Z_orig.cols(idx));
                    rhs.row(row) = (coordinates * weights).t();
                    rhs.row(0) += rhs.row(row);
                    rhs_abs.row(row) = mass * max_abs_input;
                    rhs_abs.row(0) += rhs_abs.row(row);
                }
                gram.diag() += lamb_vec;
                return;
            }
            arma::mat gram_comp, rhs_comp;
            if (compensated) {
                gram_comp.zeros(gram.n_rows, gram.n_cols);
                rhs_comp.zeros(rhs.n_rows, rhs.n_cols);
            }
            std::vector<unsigned> rows;
            rows.reserve(n_covariates + 1);
            for (int j = 0; j < N; ++j) {
                rows.clear();
                rows.push_back(0);
                for (int c = 0; c < n_covariates; ++c) {
                    unsigned row = batch_row[batch_ids(c, j)];
                    if (row != 0) rows.push_back(row);
                }
                if (rows.size() == 1) continue;
                const double weight = R(k, j);
                for (unsigned row : rows) {
                    for (unsigned col : rows) {
                        if (compensated)
                            compensated_add(gram(row, col), gram_comp(row, col), weight);
                        else
                            gram(row, col) += weight;
                    }
                    for (int p = 0; p < d; ++p) {
                        double term = weight * static_cast<double>(Z_orig(p, j));
                        if (compensated)
                            compensated_add(rhs(row, p), rhs_comp(row, p), term);
                        else
                            rhs(row, p) += term;
                        rhs_abs(row, p) += std::abs(term);
                    }
                }
            }
            gram.diag() += lamb_vec;
        };

        arma::mat coefficients;
        const double eps = std::numeric_limits<double>::epsilon();
        double fit_error = 0.0;
        auto assess = [&](double assembly_factor) {
            double reciprocal_condition = arma::rcond(gram);
            // At this condition, float64 roundoff can consume four digits.
            if (!std::isfinite(reciprocal_condition) || reciprocal_condition < 1e-12)
                return false;
            if (!arma::solve(coefficients, gram, rhs,
                             arma::solve_opts::likely_sympd + arma::solve_opts::no_approx)
                || !coefficients.is_finite())
                return false;
            arma::mat inverse;
            if (!arma::inv_sympd(inverse, gram) || !inverse.is_finite())
                return false;
            const double matrix_norm = arma::norm(gram, "inf");
            const double inverse_norm = arma::norm(inverse, "inf");
            // The factor of two below covers the change in the inverse only
            // while the assembled matrix perturbation is below one half.
            if (!std::isfinite(matrix_norm * inverse_norm)
                || assembly_factor * matrix_norm * inverse_norm >= 0.5)
                return false;
            const arma::mat residual = rhs - gram * coefficients;
            fit_error = 0.0;
            for (int p = 0; p < d; ++p) {
                double assembly_scale = matrix_norm * arma::abs(coefficients.col(p)).max()
                                      + rhs_abs.col(p).max();
                double column_error = 2.0 * n_covariates * inverse_norm
                    * (arma::abs(residual.col(p)).max() + assembly_factor * assembly_scale);
                fit_error = std::max(fit_error, column_error);
            }
            return std::isfinite(fit_error) && coordinate_error_bound + fit_error <= 1e-4;
        };

        // Ordinary float64 accumulation is faster. Rebuild with compensated
        // sums when its conservative N-term error estimate uses the budget.
        assemble(false);
        double gamma_n = N * eps / (1.0 - N * eps);
        if (!assess(8.0 * eps + 2.0 * gamma_n)) {
            assemble(true);
            if (!assess(8.0 * eps))
                numerical_error("ridge accuracy", "cannot support 1e-4 absolute coordinate accuracy");
        }
        coordinate_error_bound += fit_error;

        Y.col(k) = arma::conv_to<VECTYPE>::from(coefficients.row(0).t());
        coefficients.row(0).zeros();
        W = arma::conv_to<MATTYPE>::from(coefficients);
        fits.push_back({std::move(batch_row), std::move(coefficients), k});
    }

    // Each cell receives one final float32 rounding, regardless of K.
    double max_result = 0.0;
    double max_sum_error = 0.0;
    double terms = 1.0 + static_cast<double>(fits.size()) * n_covariates;
    double gamma = terms * std::numeric_limits<double>::epsilon();
    if (gamma >= 1.0)
        numerical_error("ridge accuracy", "too many correction terms to bound accuracy");
    gamma /= 1.0 - gamma;
    std::vector<double> values(d), absolute_sums(d);
    for (int j = 0; j < N; ++j) {
        for (int p = 0; p < d; ++p) {
            values[p] = Z_orig(p, j);
            absolute_sums[p] = std::abs(values[p]);
        }
        for (const RidgeFit& fit : fits) {
            double weight = R(fit.cluster, j);
            for (int c = 0; c < n_covariates; ++c) {
                unsigned row = fit.batch_row[batch_ids(c, j)];
                if (row == 0) continue;
                for (int p = 0; p < d; ++p) {
                    double term = weight * fit.coefficients(row, p);
                    values[p] -= term;
                    absolute_sums[p] += std::abs(term);
                }
            }
        }
        for (int p = 0; p < d; ++p) {
            max_sum_error = std::max(max_sum_error, gamma * absolute_sums[p]);
            max_result = std::max(max_result, std::abs(values[p]));
            Z_corr(p, j) = static_cast<float>(values[p]);
        }
    }
    coordinate_error_bound += max_sum_error
        + 0.5 * std::numeric_limits<float>::epsilon() * max_result;
    if (!std::isfinite(coordinate_error_bound) || coordinate_error_bound > 1e-4)
        numerical_error("ridge accuracy", "cannot support 1e-4 absolute coordinate accuracy");

    Y = arma::normalise(Y, 2, 0);
}

} // namespace harmony
