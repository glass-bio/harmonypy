// Exercise the production ridge method with Armadillo allocation tracking.
// Including its implementation here keeps the allocator hooks consistent
// across the test and backend without changing the installed extension.
#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <stdexcept>

static size_t largest_allocation = 0;

void* tracked_alloc(size_t bytes) {
    if (bytes > largest_allocation) largest_allocation = bytes;
    return std::malloc(bytes);
}

void tracked_free(void* pointer) { std::free(pointer); }

#define ARMA_ALIEN_MEM_ALLOC_FUNCTION tracked_alloc
#define ARMA_ALIEN_MEM_FREE_FUNCTION tracked_free
#include "../src/harmony.cpp"

void check_ridge_memory(int levels, bool prune, bool empty = false) {
    constexpr int n = 4096;
    constexpr int d = 32;
    arma::mat data(d, n);
    arma::Mat<int64_t> ids(2, n);
    for (int j = 0; j < n; ++j) {
        int a = j % levels;
        // Mostly nested covariates with occasional cross-group intersections.
        int b = ((j / levels) % 8 == 0) ? (a + levels / 2) % levels : a;
        ids(0, j) = a;
        ids(1, j) = b + levels;
        for (int pc = 0; pc < d; ++pc)
            data(pc, j) = 2 + 0.1 * a + 0.2 * b + std::sin(0.01 * j + pc);
    }
    arma::vec proportions(2 * levels, arma::fill::ones);
    proportions /= levels;
    arma::vec sigma(2, arma::fill::ones);
    sigma *= 0.3;
    arma::vec theta(2 * levels, arma::fill::zeros);
    arma::vec lambda(2 * levels + 1, arma::fill::ones);
    lambda(0) = 0;
    harmony::Harmony model(data, ids, proportions, sigma, theta, lambda,
        0.2, 0, 0, 1e-3, 1e-2, 2, 0.05, {levels, levels}, empty ? 1.0 : (prune ? 0.75 : 0.0),
        false, 42);

    // Freeze nonuniform assignments. For either denominator, cluster zero
    // retains two levels of each covariate, with a union larger than one level.
    model.O.zeros();
    for (int j = 0; j < n; ++j) {
        model.R(0, j) = empty ? 0.5f : (ids(0, j) < 2 ? 0.9f : 0.1f);
        model.R(1, j) = 1.0f - model.R(0, j);
        for (int c = 0; c < 2; ++c)
            model.O.col(ids(c, j)) += model.R.col(j);
    }
    model.E = arma::sum(model.R, 1) * model.Pr_b.t();
    std::vector<bool> retained(2 * levels);
    unsigned retained_levels = 0;
    for (int b = 0; b < 2 * levels; ++b) {
        retained[b] = model.O(0, b) / model.batch_sizes(b) > model.batch_proportion_cutoff;
        retained_levels += retained[b];
    }
    unsigned retained_cells = 0;
    for (int j = 0; j < n; ++j)
        retained_cells += retained[ids(0, j)] || retained[ids(1, j)];
    const unsigned expected_levels = empty ? 0 : (prune ? 4 : 2 * levels);
    const unsigned expected_cells = empty ? 0 : (prune ? n * 9 / (4 * levels) : n);
    if (retained_levels != expected_levels || retained_cells != expected_cells)
        throw std::runtime_error("allocation fixture did not exercise the intended retained set");
    const harmony::MATTYPE assignments = model.R;
    largest_allocation = 0;
    model.moe_correct_ridge();
    const size_t observed = largest_allocation;

    // Existing per-level gathers may copy N/levels columns. Neither the intercept
    // nor another regression should allocate the whole retained coordinate set.
    const size_t maximum_level_copy = size_t(d) * (n / levels) * sizeof(float);
    std::cout << (empty ? "empty" : (prune ? "pruned" : "unpruned"))
              << " levels=" << levels
              << " retained_cells=" << retained_cells << '/' << n
              << " largest_allocation=" << observed
              << " allowed=" << maximum_level_copy << '\n';
    if (observed > maximum_level_copy)
        throw std::runtime_error("ridge allocated more than one level's coordinates");
    const bool unchanged = arma::approx_equal(model.Z_corr, model.Z_orig, "absdiff", 0);
    if (!model.Z_corr.is_finite() || unchanged != empty)
        throw std::runtime_error("ridge correction or empty-set no-op failed");
    if (!arma::approx_equal(model.R, assignments, "absdiff", 0))
        throw std::runtime_error("ridge changed the original assignments");
}

int main() {
    try {
        check_ridge_memory(4, false);
        check_ridge_memory(4, true);
        check_ridge_memory(16, true);
        check_ridge_memory(16, true, true);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
