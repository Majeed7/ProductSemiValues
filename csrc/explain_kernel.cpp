#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace py = pybind11;

struct PreparedTreeData {
    py::array_t<int32_t, py::array::c_style> feature_ids; // (n_leaves, D)
    py::array_t<double, py::array::c_style> lower;        // (n_leaves, D)
    py::array_t<double, py::array::c_style> upper;        // (n_leaves, D)
    py::array_t<double, py::array::c_style> invw;         // (n_leaves, D)
    py::array_t<double, py::array::c_style> alpha;        // (n_leaves, n_outputs)
    py::array_t<double, py::array::c_style> quad_x;       // (m_q,)
    py::array_t<double, py::array::c_style> quad_log_w;   // (m_q,)
    int n_leaves;
    int max_d;
};

void explain_trees(
    py::array_t<double, py::array::c_style> X,     // (n_samples, n_features)
    py::array_t<double, py::array::c_style> out,   // (n_samples, n_features, n_outputs)
    std::vector<PreparedTreeData>& trees,
    int n_trees)
{
    // Access raw pointers for maximum performance — avoid pybind11 accessor overhead
    const double* X_ptr = X.data();
    double* out_ptr = out.mutable_data();

    const ssize_t n_samples = X.shape(0);
    const ssize_t n_features = X.shape(1);
    const ssize_t n_outputs = out.shape(2);
    const ssize_t out_stride_s = n_features * n_outputs;
    const ssize_t out_stride_f = n_outputs;

    for (int t = 0; t < n_trees; t++) {
        const auto& tree = trees[t];
        const int D = tree.max_d;
        if (D == 0) continue;

        const int32_t* feat_ptr = tree.feature_ids.data();
        const double* lower_ptr = tree.lower.data();
        const double* upper_ptr = tree.upper.data();
        const double* invw_ptr = tree.invw.data();
        const double* alpha_ptr = tree.alpha.data();
        const double* qx_ptr = tree.quad_x.data();
        const double* qlw_ptr = tree.quad_log_w.data();

        const int n_leaves = tree.n_leaves;
        const int m_q = static_cast<int>(tree.quad_x.shape(0));

        // Pre-compute actual_d for each leaf (avoid repeated scanning)
        std::vector<int> leaf_actual_d(n_leaves);
        for (int leaf = 0; leaf < n_leaves; leaf++) {
            const int32_t* frow = feat_ptr + leaf * D;
            int ad = 0;
            for (int j = 0; j < D; j++) {
                if (frow[j] < 0) break;
                ad++;
            }
            leaf_actual_d[leaf] = ad;
        }

        // Reusable scratch buffers (allocated once per tree, sized for max D)
        std::vector<double> K(D);
        std::vector<int> feat(D);
        std::vector<double> Phi(D);
        std::vector<double> log_B(D);

        for (ssize_t s = 0; s < n_samples; s++) {
            const double* x_row = X_ptr + s * n_features;
            double* out_s = out_ptr + s * out_stride_s;

            for (int leaf = 0; leaf < n_leaves; leaf++) {
                const int actual_d = leaf_actual_d[leaf];
                if (actual_d == 0) continue;

                const int leaf_off = leaf * D;
                const int32_t* f_row = feat_ptr + leaf_off;
                const double* lo_row = lower_ptr + leaf_off;
                const double* hi_row = upper_ptr + leaf_off;
                const double* iw_row = invw_ptr + leaf_off;
                const double* al_row = alpha_ptr + leaf * n_outputs;

                // Compute K for each feature
                for (int j = 0; j < actual_d; j++) {
                    int fid = f_row[j];
                    feat[j] = fid;
                    double x_val = x_row[fid];
                    double q = (x_val > lo_row[j] && x_val <= hi_row[j]) ? iw_row[j] : 0.0;
                    K[j] = q - 1.0;
                }

                // Zero Phi
                std::memset(Phi.data(), 0, actual_d * sizeof(double));

                // Quadrature loop
                for (int qi = 0; qi < m_q; qi++) {
                    double t_val = qx_ptr[qi];
                    double log_w = qlw_ptr[qi];

                    double total_log = 0.0;
                    for (int j = 0; j < actual_d; j++) {
                        double lb = std::log1p(t_val * K[j]);
                        log_B[j] = lb;
                        total_log += lb;
                    }

                    double base = log_w + total_log;
                    for (int j = 0; j < actual_d; j++) {
                        Phi[j] += std::exp(base - log_B[j]);
                    }
                }

                // Accumulate Phi * K * alpha into output
                if (n_outputs == 1) {
                    // Fast path for single-output models
                    double a0 = al_row[0];
                    for (int j = 0; j < actual_d; j++) {
                        out_s[feat[j] * out_stride_f] += Phi[j] * K[j] * a0;
                    }
                } else {
                    for (int j = 0; j < actual_d; j++) {
                        double pk = Phi[j] * K[j];
                        double* dst = out_s + feat[j] * out_stride_f;
                        for (ssize_t o = 0; o < n_outputs; o++) {
                            dst[o] += pk * al_row[o];
                        }
                    }
                }
            }
        }
    }
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ acceleration for pgshapley TreeSHAP";

    py::class_<PreparedTreeData>(m, "PreparedTreeData")
        .def(py::init<>())
        .def_readwrite("feature_ids", &PreparedTreeData::feature_ids)
        .def_readwrite("lower", &PreparedTreeData::lower)
        .def_readwrite("upper", &PreparedTreeData::upper)
        .def_readwrite("invw", &PreparedTreeData::invw)
        .def_readwrite("alpha", &PreparedTreeData::alpha)
        .def_readwrite("quad_x", &PreparedTreeData::quad_x)
        .def_readwrite("quad_log_w", &PreparedTreeData::quad_log_w)
        .def_readwrite("n_leaves", &PreparedTreeData::n_leaves)
        .def_readwrite("max_d", &PreparedTreeData::max_d);

    m.def("explain_trees", &explain_trees,
          "Compute SHAP values for all trees (C++ hot loop)",
          py::arg("X"), py::arg("out"), py::arg("trees"), py::arg("n_trees"));
}
