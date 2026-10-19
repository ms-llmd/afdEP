#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <deep_ep/common/compiled.cuh>

#include "elastic/buffer.hpp"
#include "jit/api.hpp"
#include "utils/event.hpp"

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

namespace py = pybind11;

bool is_sm90_compiled() {
#ifndef DISABLE_SM90_FEATURES
    return true;
#else
    return false;
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    // afdEP: fail clearly when another dependency loaded a different NCCL runtime.
    m.attr("compiled_nccl_version") = NCCL_VERSION_CODE;
    m.def("get_nccl_version", []() {
        int version = 0;
        const auto status = ncclGetVersion(&version);
        if (status != ncclSuccess)
            throw std::runtime_error(ncclGetErrorString(status));
        return version;
    });

    // Whether support FP8 and TMA features
    m.def("is_sm90_compiled", []() { return deep_ep::kEnableSM90Features; });

    // The integer type of top-k indices
    m.attr("topk_idx_t") = py::cast(c10::CppTypeToScalarType<deep_ep::topk_idx_t>::value);

    // ### PATCH START: module_local so this coexists with vLLM's deep_ep._C
    py::class_<deep_ep::EventHandle>(m, "EventHandle", py::module_local())
    // ### PATCH END: module_local so this coexists with vLLM's deep_ep._C
        .def(py::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    // JIT API
    deep_ep::jit::register_apis(m);

    // Register DeepEP V2 elastic buffer APIs.
    deep_ep::elastic::register_apis(m);
}
