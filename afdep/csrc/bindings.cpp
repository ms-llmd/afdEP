// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFDEP project
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include "buffer.hpp"
#include "nccl_runtime.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "afdEP native directional Attention/FFN communication backend";
    module.def("get_nccl_unique_id", &afdep::get_nccl_unique_id);
    module.def("create_nccl_comm", &afdep::create_nccl_comm);
    module.def("destroy_nccl_comm", &afdep::destroy_nccl_comm);
    module.def("get_physical_domain_size", &afdep::get_physical_domain_size);
    module.def("get_nccl_version", &afdep::get_nccl_version);
    module.attr("compiled_nccl_version") = NCCL_VERSION_CODE;
    module.attr("protocol_version") = 1;
    afdep::register_apis(module);
}
