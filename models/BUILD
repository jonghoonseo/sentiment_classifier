load("@rules_python//python:defs.bzl", "py_library")
load("@third_party_py_requirements//:requirements.bzl", "requirement")

py_library(
    name = "rnn",
    srcs = ["rnn.py"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("tensorflow"),
    ],
)
