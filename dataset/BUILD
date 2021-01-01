load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@third_party_py_requirements//:requirements.bzl", "requirement")

py_library(
    name = "shakespeare",
    srcs = ["shakespeare.py"],
    deps = [
        requirement("numpy"),
        requirement("tensorflow"),
    ],
)

py_test(
    name = "shakespeare_test",
    srcs = ["shakespeare_test.py"],
    deps = [
        ":shakespeare",
    ],
)
