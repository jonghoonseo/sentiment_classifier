load("@rules_python//python:defs.bzl", "py_binary")
load("@third_party_py_requirements//:requirements.bzl", "requirement")

py_binary(
    name = "trainer",
    srcs = ["trainer.py"],
    deps = [
        "//dataset:shakespeare",
        "//models:rnn",
        requirement("tensorflow"),
    ],
)
