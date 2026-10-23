"""GEMM + Reduce-Scatter; compatible declarations from the package provider."""
from etx.frontends import hip_link
from etx.ir import Graph, Resource
from etx.operators.gemm_rs import T, BM, BN, build, bindings, runtime
