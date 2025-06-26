using LinearAlgebra
using Pkg
using MPI
BLAS.set_num_threads(1)
# @everywhere using LinearAlgebra
using Random
using Dates
# using Arpack, SparseArrays
using Base.Threads
# using ChunkSplitters
# using CairoMakie
# Pkg.add("Plots")
# using Plots
using JLD
using HDF5
using Printf


# Fermi-Dirac distribution
function DF_dis(E::AbstractArray, Ef::Real, T::Real)
    return @. 1 / (exp((E - Ef) / T) + 1)
end


# 洛伦兹型 δ 函数近似（实部，支持数组）
function zero_delta(E::AbstractArray, Ef::Real, gamma::Real)
    return @. -(1 / π) * (gamma / ((E - Ef)^2 + gamma^2))
end

# 洛伦兹型 δ 函数（复数形式，支持数组）
function zero_delta_realimag(E::Real, Ef::AbstractArray, gamma::Real)
    return @. (1 / π) * 1 / (E - Ef + im * gamma)
end

# 零温费米分布（阶跃函数，支持数组）
function zero_fd(E::AbstractArray, Ef::Real)
    return Int.(E .< Ef)  # 元素级布尔转整数
end