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


function BG(t::Real, tv::Real, delta::Real, N::Int)
    # 构造 h0 矩阵 (4x4)
    h0 = ComplexF64[
        delta    t        0       0;
        t        delta    tv      0;
        0        tv      -delta   t;
        0        0        t      -delta
    ]

    # 构造 T1 矩阵 (4x4)
    T1 = zeros(ComplexF64, 4, 4)
    T1[2, 1] = t  # Python [1,0] → Julia [2,1]
    T1[4, 3] = t  # Python [3,2] → Julia [4,3]

    # 构造 T2 矩阵 (4x4)
    T2 = zeros(ComplexF64, 4, 4)
    T2[1, 2] = t  # Python [0,1] → Julia [1,2]
    T2[3, 4] = t  # Python [2,3] → Julia [3,4]

    # 构造 H0 块
    H0_1 = kron(diagm(0 => ones(ComplexF64, N)), h0)
    H0_2 = kron(diagm(1 => ones(ComplexF64, N-1)), T2)
    H0_3 = kron(diagm(-1 => ones(ComplexF64, N-1)), adjoint(T2))  # T2.T.conj()
    H0 = H0_1 + H0_2 + H0_3

    # 构造 H1 块 (4N×4N)
    H1 = kron(diagm(0 => ones(ComplexF64, N)), T1)

    # 构造主哈密顿量 H (4N²×4N²)
    H_1 = kron(diagm(0 => ones(ComplexF64, N)), H0)
    H_2 = kron(diagm(1 => ones(ComplexF64, N-1)), H1)
    H_3 = kron(diagm(-1 => ones(ComplexF64, N-1)), adjoint(H1))  # H1.T.conj()
    H = H_1 + H_2 + H_3

    # 构造 Ty0 (4N×4N)
    Ty0 = zeros(ComplexF64, 4N, 4N)
    Ty0[(4N-3):4N, 1:4] .= T2  # Python [4N-4:4N, :4] → Julia [4N-3:4N, 1:4]

    # 构造 Ty (4N²×4N²)
    Ty = kron(diagm(0 => ones(ComplexF64, N)), Ty0)

    # 构造 Tx (4N²×4N²)
    Tx = zeros(ComplexF64, 4N^2, 4N^2)
    Tx[(4N^2 - 4N + 1):4N^2, 1:4N] .= H1  # Python [-4N:, :4N] → Julia [end-4N+1:end, 1:4N]

    return H, Tx, Ty
end



function generate_H_sk(kvec::Vector{<:Real}, t::Real, tv::Real, delta::Real, 
                       N::Int, W::Real, random_vector::Vector{<:Real})
    # 获取 H, Tx, Ty 矩阵
    H, Tx, Ty = BG(t, tv, delta, N)
    
    # 提取 k 分量 (Julia 索引从1开始)
    kx = kvec[1]
    ky = kvec[2]
    
    # 计算平面波相位因子
    x_phase = sqrt(3)/2 * kx + 1.5 * ky
    x_plane_wave = exp(im * x_phase)
    x_plane_wave_f = exp(-im * x_phase)
    
    y_phase = sqrt(3)/2 * kx - 1.5 * ky
    y_plane_wave = exp(im * y_phase)
    y_plane_wave_f = exp(-im * y_phase)
    
    # 构造跳跃项 (注意 adjoint 同时转置和共轭)
    H_x_hop = x_plane_wave * Tx + x_plane_wave_f * adjoint(Tx)
    H_y_hop = y_plane_wave * Ty + y_plane_wave_f * adjoint(Ty)
    
    # 构造无序项 (利用 Diagonal 创建对角矩阵)
    disorder = W * Diagonal(ComplexF64.(random_vector))
    
    # 合并哈密顿量
    H_total = H + disorder + H_x_hop + H_y_hop
    
    return H_total
end
