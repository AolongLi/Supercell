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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~ some useful functions ~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Fermi-Dirac distribution
function DF_dis(E::AbstractArray{<:Real}, Ef::Real, T::Real)
    return @. 1 / (exp((E - Ef) / T) + 1)
end


# 洛伦兹型 δ 函数近似（实部，支持数组）
function zero_delta(E::AbstractArray{<:Real}, Ef::Real, gamma::Real)
    return @. -(1 / π) * (gamma / ((E - Ef)^2 + gamma^2))
end

# 洛伦兹型 δ 函数（复数形式，支持数组）
function zero_delta_realimag(E::Real, Ef::AbstractArray{<:Real}, gamma::Real)
    return @. (1 / π) * 1 / (E - Ef + im * gamma)
end

# 零温费米分布（阶跃函数，支持数组）
function zero_fd(E::AbstractArray{<:Real}, Ef::Real)
    return Int.(E .< Ef)  # 元素级布尔转整数
end
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~ Kvectors-mesh Construction ~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
function kmesh_generator(knum::Int; method::Symbol=:rhombus, delta::Real=1.0)
    if method === :rhombus
        # 方法1：基于基矢生成菱形网格
        # 计算基矢 v1, v2
        v1 = 2π .* [sqrt(3)/3, 1/3] ./ knum
        v2 = 2π .* [sqrt(3)/3, -1/3] ./ knum
        
        # 初始化 k 矢量数组 (knum^2 × 2)
        kvecs = zeros(Float64, knum^2, 2)
        
        # 遍历所有 (i,j) 组合生成 k 矢量
        for i in 0:knum-1
            for j in 0:knum-1
                idx = i * knum + j + 1  # Julia 数组索引从 1 开始
                kvecs[idx, :] = i * v1 + j * v2
            end
        end
        
        # 计算面积元素 (用于积分)
        ds = norm(v1)^2 * sin(π/3)    
        return kvecs, ds

    elseif method === :square
        # 方法2：在K点附近生成方形网格
        v1 = 2π .* [sqrt(3)/3, 1/3]
        v2 = 2π .* [sqrt(3)/3, -1/3]
        K = (v1 + v2) / 3

        qujian = range(-delta, delta, length=knum)
        dk = abs(qujian[2] - qujian[1])
        xK = K[1] .+ qujian
        yK = K[2] .+ qujian

        # 创建网格坐标
        kx_grid = repeat(xK', knum, 1)
        ky_grid = repeat(yK, 1, knum)

        kvecs = hcat(vec(kx_grid), vec(ky_grid))
        ds = dk^2
        return kvecs, ds
    else
        error("Unknown k-space generation method: ", method)
    end
end
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# function pxsbx_generator(knum::Int)
#    # 计算基矢 v1, v2
#    v1 = 2π .* [sqrt(3)/3, 1/3]
#    v2 = 2π .* [sqrt(3)/3, -1/3]
#    K = (v1 + v2)/3
#
#    delta=1

#    qujian=range(-delta,delta,length=knum)
#    dk=abs(qujian[2]-qujian[1])
#    xK = K[1] .+ qujian
#    yK = K[2] .+ qujian
#    # xK = qujian
#    # yK = qujian

#    kx = xK' .* ones(eltype(xK), knum)  # 行扩展
#    ky = ones(eltype(yK), knum)' .* yK   # 列扩展

#    kvecs = hcat(vec(permutedims(kx)), vec(permutedims(ky)))
#    ds=dk*dk
#    return kvecs, ds
#end
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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

# W,N,knum,geometry_method,t,tv,delta
function shiftxyy(kvec::Vector{<:Real},W::Real,N::Int,omegalist::AbstractVector{<:Real},
    ds::Real,Ef::Real,gamma::Real,t::Real,tv::Real,delta::Real,
    random_vector::Vector{<:Real},λ::Real)
    # println("shiftyyy start:",Dates.now())

    # k offsts
    dkx = [λ ; 0]
    dky = [0 ; λ]
    dkxdky = [λ ; λ]



    H_total = generate_H_sk(kvec, t, tv, delta, N, W, random_vector)
    eigen_result = eigen(Hermitian(H_total)) # 优化：显式声明厄米矩阵
    evalues = eigen_result.values
    Uk = eigen_result.vectors
    
    H_zx = generate_H_sk(kvec .+ dkx, t, tv, delta, N, W, random_vector)
    Uzx = eigen(Hermitian(H_zx)).vectors

    H_fx = generate_H_sk(kvec .- dkx, t, tv, delta, N, W, random_vector)
    Ufx = eigen(Hermitian(H_fx)).vectors

    H_zy = generate_H_sk(kvec .+ dky, t, tv, delta, N, W, random_vector)
    Uzy = eigen(Hermitian(H_zy)).vectors

    H_fy = generate_H_sk(kvec .- dky, t, tv, delta, N, W, random_vector)
    Ufy = eigen(Hermitian(H_fy)).vectors

    H_zxy = generate_H_sk(kvec .+ dkxdky, t, tv, delta, N, W, random_vector)
    Uzxy = eigen(Hermitian(H_zxy)).vectors

    H_fxy = generate_H_sk(kvec .- dkxdky, t, tv, delta, N, W, random_vector)
    Ufxy = eigen(Hermitian(H_fxy)).vectors

    # println("eigen end:",Dates.now())

    basis_size = size(evalues, 1)
    skewness_tensor = zeros(Float64, basis_size÷2, basis_size÷2)

    a1 = Uk' * Uzx
    a2 = Uk' * Ufx
    a3 = Uk' * Uzy
    a4 = Uk' * Ufy
    a5 = Uk' * Uzxy
    a6 = Uk' * Ufxy
    a7 = Uzx' * Uzxy
    a8 = Ufx' * Uzxy
    a9 = Ufx' * Uzx
    a10 = Uzx' * Uzy
    a11 = Ufx' * Uzy
    a12 = Uzx' * Ufy
    a13 = Ufx' * Ufy
    a14 = Ufx' * Ufxy
    a15 = Uzx' * Ufxy

    t1_1 = transpose(a1) .* a7 .- transpose(a2) .* a8 .* transpose(diag(a5'))
    t1_2 = transpose(a2) .* a9 .* transpose(diag(a1'))
    t1_3 = -transpose(a1) .* a10 + transpose(a2) .* a11 .* transpose(diag(a3'))
    t1_4 = 2 .* transpose(a1) .* a1' .- 2 .* transpose(a2) .* a2'
    t1_5 = - transpose(a1) .* a9'
    t1_6 = -transpose(a1) .* a12 .+ transpose(a2) .* a13 .* transpose(diag(a4'))
    t1_7 = transpose(a1) .* a15 .- transpose(a2) .* a14 .* transpose(diag(a6'))

    term1 = (1 / (4 * λ^3)) .* (t1_1 + t1_2 + t1_3 + t1_4 + t1_5 + t1_6 + t1_7)

    term2 = (1 / (8 * λ^3)) .* (transpose(a1) .* diag(a10) .* a10' .* transpose(diag(a1')) 
                                - transpose(a1) .* diag(a10) .* a11' .* transpose(diag(a2')) 
                                - transpose(a1) .* diag(a12) .* a12' .* transpose(diag(a1')) 
                                + transpose(a1) .* diag(a12) .* a13' .* transpose(diag(a2')) 
                                - transpose(a2) .* diag(a11) .* a10' .* transpose(diag(a1'))
                                + transpose(a2) .* diag(a11) .* a11' .* transpose(diag(a2'))
                                + transpose(a2) .* diag(a13) .* a12' .* transpose(diag(a1')) 
                                - transpose(a2) .* diag(a13) .* a13' .* transpose(diag(a2'))
                                )

    term=term1 .+term2
    term_img=imag(0.5*(term -transpose(term)))
    skewness_tensor+=term_imag[1:basis_size÷2,(basis_size÷2+1):basis_size]

    # println("skewness_tensor",skewness_tensor)
    ΔE=evalues[1:basis_size÷2] .- reshape(evalues[basis_size÷2+1:basis_size],1,basis_size÷2)

    num_w=size(omegalist,1)
    result_list=zeros(Float64,num_w)
    joint_list=zeros(Float64,num_w)
    # println("tensor end:",Dates.now())
    # Fermi分布函数 (需实现DF_dis函数)
    fermi = DF_dis(evalues, Ef, gamma)
    basissize = size(skewness_tensor, 2) * 2  # 获取张量第二维大小

    # 重塑费米分布矩阵
    fermi_row = reshape(fermi[1:div(basissize,2)], div(basissize,2), 1)
    fermi_col = reshape(fermi[div(basissize,2)+1:basissize], 1, div(basissize,2) )

    delta_fermi = fermi_row .- fermi_col

    for i=1:num_w
        omega=omegalist[i]
        
        delt_func_leftdown = -1 .* zero_delta_realimag(omega, -permutedims(ΔE, [2,1]), gamma)

        term = permutedims(skewness_tensor ,[2,1]).* permutedims(delta_fermi ,[2,1]).* delt_func_leftdown
    
        # 计算结果
        result = ds * sum(term) * 2 * delta * N
        result_list[i]=imag(result)
        joint = ds * sum(imag(delt_func_leftdown))
        joint_list[i]=joint
    end
    # println("wlist for end:",Dates.now())
    return result_list, joint_list
    end



function shiftyyy(kvec::Vector{<:Real},W::Real,N::Int,omegalist::AbstractVector{<:Real},
    ds::Real,Ef::Real,gamma::Real,t::Real,tv::Real,delta::Real,
    random_vector::Vector{<:Real},λ::Real)
    # println("shiftyyy start:",Dates.now())
    dky = [0 ;λ]

    H_total = generate_H_sk(kvec, t, tv, delta, N, W, random_vector)
    eigen_result = eigen(Hermitian(H_total)) # 优化：显式声明厄米矩阵
    evalues = eigen_result.values
    Uk = eigen_result.vectors


    H_zy = generate_H_sk(kvec .+ dky, t, tv, delta, N, W, random_vector)
    zy_result = eigen(Hermitian(H_zy))
    #evalues_zy = zy_result.values
    Uzy = zy_result.vectors
    # println("evectors_zy")


    H_fy = generate_H_sk(kvec .- dky, t, tv, delta, N, W, random_vector)
    fy_result = eigen(Hermitian(H_fy))
    #evalues_fy = fy_result.values
    Ufy = fy_result.vectors

    # println("eigen end:",Dates.now())

    basis_size = size(evalues, 1)
    skewness_tensor = zeros(Float64, basis_size÷2, basis_size÷2)

    m1=Uk' *Uzy
    m2=Uk' *Ufy
    m3=Uzy' *Ufy

    # re1=diagm(diag(m1) .*diag(m1'))
    re2=transpose(m1) .*m3 .*transpose(diag(m2'))
    re3=transpose(m2) .*m3' .*transpose(diag(m1'))
    # re4=diagm(diag(m2) .*diag(m2'))

    # term1=(1/(2*λ^3)) .*( re1
    #                 .- 2 .*m1' .*conj(m1')
    #                 .+ re2
    #                 .- re3
    #                 .+ 2 .*m2' .*conj(m2')
    #                 .- re4 )
    term1=(1/(2*λ^3)) .*(- 2 .*m1' .*conj(m1')
                    .+ re2
                    .- re3
                    .+ 2 .*m2' .*conj(m2'))

    term2_1=-re2
            .-transpose(m1) .*diag(m3) .*m3' .*transpose(diag(m1'))

    term2_2= transpose(m2) .*diag(m3') .*m3 .*transpose(diag(m2'))
            .+re3

    term2=(1/(8*λ^3)) .*(term2_1 .+ term2_2)
    term=term1 .+term2
    term_img=imag(0.5*(term -transpose(term)))
    skewness_tensor+=term_imag[1:basis_size÷2,(basis_size÷2+1):basis_size]

    # println("skewness_tensor",skewness_tensor)
    ΔE=evalues[1:basis_size÷2] .- reshape(evalues[basis_size÷2+1:basis_size],1,basis_size÷2)

    num_w=size(omegalist,1)
    result_list=zeros(Float64,num_w)
    joint_list=zeros(Float64,num_w)
    # println("tensor end:",Dates.now())
    # Fermi分布函数 (需实现DF_dis函数)
    fermi = DF_dis(evalues, Ef, gamma)
    basissize = size(skewness_tensor, 2) * 2  # 获取张量第二维大小

    # 重塑费米分布矩阵
    fermi_row = reshape(fermi[1:div(basissize,2)], div(basissize,2), 1)
    fermi_col = reshape(fermi[div(basissize,2)+1:basissize], 1, div(basissize,2) )

    delta_fermi = fermi_row .- fermi_col

    for i=1:num_w
        omega=omegalist[i]
        
        delt_func_leftdown = -1 .* zero_delta_realimag(omega, -permutedims(ΔE, [2,1]), gamma)

        term = permutedims(skewness_tensor ,[2,1]).* permutedims(delta_fermi ,[2,1]).* delt_func_leftdown
    
        # 计算结果
        result = ds * sum(term) * 2 * delta * N
        result_list[i]=imag(result)
        joint = ds * sum(imag(delt_func_leftdown))
        joint_list[i]=joint
    end
    # println("wlist for end:",Dates.now())
    return result_list, joint_list
    end




function calculate_shift_current(component::Symbol, kvec::Vector{<:Real}, W::Real, N::Int, omegalist::AbstractVector{<:Real}, ds::Real, Ef::Real, gamma::Real, t::Real, tv::Real, delta::Real, random_vector::Vector{<:Real}, λ::Real)
    if component === :yyy
        return shiftyyy(kvec, W, N, omegalist, ds, Ef, gamma, t, tv, delta, random_vector, λ)
    elseif component === :xyy
        return shiftxyy(kvec, W, N, omegalist, ds, Ef, gamma, t, tv, delta, random_vector, λ)
    else
        error("Unknown shift current component: ", component)
    end
end


function main(job_id::String)
    nk::Int=51
    # 通过 method 参数选择k点生成方法 (:rhombus 或 :square)
    kvecs,ds=kmesh_generator(nk, method=:rhombus)
    kvecs_length=size(kvecs,1)

    gamma=10
    #λ=sqrt(ds)
    λ=1e-4
    #println("lambda=",λ)
    t=-3.16*1000
    tv=0.381*1000
    delta=40
    N=26;basis_size=4*N^2
    W=400
    
    # 在这里设置要计算的分量 (:yyy 或 :xyy)
    component_to_run::Symbol = :yyy

    random_vector=zeros(4*N^2)

    Ef::Float64 = 0.0
    num_w::Int = 60
    wlist::LinRange{Float64} = LinRange(0.0, 7*delta, num_w)
    shift_list = zeros(Float64, num_w)
    joint_list = zeros(Float64, num_w)
    # Parameter for MPI 
    MPI.Init()
    comm = MPI.COMM_WORLD
    #BLAS.set_num_threads(4)
    root = 0
    numcore = MPI.Comm_size(comm)  # 申请的核心数量
    indcore = MPI.Comm_rank(comm)  # 核心的id标号
    println("num_core",numcore)
    nki = Int(floor(indcore * kvecs_length/numcore) + 1)
    nkf = Int(floor((indcore + 1) * kvecs_length/numcore)) 

    if (MPI.Comm_rank(comm) == root)
        println("BLAS:",BLAS.get_num_threads())
        println("开始计算inter-band QT: ",Dates.now())
        println("Number of nk : ",kvecs_length)
    end

    if indcore == 0
        #Random.seed!(12138) 
        random_vector=rand(4*N^2) .- 1/2
    end
    MPI.Bcast!(random_vector, 0, comm)
    MPI.Barrier(comm)

    #if (MPI.Comm_rank(comm) == root)
    #    println(rank,":",random_vector)
    #end
    #MPI.Barrier(comm)
    #if (MPI.Comm_rank(comm) == 1)
    #    println(rank,":",random_vector)
    #end
    #MPI.Barrier(comm)

    for i in nki:nkf
        if (MPI.Comm_rank(comm) == root)
            println(i,"/",(nkf-nki+1))
        end
        kvec=kvecs[i,:]
        result,joint=calculate_shift_current(component_to_run, kvec,W,N,wlist,ds,Ef,gamma,t,tv,delta,random_vector,λ)
        shift_list +=result
        joint_list +=joint
    end
    
    MPI.Barrier(comm)
    shift_list = MPI.Reduce(shift_list,MPI.SUM,root,comm)
    joint_list = MPI.Reduce(joint_list,MPI.SUM,root,comm)

    if (MPI.Comm_rank(comm) == root)
        println("inter-band QT计算结束: ",Dates.now())
        nk_string=(a->(@sprintf "%d" a)).(nk)
        N_string=(a->(@sprintf "%d" a)).(N)
        num_w_string=(a->(@sprintf "%d" a)).(num_w)
        W_string = (a->(@sprintf "%d" a)).(W)
        component_string = string(component_to_run)
        #println("shift", shift_list)
        #println("wlist", wlist)
        #println("delta",delta,"gamma",gamma,"W",W)
        fx1 ="cNonl1_W" * W_string * "_1_" * job_id * "-" * component_string * "-" * nk_string * "-" * N_string * "-"* num_w_string*".jld"
        target_dir = "N26"
        mkpath(target_dir)
        full_path = joinpath(target_dir, fx1)
        save(full_path, "shift", shift_list, "wlist", wlist,"delta",delta,"gamma",gamma,"W",W,"random_vector",random_vector,"joint",joint_list)    
    end

    MPI.Finalize()
end

job_id_str = ARGS[1]

main(job_id_str)
# wlist,shift_list=main()
# plot(wlist,shift_list)
# savefig("myplot.png")  
