function shiftyyy(kvec::Vector{<:Real},W::Real,N::Int,omegalist,
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
    term=imag(0.5*(term -transpose(term)))
    skewness_tensor+=term[1:basis_size÷2,(basis_size÷2+1):basis_size]

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
