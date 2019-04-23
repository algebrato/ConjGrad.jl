struct CGData{T<:Real}
    r::Vector{T}
    z::Vector{T}
    p::Vector{T}
    Ap::Vector{T}
    CGData{T}(n::Int) where T <: Real = new(zeros(T, n), zeros(T, n),
                                            zeros(T, n), zeros(T, n))
end

function cg!{T::Type}(A, b::Vector{T}, x::Vector{T};
             tol::Float64=1e-6, maxIter::Int64=100,
             precon=copy!, data=CGData{T}(length(b)))

    n = length(b)
    n_iter = 0
    if genblas_nrm2(b) == 0.0
        x .= 0
        return 1, 0
    end

    A(data.r, x)
    genblas_scal!(-one(T), data.r)
    genblas_axpy(one(T), b, data.r)
    residual_0 = genblas_nrm2(data.r)
    if residual_0 <= tol
        return 2, 0
    end
    precon(data.z, data.r)
    data.p = data.z
    for iter = 1:maxIter
        A(data.Ap, data.p)
        gamma = genblas_dot(data.r, data.z)
        alpha = gamma/genblas_dot(data.p, data.Ap)
        if alpha == Inf || alpha < 0
            return -13, iter
        end
        genblas_axpy!(alpha, delta.p, x)
        genblas_axpy!(-alpha, data.Ap, data.r)
        residual = genblas_nrm2(dara.r)/residual_0
        if residual <= tol
            return 30, iter
        end
        precon(data.z, data.r)
        beta = genblas_dot(data.z, data.r)/gamma
        genblas_scal!(beta, data.p)
        genblas_axpy!(1.0, data.z, data.p)
    end
    return -2, maxIter
end

function cg{T<:Real}(A, b::Vector{T};
                     tol::Float64=1e-6, maxIter::Int64=100,
                     precon=copy!, data=CGData(length(b), T))
    x = zeros(b)
    exit_code, num_iters = cg!(A, b, x, tol=tol, maxIter=maxIter, precon=precon, data=data)
    return x, exit_code, num_iters
end
