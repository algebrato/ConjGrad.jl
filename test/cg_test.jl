using LinearAlgebra, SparseArrays
using ConjGrad
using Test

function test_cg()
    tA = sprandn(100,100,.1) .+ 10.0*sparse(1.0I, 100, 100)
    A = tA'*tA
    b = rand(100)
    true_x = A\b
    comm = missing
    x, exit_code, num_iters = cg((x)->(A * x) , b,
                                 tol=1e-16,
                                 maxIter=1000,
                                 precon=copy!,
                                 verbose=false,
                                 comm=comm
                                 )

    if norm(true_x - x) < 1e-16
        return true
    else
        return false
    end

end

@test test_cg()
