using ConjGrad
using LinearAlgebra, SparseArrays
using Test


function test_cg()
    tA = sprandn(100,100,.1) .+ 10.0*sparse(1.0I, 100, 100)
    A = tA'*tA
    b = rand(100)
    true_x = A\b
    x, exit_code, num_iters = ConjGrad.cg((x,y) -> mul!(x, A, y) , b)

    if norm(true_x - x) < 1e-6
        return true
    else
        return false
    end
end


@test test_cg()
