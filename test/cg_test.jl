using ConjGrad
using LinearAlgebra, SparseArrays
using Test

# Make your custom system


function funn(x, y)
    mul!(x, A, y)
end


#function test_cg()
tA = sprandn(100,100,.1) .+ 10.0*sparse(1.0I, 100, 100)
A = tA'*tA
b = rand(100)
true_x = A\b
@time x, exit_code, num_iters = ConjGrad.cg(funn, b, tol=1e-16)

#    if norm(true_x - x) < 1e-16
#        return true
#    else
#        return false
#    end
#end


@test test_cg()
