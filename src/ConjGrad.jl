module ConjGrad


using LinearAlgebra

include("linearalgebra.jl")
include("cg.jl")

export CGData, cg!, cg

end  # module ConjGrad
