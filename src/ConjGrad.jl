module ConjGrad

    using LinearAlgebra

    export CGData
    export cg
    export cg!
    export funzz

    include("linearalgebra.jl")
    include("cg.jl")


end  # module ConjGrad
