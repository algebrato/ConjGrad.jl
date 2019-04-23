genblas_dot(x, y) = dot(x,y)
genblas_scal!(a, x) = x .*= a
genblas_axpy!(a, x, y) = y .+= a.*x
genblas_nrm2(x) = norm(x)

# I ought add the BLAS wrapper. 
