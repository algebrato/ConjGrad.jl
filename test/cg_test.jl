include("../src/ConjGrad.jl")
#using ConjGrad
using LinearAlgebra, SparseArrays
using Test
using Random
using FFTW


using JLD
@load("../strip2/ciao")

N = 1024

function binned_map(N_pix::Int, tod::Array{Float64, 1},
                    points::Array{Int64, 2})
    map = zeros(N_pix, N_pix)
    hit = zeros(N_pix, N_pix)

    for i = 1:length(tod)
        map[points[i, 1], points[i, 2]] += tod[i]
        hit[points[i, 1], points[i, 2]] += 1.0
    end

    return map, hit

end


function get_tod(sky::Array{Float64, 2}, points::Array{Int64, 2})

    Num_of_pixels = size(points)[1]
    tod = [sky[points[i,1], points[i,2]] for i = 1:Num_of_pixels]

    return tod

end

# inserire il denoise
function denoise(tod::Array{Float64, 1}, noise_s::Array{Float64, 1})
  ftod = fft(tod)
  ftod ./= noise_s
  tod = real.(ifft(ftod))
  return tod
end

b = zeros(N*N)
for i in dataset_baselines
  tod = denoise(i.time_order_data, i.noise)
  a, h = binned_map(N, tod, i.pointing)
  b .+= reshape(a, N*N, 1)[:, 1]
end


function A(x)
  res = zeros(N, N)
  for i in dataset_baselines
    tod = get_tod(reshape(x, N, N), i.pointing)
    tod = denoise(tod, i.noise)
    mat, hmat = binned_map(N, tod, i.pointing)
    res .+= mat
  end
  return reshape(res, N*N, 1)[:, 1]
end

x=zeros(length(b))
data = ConjGrad.CGData{Float64}(length(b))
x, exit_code, num_iters=ConjGrad.cg(A, b, tol=1e-3, precon=A)
