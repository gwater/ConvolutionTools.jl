module ConvolutionTools

import Base.conv

export conv

function conv{T<:Base.LinAlg.BlasFloat, N}(u::Array{T, N}, v::Array{T, N})
    su = [size(u)...]
    sv = [size(v)...]
    sr = abs(su .- sv) .+ 1 # for cropping valid parts
    sp = su .+ sv .- 1      # for zero-padding
    # pad for efficient FFT
    sp2 = [n > 1024 ? nextprod([2,3,5], n) : nextpow2(n) for n in sp]
    upad = zeros(T, (sp2...))
    upad[[1:n for n in su]...] = u
    vpad = zeros(T, (sp2...))
    vpad[[1:n for n in sv]...] = v
    if T <: Real
        p = plan_rfft(upad)
    else
        p = plan_fft(upad)
    end
    # perform convolution
    y = p \ ((p * upad) .* (p * vpad))
    # only return valid part
    invalid = div(sp .- sr,  2)
    lower = invalid .+ 1
    upper = invalid .+ sr
    center = [lower[i]:upper[i] for i in 1:N]
    return y[center...]
end
conv{T<:Integer, N}(u::Array{T, N}, v::Array{T, N}) = round(Int, conv(float(u), float(v)))
conv{T<:Integer, S<:Base.LinAlg.BlasFloat, N}(u::Array{T, N}, v::Array{S, N}) = conv(float(u), v)
conv{T<:Integer, S<:Base.LinAlg.BlasFloat, N}(u::Array{S, N}, v::Array{T, N}) = conv(u, float(v))

end # module
