module ConvolutionTools

import Base.conv

export conv, conv_valid, pad_const

"""
    pad_const(a, n[, c])

Pads any multidimensional array with `n[i]` constant values `c` on both sides.
Returns an array of size `size(a) + 2*n`.
"""
function pad_const(a::Array, n::Array{Int, 1}, c=0)
    shape = collect(size(a)) .+ 2 .* n
    b = zeros(eltype(a), shape...) .+ c
    block = [(n[i] + 1):(size(a)[i] + n[i]) for i in 1:length(n)]
    b[block...] = a
    return b
end

function conv{T<:Base.LinAlg.BlasFloat, N}(u::Array{T, N}, v::Array{T, N})
    su = [size(u)...]
    sv = [size(v)...]
    sp = su .+ sv .- 1 # for zero-padding
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
    # undo FFT padding
    full = [1:sp[i] for i in 1:N]
    return y[full...]
end
conv{T<:Integer, N}(u::Array{T, N}, v::Array{T, N}) = round(Int, conv(float(u), float(v)))
conv{T<:Integer, S<:Base.LinAlg.BlasFloat, N}(u::Array{T, N}, v::Array{S, N}) = conv(float(u), v)
conv{T<:Integer, S<:Base.LinAlg.BlasFloat, N}(u::Array{S, N}, v::Array{T, N}) = conv(u, float(v))

function strip_invalid{T, S, N}(output::Array{T, N}, stencil::Array{S, N})
    shape_padded = [size(output)...]
    shape_stencil = [size(stencil)...]
    shape_valid = shape_padded .- 2 .* (shape_stencil .- 1)
    invalid = shape_stencil .- 1
    lower = invalid .+ 1
    upper = invalid .+ shape_valid
    center = [lower[i]:upper[i] for i in 1:N]
    return output[center...]
end

function conv_valid(data, stencil)
    full = conv(data, stencil)
    return strip_invalid(full, stencil)
end

end # module
