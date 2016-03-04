using ConvolutionTools
using Base.Test

import Base.isapprox

function isapprox{T, N}(a::Array{T, N}, b::Array{T, N}, eps=10e-10)
    if size(a) != size(b)
        return false
    end
    el_diff = sum(abs(a .- b)) / prod(size(a))
    return el_diff < eps
end

function test_1d()
    a = rand((2))
    b = rand((7))
    res = conv(a, b)
    return ndims(res) == 1 && eltype(res) <: Number
end

function test_2d()
    a = rand((1, 2))
    b = rand((6, 7))
    res = conv(a, b)
    return ndims(res) == 2 && eltype(res) <: Number
end

function test_3d()
    a = rand((1, 2, 3))
    b = rand((6, 7, 10))
    res = conv(a, b)
    return ndims(res) == 3 && eltype(res) <: Number
end

function test_identity()
    a = float([1])
    b = float([i^2 for i in 1:100])
    res = conv(a, b)
    return isapprox(b, res)
end

function test_zero()
    a = float([0])
    b = float([i^2 for i in 1:100])
    res = conv(a, b)
    return all(res .== zero(eltype(res)))
end

function test_derivative()
    n = 10
    a = float([1, 0, -1]) / 2
    b = float([0.5 * i^2 for i in 1:n])
    res = conv(a, b)
    ref = float([i for i in 2:(n-1)])
    return isapprox(ref, res)
end

custom_handler(r::Test.Success) = print(".")
custom_handler(r::Test.Failure) = Test.default_handler(r)
custom_handler(r::Test.Error)   = Test.default_handler(r)

Test.with_handler(custom_handler) do
    @test test_1d()
    @test test_2d()
    @test test_3d()
    @test test_zero()
    @test test_identity()
    @test test_derivative()
end

println("")

