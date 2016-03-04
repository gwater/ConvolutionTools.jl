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
    a = rand((7))
    b = rand((2))
    res = ConvolutionTools.conv_valid(a, b)
    return ndims(res) == 1 && eltype(res) <: Number
end

function test_2d()
    a = rand((6, 7))
    b = rand((1, 2))
    res = ConvolutionTools.conv_valid(a, b)
    return ndims(res) == 2 && eltype(res) <: Number
end

function test_3d()
    a = rand((6, 7, 10))
    b = rand((1, 2, 3))
    res = ConvolutionTools.conv_valid(a, b)
    return ndims(res) == 3 && eltype(res) <: Number
end

function test_identity()
    a = [i^2 for i in 1:100]
    b = [1]
    res = ConvolutionTools.conv_valid(a, b)
    return all(a .== res)
end

function test_zero()
    a = float([i^2 for i in 1:100])
    b = float([0])
    res = ConvolutionTools.conv_valid(a, b)
    return all(res .== zero(eltype(res)))
end

function test_derivative()
    n = 10
    a = float([0.5 * i^2 for i in 1:n])
    b = float([1, 0, -1]) / 2
    res = ConvolutionTools.conv_valid(a, b)
    ref = float([i for i in 2:(n-1)])
    return isapprox(ref, res)
end

function test_integers()
    n = 10
    a = [i for i in 1:n]
    b = [1, 0, -1]
    res = ConvolutionTools.conv_valid(a, b)
    return all(res .== 2)
end

function test_mixed_input()
    a = [i for i in 1:10]
    b = float([1, 2, 3])
    res1 = ConvolutionTools.conv(a, b)
    res2 = ConvolutionTools.conv(b, a)
    return isapprox(res1, res2)
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
    @test test_integers()
    @test test_mixed_input()
end

println("")

