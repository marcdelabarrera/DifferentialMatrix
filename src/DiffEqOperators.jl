using StaticArrays: SVector

struct DerivativeOperator{T <: Real, N, Wind, T2, S1, S2, S3, T3, F} 
    derivative_order::Int
    approximation_order::Int
    dx::T2
    len::Int
    stencil_length::Int
    stencil_coefs::S1
    boundary_stencil_length::Int
    boundary_point_count::Int
    low_boundary_coefs::S2
    high_boundary_coefs::S3
    offside::Int
    coefficients::T3
    coeff_func::F
   end

   """
```
compute_coeffs(coeff_func, current_coeffs)
```
Calculates the coefficients for the stencil of UpwindDifference operators.
"""
function compute_coeffs!(coeff_func::Number,
                         current_coeffs::AbstractVector{T}) where {T <: Number}
    return current_coeffs .+= coeff_func
end

function calculate_weights(order::Int, x0::T, x::AbstractVector) where {T <: Real}
    #=
        order: The derivative order for which we need the coefficients
        x0   : The point in the array 'x' for which we need the coefficients
        x    : A dummy array with relative coordinates, e.g., central differences
               need coordinates centred at 0 while those at boundaries need
               coordinates starting from 0 to the end point

        The approximation order of the stencil is automatically determined from
        the number of requested stencil points.
    =#
    N = length(x)
    @assert order<N "Not enough points for the requested order."
    M = order
    c1 = one(T)
    c4 = x[1] - x0
    C = zeros(T, N, M + 1)
    C[1, 1] = 1
    @inbounds for i in 1:(N - 1)
        i1 = i + 1
        mn = min(i, M)
        c2 = one(T)
        c5 = c4
        c4 = x[i1] - x0
        for j in 0:(i - 1)
            j1 = j + 1
            c3 = x[i1] - x[j1]
            c2 *= c3
            if j == i - 1
                for s in mn:-1:1
                    s1 = s + 1
                    C[i1, s1] = c1 * (s * C[i, s] - c5 * C[i, s1]) / c2
                end
                C[i1, 1] = -c1 * c5 * C[i, 1] / c2
            end
            for s in mn:-1:1
                s1 = s + 1
                C[j1, s1] = (c4 * C[j1, s1] - s * C[j1, s]) / c3
            end
            C[j1, 1] = c4 * C[j1, 1] / c3
        end
        c1 = c2
    end
    #=
        This is to fix the problem of numerical instability which occurs when the sum of the stencil_coefficients is not
        exactly 0.
        https://scicomp.stackexchange.com/questions/11249/numerical-derivative-and-finite-difference-coefficients-any-update-of-the-fornb
        Stack Overflow answer on this issue.
        http://epubs.siam.org/doi/pdf/10.1137/S0036144596322507 - Modified Fornberg Algorithm
    =#
    _C = C[:, end]
    if order != 0
        _C[div(N, 2) + 1] -= sum(_C)
    end
    return _C
end

index(i::Int, N::Int) = i + div(N, 2) + 1

function generate_coordinates(i::Int, stencil_x, dummy_x,
    dx::AbstractVector{T}) where {T <: Real}
len = length(stencil_x)
stencil_x .= stencil_x .* zero(T)
for idx in 1:div(len, 2)
shifted_idx1 = index(idx, len)
shifted_idx2 = index(-idx, len)
stencil_x[shifted_idx1] = stencil_x[shifted_idx1 - 1] + dx[i + idx - 1]
stencil_x[shifted_idx2] = stencil_x[shifted_idx2 + 1] - dx[i - idx]
end
return stencil_x
end


struct CenteredDifference{N} end

function CenteredDifference{N}(derivative_order::Int,
    approximation_order::Int, dx::AbstractVector{T},
    len::Int, coeff_func = 1) where {T <: Real, N}
stencil_length = derivative_order + approximation_order - 1 +
(derivative_order + approximation_order) % 2
boundary_stencil_length = derivative_order + approximation_order
stencil_x = zeros(T, stencil_length)
boundary_point_count = div(stencil_length, 2) - 1# -1 due to the ghost point

interior_x = (boundary_point_count + 2):(len + 1 - boundary_point_count)
dummy_x = (-div(stencil_length, 2)):(div(stencil_length, 2) - 1)
low_boundary_x = [zero(T); cumsum(dx[1:(boundary_stencil_length - 1)])]
high_boundary_x = cumsum(dx[(end - boundary_stencil_length + 1):end])
# Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
deriv_spots = (-div(stencil_length, 2) + 1):-1

stencil_coefs = [convert(SVector{stencil_length, T},
  calculate_weights(derivative_order, zero(T),
                    generate_coordinates(i, stencil_x, dummy_x,
                                         dx)))
for i in interior_x]
_low_boundary_coefs = SVector{boundary_stencil_length, T}[convert(SVector{
                                                   boundary_stencil_length,
                                                   T},
                                           calculate_weights(derivative_order,
                                                             low_boundary_x[i + 1],
                                                             low_boundary_x))
                                   for i in 1:boundary_point_count]
low_boundary_coefs = convert(SVector{boundary_point_count}, _low_boundary_coefs)
_high_boundary_coefs = SVector{boundary_stencil_length, T}[convert(SVector{
                                                    boundary_stencil_length,
                                                    T},
                                            calculate_weights(derivative_order,
                                                              high_boundary_x[end - i],
                                                              high_boundary_x))
                                    for i in boundary_point_count:-1:1]
high_boundary_coefs = convert(SVector{boundary_point_count}, _high_boundary_coefs)

offside = 0

coefficients = zeros(T, len)

compute_coeffs!(coeff_func, coefficients)

DerivativeOperator{T, N, false, typeof(dx), typeof(stencil_coefs),
typeof(low_boundary_coefs), typeof(high_boundary_coefs),
typeof(coefficients),
typeof(coeff_func)}(derivative_order, approximation_order, dx,
                len, stencil_length,
                stencil_coefs,
                boundary_stencil_length,
                boundary_point_count,
                low_boundary_coefs,
                high_boundary_coefs, offside, coefficients,
                coeff_func)
end


CenteredDifference(args...) = CenteredDifference{1}(args...)
