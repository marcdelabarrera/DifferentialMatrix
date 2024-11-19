# DifferentialMatrix

This package generates differential matrices. Given a function `f(x)` evaluated at an increasing vector `x`, it creates a matrix `D` such that `D*f(x)` approximates `f'(x)` using either forward or backward differences.

## Usage

```julia
using DifferentialMatrix

function f(x)
    #your function here
end

Delta = 0.01
x = collect(0:Delta:10)

D_x = create_diff_matrix(size(x), Delta = Delta, direction="forward")

f_prime_forward = D_x * f(x)
```

Note: the `forward` derivative at the upper bound of the grid is not defined and set to `0`. Similarly, the `backward` derivative at the first grid point is not defined and set to `0`.

## Further dimensions

For 2 or 3 dimensional functions, given `f(x)`, compute the matrix `D` such that `D vec(f(x))` is `vec(f_prime(x))`. Set direction `1` or `2` or `3` to specify the direction of the derivative.