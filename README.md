# DifferentialMatrix


This packages generates differential matrices. Given a functio `f(x)` evaluated at an increasing vector `x`, it creates a matrix `D` such that `D*f(x)` approximates `f'(x)` using either forward or backward differences. 

Usage
```
using DifferentialMatrix


function f(x)
...
end
Delta = 0.01
x = collect(0:Delta:10)

D_x = create_diff_matrix(size(x), Delta = Delta, direction="forward")

f_prime_forward = D_x * f(x)

```