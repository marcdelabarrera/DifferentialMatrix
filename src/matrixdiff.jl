using SparseArrays


"""
Computes the vector index corresponding to a multidimensional index in a tensor 
of the specified `shape`.

# Arguments
- `index::Tuple`: The multidimensional index (1-based) in the tensor.
- `shape::Tuple`: The shape of the tensor.
- `order::String="F"`: The ordering of the tensor, either "F" (Fortran-style, column-major)
  or "C" (C-style, row-major). Defaults to "F".
"""
function compute_vec_index(index::Tuple, shape::Tuple, order::String="F")::Int64
    @assert all(index.<=shape)
    vec_index = 0
    if order == "F"
        for i in eachindex(index)
            if i == 1
                vec_index = index[i]
            else
                vec_index += (index[i]-1)*prod(shape[1:i-1])
            end
        end
    elseif order == "C"
        for i in eachindex(index)
            if i == 1
                vec_index = index[end]
            else
                vec_index += (index[end-i+1]-1)*prod(shape[end-i+2:end])
            end
        end
    else
        error("order must be either F (Fortran, row-major) or C (C, column-major)")
    end
    return vec_index
end


function compute_vec_index_inv(vec_index::Int64, shape::Tuple, order::String="F")::Tuple
    @assert vec_index <= prod(shape)
    dims = length(shape)
    index = fill(1, dims)
    if order == "F"
        remainder = vec_index - 1  
        for i in 1:dims
            index[i] = (remainder % shape[i]) + 1  
            remainder = div(remainder, shape[i])
        end
    else
        error("Order must be F")
    end
    return tuple(index...)
end



function diff_matrix(shape::Tuple,
                    dim::Int64, 
                    direction::String="forward", 
                    Delta::Union{Float64, Array{Float64, 1}}=1.0)
    if isa(Delta, Float64)
        Delta = fill(Delta, shape[dim]-1)
    else
        @assert shape[dim]-1 == length(Delta)
    end
    if !(direction in ["forward", "backward"])
        error("direction must be either 'forward' or 'backward'")
    end
    if length(shape) == 2
        return diff_matrix_2d(shape, dim, direction, Delta)
    elseif length(shape) == 3
        return diff_matrix_3d(shape, dim, direction, Delta)
    end
end

function diff_matrix_2d(shape::Tuple, dim::Int64, direction::String, Delta:: Array{Float64, 1})
    if direction == "forward"
        return diff_matrix_2d_forward(shape, dim, Delta)
    elseif direction == "backward"
        return diff_matrix_2d_backward(shape, dim, Delta)
    end
end

function diff_matrix_3d(shape::Tuple, dim::Int64, direction::String, Delta::Array{Float64, 1})
    if direction == "forward"
        return diff_matrix_3d_forward(shape, dim, Delta)
    elseif direction == "backward"
        return diff_matrix_3d_backward(shape, dim, Delta)
    end
end

function diff_matrix_2d_forward(shape::Tuple, dim::Int64, Delta::Array{Float64, 1})
    if dim == 1
        diag = -vcat(repeat(Delta, shape[2]), zeros(shape[1]))
        off_diag = -diag[1:(end-shape[1])]
        offset = shape[1]
        return spdiagm(offset => off_diag)+spdiagm(diag)
    elseif dim == 2
        diag = -repeat(vcat(Delta,0.),shape[2])
        off_diag = -diag[1:end-1]
        offset = 1
        return spdiagm(offset=>off_diag)+spdiagm(diag)
    end
end


function diff_matrix_3d_forward(shape::Tuple, dim::Int64, Delta::Array{Float64, 1})
    if dim == 1
        diag = vcat(repeat(Delta, inner=shape[2] * shape[3]), zeros(shape[1] * shape[2] * shape[3] - shape[2] * shape[3]))
        off_diag = -diag[1:(end-shape[2] * shape[3])]
        offset = shape[2] * shape[3]
        return spdiagm(offset => off_diag) + spdiagm(diag)
    elseif dim == 2
        diag = vcat(repeat(Delta, inner=shape[3], outer=shape[1]), zeros(shape[1] * shape[2] * shape[3] - shape[1] * shape[3]))
        off_diag = -diag[1:(end-shape[3])]
        offset = shape[3]
        return spdiagm(offset => off_diag) + spdiagm(diag)
    elseif dim == 3
        diag = -repeat(vcat(Delta, 0.), shape[1] * shape[2])
        off_diag = -diag[1:end-1]
        offset = 1
        return spdiagm(offset => off_diag) + spdiagm(diag)
    end
end
