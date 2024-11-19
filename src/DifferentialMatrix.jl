module DifferentialMatrix

export compute_vec_index, compute_vec_index_inv, create_diff_matrix

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



function create_diff_matrix_1d(shape::Tuple, 
    direction::String="forward",
    Delta::Union{Float64, Array{Float64}}=1.)
    D_row, D_col, D_val = Int[], Int[], Float64[]
    if isa(Delta, Float64)
        Delta = fill(Delta, shape[1]-1)
    else
        @assert length(Delta) == shape[1]-1
    end
    for i in 1:shape[1]
        row = compute_vec_index((i,), shape)
    if direction == "forward" && i < shape[1]
        push!(D_row, row)
        push!(D_col, row)
        push!(D_val, -1/Delta[i])
        push!(D_row, row)
        push!(D_col, compute_vec_index((i + 1,), shape))
        push!(D_val, 1/Delta[i])
    elseif direction == "backward" && i > 1
        push!(D_row, row)
        push!(D_col, row)
        push!(D_val, 1/Delta[i-1])
        push!(D_row, row)
        push!(D_col, compute_vec_index((i - 1,), shape))
        push!(D_val, -1/Delta[i-1])
    end
    end

    return sparse(D_row, D_col, D_val, prod(shape), prod(shape))
end



function create_diff_matrix(shape::Tuple, 
                            dim::Int = 1;
                            direction::String="forward",
                            Delta::Union{Float64, Array{Float64}}=1.)
    D_row, D_col, D_val = Int[], Int[], Float64[]
    if isa(Delta, Float64)
        Delta = fill(Delta, shape[dim]-1)
    else
        @assert length(Delta) == shape[dim]-1
    end

    if length(shape) == 1
        return create_diff_matrix_1d(shape, direction, Delta)
    end

    for k in 1:shape[3]
        for j in 1:shape[2]
            for i in 1:shape[1]
                row = compute_vec_index((i, j, k), shape)
                if dim == 1
                    if direction == "forward" && i < shape[1]
                        push!(D_row, row)
                        push!(D_col, row)
                        push!(D_val, -1/Delta[i])
                        push!(D_row, row)
                        push!(D_col, compute_vec_index((i + 1, j, k), shape))
                        push!(D_val, 1/Delta[i])
                    elseif direction == "backward" && i > 1
                        push!(D_row, row)
                        push!(D_col, row)
                        push!(D_val, 1/Delta[i-1])
                        push!(D_row, row)
                        push!(D_col, compute_vec_index((i - 1, j, k), shape))
                        push!(D_val, -1/Delta[i-1])
                    end
                elseif dim == 2
                    if direction == "forward" && j < shape[2]
                        push!(D_row, row)
                        push!(D_col, row)
                        push!(D_val, -1/Delta[j])
                        push!(D_row, row)
                        push!(D_col, compute_vec_index((i, j + 1, k), shape))
                        push!(D_val, 1/Delta[j])
                    elseif direction == "backward" && j > 1
                        push!(D_row, row)
                        push!(D_col, row)
                        push!(D_val, 1/Delta[j-1])
                        push!(D_row, compute_vec_index((i, j, k), shape))
                        push!(D_col, compute_vec_index((i, j - 1, k), shape))
                        push!(D_val, -1/Delta[j-1])
                    end
                elseif dim == 3
                    if direction == "forward" && k < shape[3]
                        push!(D_row, row)
                        push!(D_col, row)
                        push!(D_val, -1/Delta[k])
                        push!(D_row, row)
                        push!(D_col, row)
                        push!(D_val, 1/Delta[k])
                    elseif direction == "backward" && k > 1
                        push!(D_row, row)
                        push!(D_col, row)
                        push!(D_val, 1/Delta[k-1])
                        push!(D_row, row)
                        push!(D_col, compute_vec_index((i, j, k - 1), shape))
                        push!(D_val, -1/Delta[k-1])
                    end
                end
            end
        end
    end

    return sparse(D_row, D_col, D_val, prod(shape), prod(shape))
end

end