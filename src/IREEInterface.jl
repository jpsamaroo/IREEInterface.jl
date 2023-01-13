module IREEInterface

using Random

const IREE_ROOT = Ref{String}()

## IREE command-line interface

function compile(output_path::String, input_path::String; output_format=:bytecode)
    @assert isassigned(IREE_ROOT)
    @assert endswith(input_path, ".mlir")
    output_format_flag = if output_format == :bytecode
        @assert endswith(output_path, ".vmfb")
        "vm-bytecode"
    elseif output_format == :c
        @assert endswith(output_path, ".c")
        "vm-c"
    elseif output_format == :asm
        @assert endswith(output_path, ".vmvx")
        "vm-asm"
    else
        throw(ArgumentError("Invalid output format: $output_format"))
    end
    run(`$(joinpath(IREE_ROOT[], "iree-compile")) --iree-hal-target-backends=llvm-cpu --iree-input-type=tosa --output-format=$output_format_flag -o $output_path $input_path`)
end
function run_module(path::String, entry::String, @nospecialize(args...))
    @assert isassigned(IREE_ROOT)
    @assert endswith(path, ".vmfb")
    cmd = `$(joinpath(IREE_ROOT[], "iree-run-module")) --module_file=$path --entry_function=$entry`
    for arg in args
        input = julia_to_iree(arg)
        push!(cmd.exec, "--function_input=$input")
    end
    out = IOBuffer()
    run(pipeline(cmd; stdout=out))

    # Parse outputs
    seek(out, 0)
    out_str = String(take!(out))
    outputs = []
    lines = split(out_str, '\n')
    while !isempty(lines)
        line = popfirst!(lines)
        if startswith(line, "result[")
            if (m = match(r"result\[[0-9]*\]: ([0-9a-zA-Z]*)=([\.0-9a-zA-Z]*)", line)) !== nothing
                T, value = m.captures
                push!(outputs, iree_to_julia(T, value))
            elseif (m = match(r"result\[[0-9]*\]: ([\._0-9a-zA-Z]*)", line)) !== nothing
                value = m.captures[1]
                @assert value == "hal.buffer_view"
                line = popfirst!(lines)
                m = match(r"([0-9a-zA-Z]*)=([\.0-9a-zA-Z]*)", line)
                T, value = m.captures
                push!(outputs, iree_to_julia(T, value))
            end
        end
    end
    return outputs
end
function compile_and_run(input_path::String, entry::String, @nospecialize(args...))
    output_path = replace(input_path, ".mlir"=>".vmfb")
    compile(output_path, input_path)
    try
        return run_module(output_path, entry, args...)
    finally
        rm(output_path)
    end
end

## Julia to IREE argument conversion

julia_to_iree(arg) = throw(ArgumentError("Cannot map Julia argument of type $(typeof(arg)) to IREE: $arg"))
function julia_to_iree(::Type{T}) where T<:AbstractFloat
    @assert isprimitivetype(T)
    return "f"*repr(sizeof(T)*8)
end
function julia_to_iree(::Type{T}) where T<:Integer
    @assert isprimitivetype(T)
    return "i"*repr(sizeof(T)*8)
end
julia_to_iree(arg::T) where T<:Union{AbstractFloat,Integer} = "$arg"
function julia_to_iree(vec::Vector{T}) where T
    n = length(vec)
    t = julia_to_iree(T)
    return "$(n)x$t=[$(join(map(repr, vec), ' '))]"
end

## IREE to Julia result conversion

function iree_to_julia(T_str::AbstractString, value_str::AbstractString)
    T = iree_to_julia(T_str)
    return iree_to_julia(T, value_str)
end
function iree_to_julia(T_str::AbstractString)
    if startswith(T_str, 'f')
        nbits = parse(Int, T_str[2:end])
        if nbits == 16
            return Float16
        elseif nbits == 32
            return Float32
        elseif nbits == 64
            return Float64
        else
            error("Invalid number of bits for float: $nbits")
        end
    elseif startswith(T_str, 'i')
        nbits = parse(Int, T_str[2:end])
        if nbits == 8
            return Bool
        elseif nbits == 8
            return Int8
        elseif nbits == 16
            return Int16
        elseif nbits == 32
            return Int32
        elseif nbits == 64
            return Int64
        elseif nbits == 128
            return Int128
        else
            error("Invalid number of bits for integer: $nbits")
        end
    elseif isdigit(T_str[1])
        # Vector
        dims = Int[]
        local T
        while true
            x_idx = findfirst(c->c=='x', T_str)
            if x_idx !== nothing
                push!(dims, parse(Int, T_str[1:x_idx-1]))
                T_str = T_str[x_idx+1:end]
            else
                T = iree_to_julia(T_str)
                break
            end
        end
        return Array{T,length(dims)}
    else
        throw(ArgumentError("Cannot map IREE type $T_str to Julia"))
    end
end
function iree_to_julia(::Type{T}, value_str::AbstractString) where T
    if T <: Array
        # FIXME: Parse elements
        return parse(eltype(T), String(value_str))
    else
        # Scalar
        return parse(T, String(value_str))
    end
end

## MLIRFunction

const COMPILE_CACHE_LOCK = Threads.ReentrantLock()
const COMPILE_CACHE = Dict{String,String}()

struct MLIRFunction{RT,AT<:Tuple}
    data::String
    entry::String
    data_is_file::Bool
end
MLIRFunction(RT, AT, data, entry; data_is_file::Bool=false) =
    MLIRFunction{RT,Base.to_tuple_type(AT)}(data, entry, data_is_file)
Base.show(io::IO, fun::MLIRFunction{RT,AT}) where {RT,AT} =
    print(io, "MLIRFunction{$RT,$AT}(entry=$(fun.entry)$(fun.data_is_file ? ", data=<$(fun.data)>" : ""))")
function (fun::MLIRFunction{RT,AT})(args...) where {RT,AT}
    if !((args...,) isa AT)
        throw(ArgumentError("Argument types do not match MLIR function type\nFunction type: $AT\nArgument types: $(typeof(args))"))
    end

    # Try to get an already-compiled executable path
    exe_path = lock(COMPILE_CACHE_LOCK) do
        get(COMPILE_CACHE, fun.data, nothing)
    end

    # If not yet compiled, do so now
    if exe_path === nothing
        if fun.data_is_file
            mlir_path = fun.data
        else
            # Write out MLIR
            mlir_path = joinpath(tempdir(), "ireei_$(randstring()).mlir")
            open(mlir_path, "w+") do io
                write(io, fun.data)
            end
            atexit() do
                rm(mlir_path; force=true)
            end
        end

        # Compile to IREE vmfb file
        if fun.data_is_file
            exe_path = joinpath(tempdir(), "ireei_$(randstring()).vmfb")
        else
            exe_path = replace(mlir_path, ".mlir"=>".vmfb")
        end
        compile(exe_path, mlir_path)
        atexit() do
            rm(exe_path; force=true)
        end
        lock(COMPILE_CACHE_LOCK) do
            COMPILE_CACHE[fun.data] = exe_path
        end
    end

    # Run it!
    result = only(run_module(exe_path, fun.entry, args...))
    if !(result isa RT)
        throw(ArgumentError("Result type does not match MLIR function type\nFunction type: $RT\nResult type: $(typeof(result))"))
    end
    return result
end
function invalidate!(fun::MLIRFunction)
    lock(COMPILE_CACHE_LOCK) do
        if haskey(COMPILE_CACHE, fun.data)
            delete!(COMPILE_CACHE, fun.data)
        end
    end
end

function __init__()
    if haskey(ENV, "JULIA_IREE_ROOT")
        IREE_ROOT[] = ENV["JULIA_IREE_ROOT"]
    else
        path = try
            readline(run(`which iree-compile`))
        catch
            nothing
        end
        if path !== nothing
            IREE_ROOT[] = dirname(path)
        end
    end
end

end # module IREEInterface
