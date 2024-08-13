###############################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                               v1.0
#
###############################################################################
# Made by Thomas E. Baker and « les qubits volants » (2024)
# See accompanying license with this program
# This code is native to the julia programming language (v1.10.0+)
#

"""
    V = environment(T...)

Inputs tensors `T` into environment `V`
"""
function (environment(T::G...) where G <: Union{tens{W},Array{W,N},Qtens{W,Q}}) where {W <: Number, N, Q <: Qnum}
#  return environment{G}(network([T...]))
  if eltype(T[1]) <: Number
    out = environment{G}(network([T...]))
  else
#    println(T)
    check = vcat(copy(T)...)
#    println(check)
    out = environment{typeof(T[1][1])}(network(check))
  end
  return out
end
export environment

"""
    V = environment(W,Ns)

Creates a blank environment of type `W` with entries for `Ns` tensors
"""
function environment(T::W,Ns::Integer) where W <: TensType
  vect = Array{W,1}(undef,Ns)
  for w = 1:Ns
    vect[w] = copy(T)
  end
  return environment{W}(network(vect))
end

"""
    V = environment(P)

Inputs tensors `P` representing an `MPS` into environment `V`

See also: [`MPS`](@ref)
"""
function environment(network::MPS)
  return environment(copy(network.A.net))
end

"""
    V = environment(P)

Inputs tensors `P` representing an `MPO` into environment `V`

See also: [`MPO`](@ref)
"""
function environment(network::MPO)
  return environment(copy(network.H.net))
end

"""
    V = environment(P)

Inputs tensors `P` representing a `network` into environment `V`

See also: [`network`](@ref)
"""
function environment(network::network{W}) where W <: TensType
  return environment{W}(network(copy(network.net)))
end

"""
    V = environment(P)

Inputs tensors `P` representing an `Array` of tensors into environment `V`

See also: [`Array`](@ref)
"""
function environment(input::Array{W,1}) where W <: TensType
  return environment{W}(network(copy(input)))
end