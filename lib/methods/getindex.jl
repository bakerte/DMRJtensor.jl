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
    G = getindex(A,i)

`getindex` allows to retrieve tensor `G` at position `i` from `MPS`
"""
function getindex(A::regMPS,i::Integer)
  return A.A[i]
end

"""
    G = getindex(A,i)

`getindex` allows to retrieve tensor `G` at position `i` from `MPO`
"""
function getindex(H::regMPO,i::Integer)
  return H.H[i]
end

"""
    G = getindex(A,i)

`getindex` allows to retrieve tensor `G` at position `i` from `Env`
"""
function getindex(G::regEnv,i::Integer)
  return G.V[i]
end

"""
    G = getindex(A,i)

`getindex` allows to retrieve tensor `G` at position `i` from `MERA`
"""
function getindex(H::MERA,i::Integer)
  return H.H[i]
end

"""
    G = getindex(A,r)

`getindex` allows to retrieve a range `r` (ex: r = 2:6) tensor `G` from `MPS`
"""
function getindex(A::regMPS,r::UnitRange{W}) where W <: Integer
  if A.oc in r
    newoc = findfirst(w->w == A.oc,r)
  else
    newoc = 0
  end
  return MPS(A.A[r],oc=newoc)
end

"""
    G = getindex(A,r)

`getindex` allows to retrieve a range `r` (ex: r = 2:6) tensor `G` from `MPO`
"""
function getindex(H::regMPO,r::UnitRange{W}) where W <: Integer
  return MPO(H.H[r])
end

"""
    G = getindex(A,r)

`getindex` allows to retrieve a range `r` (ex: r = 2:6) tensor `G` from `Env`
"""
function getindex(G::regEnv,r::UnitRange{W}) where W <: Integer
  return environment(G.V[r])
end

"""
    G = getindex(A,i)

`getindex` allows to retrieve tensor `G` at position `i` from `largeMPS`
"""
function getindex(A::largeMPS,i::Integer)
  return tensorfromdisc(A.A[i],ext=file_extension)
end

"""
    G = getindex(A,i)

`getindex` allows to retrieve tensor `G` at position `i` from `largeMPO`
"""
function getindex(A::largeMPO,i::Integer)
  return tensorfromdisc(A.H[i],ext=file_extension)
end

"""
    G = getindex(A,i)

`getindex` allows to retrieve tensor `G` at position `i` from `largeEnv`
"""
function getindex(A::largeEnv,i::Integer)
  return tensorfromdisc(A.V[i],ext=file_extension)
end
