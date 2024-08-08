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
    mpo[i] = G

`setindex!` allows to assign elements `G` to an `MPO` at element `i`
"""
function setindex!(H::regMPO,A::TensType,i::intType)
  H.H[i] = A
  nothing
end

"""
    psi[i] = G

`setindex!` allows to assign elements `G` to an `MPS` at element `i`
"""
function setindex!(H::regMPS,A::TensType,i::intType)
  H.A[i] = A
  nothing
end

"""
    Lenv[i] = G

`setindex!` allows to assign elements `G` to an `Env` at element `i`
"""
function setindex!(G::regEnv,A::TensType,i::intType)
  G.V[i] = A
  nothing
end

"""
    Lenv[i] = G

`setindex!` allows to assign elements `G` to an `largeMPS` at element `i`
"""
function setindex!(H::largeMPS,A::TensType,i::intType;ext::String=file_extension)
  tensor2disc(H.A[i],A,ext=ext)
  nothing
end

"""
    Lenv[i] = G

`setindex!` allows to assign elements `G` to an `largeMPO` at element `i`
"""
function setindex!(H::largeMPO,A::TensType,i::intType;ext::String=file_extension)
  tensor2disc(H.H[i],A,ext=ext)
  nothing
end

"""
    Lenv[i] = G

`setindex!` allows to assign elements `G` to an `largeEnv` at element `i`
"""
function setindex!(H::largeEnv,A::TensType,i::intType;ext::String=file_extension)
  tensor2disc(H.V[i],A,ext=ext)
  nothing
end
