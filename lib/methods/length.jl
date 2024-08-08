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
    G = length(H)

`length` prints out the number of entries of the input `H` being a `MPO`; this is effecitvely the number of sites
"""
function length(H::MPO)
  return length(H.H)
end

"""
    G = length(H)

`length` prints out the number of entries of the input `H` being a `MPS`; this is effecitvely the number of sites
"""
function length(psi::MPS)
  return length(psi.A)
end

"""
    G = length(H)

`length` prints out the number of entries of the input `H` being a `MERA`; this is  the number of tensors in the layer of isometries or unitaries
"""
function length(psi::MERA)
  return length(psi.H)
end

"""
    G = length(H)

`length` prints out the number of entries of the input `H` being a `Env`; this is effecitvely the number of sites
"""
function length(G::regEnv)
  return length(G.V)
end

"""
    G = length(H)

`length` prints out the number of entries of the input `H` being a `largeMPS`; this is effecitvely the number of sites
"""
function length(A::largeMPS)
  return length(A.A)
end

"""
    G = length(H)

`length` prints out the number of entries of the input `H` being a `largeMPO`; this is effecitvely the number of sites
"""
function length(H::largeMPO)
  return length(H.H)
end

"""
    G = length(H)

`length` prints out the number of entries of the input `H` being a `largeEnv`; this is effecitvely the number of sites
"""
function length(H::largeEnv)
  return length(H.V)
end
