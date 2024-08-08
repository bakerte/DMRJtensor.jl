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
    G = size(H)

`size` prints out the size tuple `G` of the tensor field of a `MPS`; this is effectively the number of sites in a tuple
"""
function size(psi::MPS)
  return size(psi.A)
end

"""
    G = size(H)

`size` prints out the size tuple `G` of the tensor field of a `MPO`; this is effectively the number of sites in a tuple
"""
 function size(H::MPO)
  return size(H.H)
end

"""
    G = size(H)

`size` prints out the size tuple `G` of the tensor field of a `Env`; this is effectively the number of sites in a tuple
"""
function size(G::regEnv)
  return size(G.V)
end

"""
    G = size(H,i)

`size` prints out the size tuple `G` of the tensor field of a `MPS`; this is effectively the number of sites in a tuple
"""
function size(psi::MPS,i::Integer)
  return size(psi.A[i])
end

"""
    G = size(H,i)

`size` prints out the size tuple `G` of the tensor field of a `MPO`; this is effectively the number of sites in a tuple
"""
function size(H::MPO,i::Integer)
  return size(H.H[i])
end

"""
    G = size(H,i)

`size` prints out the size tuple `G` of the tensor field of a `Env`; this is effectively the number of sites in a tuple
"""
function size(G::regEnv,i::Integer)
  return size(G.V[i])
end



function size(A::network)
  return size(A.net)
end
