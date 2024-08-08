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
    B = psi[end]

`lastindex!` allows to get the end element of an `MPS`
"""
function lastindex(A::regMPS)
  return lastindex(A.A)
end

"""
    B = psi[end]

`lastindex!` allows to get the end element of an `MPO`
"""
function lastindex(H::regMPO)
  return lastindex(H.H)
end

"""
    B = psi[end]

`lastindex!` allows to get the end element of an `Env`
"""
function lastindex(G::regEnv)
  return lastindex(G.V)
end

