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
    vect = fullpsi(psi)

Generates the full wavefunction `vect` from an MPS `psi` (memory providing)
"""
function fullpsi(psi::MPS)
  Ns = length(psi)
  fullpsi = psi[1]
  for p = 2:Ns
    fullpsi = contract(fullpsi,ndims(fullpsi),psi[p],1)
  end
  return reshape!(fullpsi,prod(size(fullpsi)))
end
export fullpsi
