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
    H = trace(mpo)

Takes the trace of the full MPO representation of the Hamiltonian
"""
function trace(mpo::MPO)
  tr = trace(mpo[1],[2,3])
  for w = 2:Ns
    tr *= mpo[w]
    tr = trace(tr,[2,3])
  end
  return trace(tr)
end
#export fullH
