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
    B = get_tensors(A)

Unified function for obtaining the vector of tensors from an `MPS`
"""
function get_tensors(input::MPS)
  return input.A
end

"""
    B = get_tensors(A)

Unified function for obtaining the vector of tensors from an `MPO`
"""
function get_tensors(input::MPO)
  return input.H
end

"""
    B = get_tensors(A)

Unified function for obtaining the vector of tensors from an `MERA`
"""
function get_tensors(input::MERA)
  return input.H
end