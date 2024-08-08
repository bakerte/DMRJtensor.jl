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
    psi = conj!(psi)

Conjugates all elements in an `MPS` in-place; outputs `psi` which was input

See also [`conj`](@ref)
"""
function conj!(A::regMPS)
  conj!.(A.A.net)
  return A
end

#import Base.conj
"""
    newpsi = conj(psi)

Conjugates all elements in an `MPS` and makes a copy `psi`

See also [`conj!`](@ref)
"""
function conj(A::regMPS)
  B = copy(A)
  conj!(B.A)
  return B
end