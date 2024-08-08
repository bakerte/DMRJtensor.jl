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
    mult!(psi,c)

takes input MPS `psi` and multiplies orthogonality center by a number `c`. Some number types require conversaion of `psi`

See also: [`convertTensor`](@ref) [`MPS`](@ref)
"""
function mult!(psi::MPS,num::Number)
  psi[psi.oc] = mult!(psi[psi.oc],num)
  return psi
end

"""
    mult!(c,psi)

takes input MPS `psi` and multiplies orthogonality center by a number `c`. Some number types require conversaion of `psi`

See also: [`convertTensor`](@ref) [`MPS`](@ref)
"""
function mult!(num::Number,psi::MPS)
  return mult!(psi,num)
end
