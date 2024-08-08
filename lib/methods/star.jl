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
    *(psi,c)

makes copy of input MPS `psi` and multiplies orthogonality center by a number `c`. Some number types require conversaion of `psi`

See also: [`convertTensor`](@ref) [`MPS`](@ref)
"""
function *(psi::MPS,num::Number)
  return mult!(copy(psi),num)
end

"""
    *(c,psi)

makes copy of input MPS `psi` and multiplies orthogonality center by a number `c`. Some number types require conversaion of `psi`

See also: [`convertTensor`](@ref) [`MPS`](@ref)
"""
function *(num::Number,psi::MPS)
  return *(psi,num)
end

#import Base.push!
"""
    *(X,Y)

concatenates two MPSs `X` and `Y` (same as vcat in Base Julia)
"""
function *(X::MPS,Y::MPS)
  return MPS(vcat(X.A,Y.A),oc=0)
end