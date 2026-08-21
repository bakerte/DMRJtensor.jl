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
    S = SvN(psi)

Computes the entropy `S` from an MPS `psi` gauged to site `psi.oc`

Note: DMRjulia labels the bonds in the MPS to the right of the MPS's orthogonality centre. The default for any higher order terms is to take the first r/2+1 (rounded up) terms and generate the SVD with the remaining terms
"""
function SvN(psi::MPS)
#  if ndims(psi[psi.oc]) 
  centredim = cld(ndims(psi[psi.oc]),2)
  ldims = [w for w = 1:centredim]
  rdims = [w for w = centredim+1:ndims(psi[psi.oc])]

  U,D,V = svd(psi[psi.oc],[ldims,rdims])
  SvN = 0
  @inbounds @simd for w = 1:length(D)
    SvN -= D[w]^2 * log(D[w]^2)
  end
  return SvN
end

"""
    S = SvN(psi,oc)

Computes the entropy `S` from an MPS `psi` which is gauged to site `oc` (creating a copy of `psi`)
"""
function SvN(psi::MPS,oc::Integer)
  newpsi = move(psi,oc)
  return SvN(newpsi)
end

"""
    S = SvN!(psi,oc)

Computes the entropy `S` from an MPS `psi` which is gauged to site `oc` (moves `psi` in-place)
"""
function SvN!(psi::MPS,oc::Integer)
  move!(psi,oc)
  return SvN(psi)
end
