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

#       +--------------------------------------+
#>------|    (bad) Methods for excitations     |---------<
#       +--------------------------------------+

"""
    penalty!(mpo,lambda,psi[,compress=])

Adds penalty to Hamiltonian (`mpo`), H0, of the form H0 + `lambda` * |`psi`><`psi`|; toggle to compress resulting wavefunction

See also: [`penalty`](@ref)
"""
function penalty!(mpo::MPO,lambda::Float64,psi::MPS;compress::Bool=true)
  for i = 1:length(psi)
    QS = size(psi[i],2)
    R = eltype(mpo[i])
    temp_psi = reshape(psi[i],size(psi[i])...,1)
    if i == psi.oc
      term = contractc(temp_psi,4,temp_psi,4,alpha=lambda)
    else
      term = contractc(temp_psi,4,temp_psi,4)
    end

    rho = reshape!(term,[[1,4],[5],[2],[3,6]],merge=true)

    if i == 1
      mpo[i] = joinindex!(4,mpo[i],rho)
    elseif i == length(psi)
      mpo[i] = joinindex!(1,mpo[i],rho)
    else
      mpo[i] = joinindex!([1,4],mpo[i],rho)
    end
  end

  return mpo #compress ? compressMPO!(mpo) : mpo
end
export penalty!

"""
    penalty(mpo,lambda,psi[,compress=])

Same as `penalty` but makes a copy of `mpo`

See also: [`penalty!`](@ref)
  """
function penalty(mpo::MPO,lambda::Float64,psi::MPS;compress::Bool=true)
  newmpo = copy(mpo)
  return penalty!(newmpo,lambda,psi,compress=compress)
end
export penalty