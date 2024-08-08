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
    Rupdate(Renv,dualpsi,psi,mpo)

Updates right environment tensor `Renv` with dual MPS `dualpsi`, MPS `psi`, and MPO `mpo`
"""
 function  Rupdate(Renv::TensType,dualpsi::TensType,psi::TensType,mpo::TensType...)
  nMPOs = length(mpo)
  nLsize = nMPOs+2
  nRsize = nLsize+1
  tempRenv = ccontract(dualpsi,3,Renv,nLsize)
  for j = 1:nMPOs
    tempRenv = contract(mpo[j],(3,4),tempRenv,(2,nRsize))
  end
  return contract(psi,(2,3),tempRenv,(2,nRsize))
end
export Rupdate

"""
    Rupdate!(i,Renv,dualpsi,psi,mpo)

Updates right environment's (`Renv`) `i`-1 site from info on the site `i` based on dual MPS (`dualpsi`), MPS (`psi), and MPO (`mpo`)
"""
 function Rupdate!(i::Integer,Renv::Env,dualpsi::MPS,psi::MPS,mpo::MPO...)
  Renv[i-1] = Rupdate(Renv[i],dualpsi[i],psi[i],[mpo[a][i] for a = 1:length(mpo)]...)
  nothing
end

"""
    Rupdate!(i,Renv,psi,mpo)

Updates right environment's (`Renv`) `i`-1 site from info on the site `i` based on MPS (`psi) and MPO (`mpo`)
"""
 function Rupdate!(i::Integer,Renv::Env,psi::MPS,mpo::MPO...)
  Rupdate!(i,Renv,psi,psi,mpo...)
end
export Rupdate!