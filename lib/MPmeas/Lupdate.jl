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
    Lupdate(Lenv,dualpsi,psi,mpo)

Updates left environment tensor `Lenv` with dual MPS `dualpsi`, MPS `psi`, and MPO `mpo`
"""
 function  Lupdate(Lenv::TensType,dualpsi::TensType,psi::TensType,mpo::TensType...)
  nMPOs = length(mpo)
  nLsize = nMPOs+2
  tempLenv = contractc(Lenv,1,dualpsi,1)

  for j = 1:nMPOs
    tempLenv = contract(tempLenv,(1,nLsize),mpo[j],(1,3))
  end
  return contract(tempLenv,(1,nLsize),psi,(1,2))
end
export Lupdate

"""
    Lupdate!(i,Lenv,psi,dualpsi,mpo)

Updates left environment's (`Lenv`) `i`+1 site from info on the site `i` based on dual MPS (`dualpsi`), MPS (`psi), and MPO (`mpo`)
"""
 function Lupdate!(i::Integer,Lenv::Env,dualpsi::MPS,psi::MPS,mpo::MPO...)
  Lenv[i+1] = Lupdate(Lenv[i],dualpsi[i],psi[i],[mpo[a][i] for a = 1:length(mpo)]...)
  nothing
end

"""
    Lupdate!(i,Lenv,psi,mpo)

Updates left environment's (`Lenv`) `i`+1 site from info on the site `i` based on MPS (`psi) and MPO (`mpo`)
"""
 function Lupdate!(i::Integer,Lenv::Env,psi::MPS,mpo::MPO...)
  Lupdate!(i,Lenv,psi,psi,mpo...)
end
export Lupdate!