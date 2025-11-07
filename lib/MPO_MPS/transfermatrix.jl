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
    transfermatrix(dualpsi,psi,i,j[,transfermat=])

Forms the transfer matrix (an MPS `psi` and its dual `dualpsi` contracted along the physical index) between sites `i` and `j` (inclusive). If not specified, the `transfermat` field will initialize to the transfer matrix from the `i-1` site

The form of the transfer matrix must be is as follows (dual wavefunction tensor on top, conjugated)

1 ------------ 3
        |
        |
        |
        |
2 ------------ 4

There is no in-place version of this function

"""
function transfermatrix(dualpsi::MPS,psi::MPS,i::Integer,j::Integer;startsite::Integer=i,transfermat::TensType=ccontract([1,3,2,4],dualpsi[startsite],2,psi[startsite],2))
  for k = startsite+1:j
    transfermat = contractc(transfermat,3,dualpsi[k],1)
    transfermat = contract(transfermat,(3,4),psi[k],(1,2))
  end
  return transfermat
end

"""
    transfermatrix(psi,i,j[,transfermat=])

Forms the transfer matrix (an MPS `psi`) between sites `i` and `j` (inclusive). If not specified, the `transfermat` field will initialize to the transfer matrix from the `i-1` site

The form of the transfer matrix must be is as follows (dual wavefunction tensor on top, conjugated)

1 ------------ 3
        |
        |
        |
        |
2 ------------ 4

There is no in-place version of this function

"""
function transfermatrix(psi::MPS,i::Integer,j::Integer;startsite::Integer=i,transfermat::TensType=ccontract([1,3,2,4],psi[startsite],2,psi[startsite],2))
  return transfermatrix(psi,psi,i,j,transfermat=transfermat)
end
export transfermatrix
