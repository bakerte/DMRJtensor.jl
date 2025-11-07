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
    rho = correlationmatrix(dualpsi,psi,Cc,Ca[,F,silent=])

Compute the correlation funciton (example, <`dualpsi`|`Cc`_i . `Ca`_j|`psi`>) on all sites; can also specify a trail `F` for Fermion strings; `silent` toggles output of every line; outputs matrix `rho`

# Note:
+ More efficient than using `mpoterm`s or `correlation`
+ Use `mpoterm` and `applyMPO` for higher order correlation functions or write a new function
"""
function correlationmatrix(dualpsi::MPS, psi::MPS, Cc::TensType, Ca::TensType; trail=intType[])
  rho = Array{eltype(psi[1]),2}(undef,length(psi),length(psi))
  if trail != []
    FCc = contract(Cc,2,trail,1)
  else
    FCc = Cc
  end
  diffTensors = !(psi == dualpsi)
  onsite = contract(Cc,2,Ca,1)
  for i = 1:length(psi)
    move!(psi,i)
    if diffTensors
      move!(dualpsi,i)
    end
    TopTerm = contract([2,1,3],onsite,2,psi[i],2)
    rho[i,i] = contractc(TopTerm,dualpsi[i])
  end
  for i = 1:length(psi)-1
    move!(psi,i)
    if diffTensors
      move!(dualpsi,i)
    end
    TopTerm = contract(FCc,2,psi[i],2)
    Lenv = contractc(TopTerm,(2,1),dualpsi[i],(1,2))
    for j = i+1:length(psi)
      Renv = contract(Ca,2,psi[j],2)
      Renv = contractc(Renv,(1,3),dualpsi[j],(2,3))
      DMElement = contract(Lenv,Renv)
      if j < length(psi)
        if trail != []
          Lenv = contract(Lenv,1,psi[j],1)
          Lenv = contract(Lenv,2,trail,2)
          Lenv = contractc(Lenv,(1,3),dualpsi[j],(1,2))
        else
          Lenv = contract(Lenv, 1, psi[j], 1)
          Lenv = contractc(Lenv, (1,2), dualpsi[j], (1,2))
        end
      end
      rho[i,j] = DMElement
      rho[j,i] = conj(DMElement)
    end
  end
  return rho
end

"""
    A = correlationmatrix(psi,Cc,Ca[,F,silent=])

Compute the correlation funciton (example, <`psi`|`Cc`_i . `Ca`_j|`psi`>) on all sites; can also specify a trail `F` for Fermion strings; `silent` toggles output of every line; outputs matrix `rho`

# Note:
+ More efficient than using `mpoterm`s or `correlation`
+ Use `mpoterm` and `applyMPO` for higher order correlation functions or write a new function

# Example:
```julia
Cup, Cdn, Nup, Ndn, Ndens, F, O, Id = fermionOps()
rho = correlationmatrix(psi,Cup',Cup,F) #density matrix
```
"""
function correlationmatrix(psi::MPS, Cc::TensType, Ca::TensType; trail=[])
  return correlationmatrix(psi,psi,Cc,Ca,trail=trail)
end
export correlationmatrix