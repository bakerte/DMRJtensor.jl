#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8.3
#
#########################################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.1) or (v1.5)
#

path = "../src/"
include(path*"DMRjulia.jl")
using .DMRJtensor

#create reduced site problem (step 2)
function make2site(Lenv,Renv,psiL,psiR,mpoL,mpoR)
  Lpsi = contract(Lenv,[3],psiL,[1])
  LHpsi = contract(Lpsi,[2,3],mpoL,[1,2])
  LHpsipsi = contract(LHpsi,[2],psiR,[1])
  LHHpsipsi = contract(LHpsipsi,[3,4],mpoR,[1,2])
  HpsiLR = contract(LHHpsipsi,[3,5],Renv,[1,2])
  return HpsiLR
end


function simplelanczos(iL,iR,Lenv,Renv,psi,mpo;betatest = 1E-10)
  Hpsi = make2site(Lenv[iL],Renv[iR],psi[iL],psi[iR],mpo[iL],mpo[iR])
  AA = contract(psi[iL],3,psi[iR],1)
  AA = div!(AA,norm(AA))
  alpha1 = real(ccontract(AA,Hpsi))

  psi2 = sub!(Hpsi, AA, alpha1)
  beta1 = norm(psi2) #sqrt(real(ccontract(psi2, psi2)))
  if beta1 > betatest
    psi2 = div!(psi2, beta1)

    Hpsi2 = contract(Lenv[iL],3,psi2,1)
    ops = contract(mpo[iL],4,mpo[iR],1)
    Hpsi2 = contract(Hpsi2,[2,3,4],ops,[1,2,4])
    Hpsi2 = contract(Hpsi2,[2,5],Renv[iR],[1,2])

    alpha2 = real(ccontract(psi2,Hpsi2))
    M = [alpha1 beta1; beta1 alpha2]
    D, U = eigen(M)
    energy = D[1,1]
    outAA = conj(U[1,1])* AA + conj(U[2,1])*psi2
  else
    energy = alpha1
    outAA = AA
  end
  return outAA,energy
end

function simpledmrg(psi,mpo;m=20,sweeps=10,cutoff=1E-9)
  Lenv,Renv = makeEnv(psi,mpo)  #make environments
  j = 1
  for n = 1:sweeps
    for s = 1:2*(length(psi)-1)
      i = psi.oc

     #select sites to form renormalized problem (step 1)
      if j > 0
        iL = i
        iR = i + 1
      else
        iL = i - 1
        iR = i
      end

      outAA,energy = simplelanczos(iL,iR,Lenv,Renv,psi,mpo) #Lanczos (step 3)

      if cld(length(psi),2) == psi.oc
        println("sweep = $n, site = $(psi.oc), energy = ",energy[1])
        println()
      end

      U,D,V = svd(outAA,[[1,2],[3,4]],m=m,cutoff=cutoff) #SVD (step 4)

      if j > 0
        psi[iL] = U
        psi[iR] = contract(D,2,V,1)
        Lenv[iR] = Lupdate(Lenv[iL],psi[iL],psi[iL],mpo[iL])  #update environments (step 5)
      else
        psi[iL] = contract(U,3,D,1)
        psi[iR] = V
        Renv[iL] = Rupdate(Renv[iR],psi[iR],psi[iR],mpo[iR])  #update environments (step 5)
      end
      psi.oc += j
        
      if j > 0 && psi.oc == length(psi)
          j *= -1
      elseif j < 0 && psi.oc == 1
          j *= -1
      end
    end
  end
end

Ns = 100
spinmag = 0.5

physindsize = convert(Int64,2*spinmag+1)

initTensor = [zeros(1,2,1) for i=1:Ns]
for i = 1:Ns
   initTensor[i][1,i%2 == 1 ? 1 : 2,1] = 1.0
end
psi = MPS(initTensor,oc=1)

Sp,Sm,Sz,Sy,Sx,O,Id = spinOps(s=spinmag)
H = [Id O O O O;
     Sp O O O O;
     Sm O O O O;
     Sz O O O O;
     O Sm/2 Sp/2 Sz Id]

mpo = makeMPO(H,physindsize,Ns)

println("#############")
println("simple DMRG (dense version)")
println("#############")

simpledmrg(psi,mpo,sweeps=20,m=45,cutoff=1E-9)
