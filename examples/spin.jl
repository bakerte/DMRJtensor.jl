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


path = "../../"
include(path*"DMRjulia.jl")
using .DMRJtensor

Ns = 100
spinmag = 0.5

hereQS = convert(Int64,2*spinmag+1)

initTensor = [zeros(1,hereQS,1) for i=1:Ns]
for i = 1:Ns
   initTensor[i][1,i%2 == 1 ? 1 : 2,1] = 1.0
end

psi = MPS(initTensor)

Sp,Sm,Sz = spinOps(s=spinmag)

function makeheisenberg(Ns,Sp,Sm,Sz)
  mpo = 0
  for j in [1] #,4] #j-ranged interaction
    for i = 1:Ns-j
      mpo += mpoterm(0.5,Sp,i,Sm,i+j)
      mpo += mpoterm(0.5,Sm,i,Sp,i+j)
      mpo += mpoterm(Sz,i,Sz,i+j)
      #add higher than quadratic terms like
      #mpo += mpoterm(Sp,i,Sm,i+j,Sz,i+1,Sz,i+j)
    end
  end

  #=
  #single site terms
  for i = 1:Ns
    mpo += mpoterm(0.3,Sz,i)
  end
  =#

  return MPO(mpo)
end

mpo = makeheisenberg(Ns,Sp,Sm,Sz)

#Quantum number specification
@makeQNs "spin" U1
Qlabels = [[spin(1),spin(-1)]]

qpsi,qmpo = MPS(Qlabels,psi,mpo)
#qmpo,qpsi = MPO(Qlabels,mpo,psi)

#mpo += expMPO(exp(-1/2.0),Sz,Sz,Ns) #to add an exponential interaction between all sites

println("#############")
println("QN version")
println("#############")

@time energyQN = dmrg(qpsi,qmpo,maxm=45,sweeps=20,cutoff=1E-9,method="twosite")

println("#############")
println("nonQN version")
println("#############")

@time energy = dmrg(psi,mpo,maxm=45,sweeps=20,cutoff=1E-9,method="twosite")

