#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v1.0
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.1+)
#

using DMRJtensor

#As of v0.17.0, there has been a revision of the automatic MPO generator in the code
#This example will explain how to use it

Ns = 4

mu = -2.0
HubU = 4.0
t = 1.0

#In this new version, we need the local operators for each site in either the regular or quantum number system

Cup,Cdn,F,Nup,Ndn,Ndens,O,Id = fermionOps()

@makeQNs "fermion" U1 U1
Qlabels = [fermion(0,0),fermion(1,1),fermion(1,-1),fermion(2,0)]

#So, for quantum number conservation, we also need the operators

qCup,qCdn,qF,qNup,qNdn,qNdens,qO,qId = Qtens(Qlabels,Cup,Cdn,F,Nup,Ndn,Ndens,O,Id)

#This function is written generically so that it will handle either quantum number tensors or regular ones
function autoHubbard(Ns,Ops...)

  Cup,Cdn,F,Nup,Ndn,Ndens,O,Id = Ops

  mpo = 0
  for i = 1:Ns-1
    mpo += mpoterm(-t,Cup,i,Cup',i+1,F)
    mpo += mpoterm(t,Cup',i,Cup,i+1,F)
    mpo += mpoterm(-t,Cdn,i,Cdn',i+1,F)
    mpo += mpoterm(t,Cdn',i,Cdn,i+1,F)
  end
  for i = 1:Ns
    mpo += mpoterm(mu,Ndens,i)
    mpo += mpoterm(HubU,Nup*Ndn,i)
  end
  return MPO(mpo) #This is where the argument strings are collected and processed into a (nearly) optimal bond dimension MPO
end

#This is two separate calls. The first gives the dense tensor implementation
@time mpo = autoHubbard(Ns,Cup,Cdn,F,Nup,Ndn,Ndens,O,Id)
#The second gives the quantum number version
@time qmpo = autoHubbard(Ns,qCup,qCdn,qF,qNup,qNdn,qNdens,qO,qId)

using TensorPACK
#If you like, you can see that the models give the same arguments
D,U = eigen(mpo)
qD,qU = eigen(qmpo)

#Increasing the system size too much will lead to a segfault depending on how much memory you have. So, switching to DMRG is a good idea if the system size is too large. See the Hubbard model example for how to initialize the Hubbard model wavefunction for different particle numbers

println(norm(sort([qD[w,w] for w = 1:size(qD,1)])-D.T))

#Note: It is not advised to make the dense MPO and then convert it to the quantum number version. The compression step can yield some important elements out of the quantum number blocks. There is a fix for this, but it is just as easy to ask the user to implement the quantum number symmetries onto the local operators and compress with those.

#Enjoy!