#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.1) or (v1.5)
#


using DMRjulia

Ns = 10 #number of sites in MPS

@makeQNs "fermion" U1 U1 #general format for quantum number initialization: U1 and Zn{x} also defined
Qlabels = [[fermion(0,0),fermion(1,1),fermion(1,-1),fermion(2,0)]] #quantum number labels for Hubbard model

Ne = 10 #number of electrons
Ne_up = ceil(Int64,div(Ne,2)) #number of up electrons
Ne_dn = Ne-Ne_up #number of down electron
QS = 4 #size of local Fock space
Cup, Cdn, Nup, Ndn, Ndens, F, O, Id = fermionOps() #operators for a Hubbard model...also available in that file are t-J and spin



#generates initial wavefunction tensors
initTensor = [zeros(1,QS,1) for i=1:Ns]
for i = 1:Ns
   initTensor[i][1,1,1] = 1.0
end

for i = 1:Ne_up
  initTensor[i] = contract([2,1,3],Matrix(Cup'),2,initTensor[i],2)
  for j = 1:i-1
    initTensor[j] = contract([2,1,3],Matrix(F),2,initTensor[j],2)
  end
end
for i = 1:Ne_dn
  initTensor[i] = contract([2,1,3],Matrix(Cdn'),2,initTensor[i],2)
  for j = 1:i-1
    initTensor[j] = contract([2,1,3],Matrix(F),2,initTensor[j],2)
  end
end

psi = MPS(initTensor,1) #MPS
qpsi = makeqMPS(psi,Qlabels) #quantum number MPS

mu = -2.0 #chemical potential
HubU = 4.0 #Hubbard U
t = 1.0 #hopping

#makes MPO representation as it appears on the page
function H(i::Int64)
onsite(i::Int64) = mu * Ndens + HubU * Nup * Ndn
        return [Id  O O O O O;
            -t*Cup' O O O O O;
            conj(t)*Cup  O O O O O;
            -t*Cdn' O O O O O;
            conj(t)*Cdn  O O O O O;
            onsite(i) Cup*F Cup'*F Cdn*F Cdn'*F Id]
    end

println("Making qMPO")

@time mpo = convert2MPO(H,QS,Ns) #converts matrix to MPO
@time qmpo = makeqMPO(mpo,Qlabels) #makes quantum number MPO


println("#############")
println("QN version")
println("#############")

QNenergy = dmrg(qpsi,qmpo,maxm=45,sweeps=20,cutoff=1E-9)


println("#############")
println("nonQN version")
println("#############")

energy = dmrg(psi,mpo,maxm=45,sweeps=20,cutoff=1E-9)

println(QNenergy-energy)


move!(psi,4) #moves MPS to site 4

