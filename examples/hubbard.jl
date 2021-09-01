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


path = "../src/"
include(path*"DMRjulia.jl")

Ns = 10

@makeQNs "fermion" U1 U1
Qlabels = [[fermion(0,0),fermion(1,1),fermion(1,-1),fermion(2,0)]]

Ne = Ns
Ne_up = ceil(Int64,div(Ne,2))
Ne_dn = Ne-Ne_up
QS = 4
Cup, Cdn, Nup, Ndn, Ndens, F, O, Id = fermionOps()

psi = MPS(QS,Ns)
upsites = [i for i = 1:2:Ns]
Cupdag = Matrix(Cup')
applyOps!(psi,upsites,Cupdag,trail=F)


dnsites = [i for i = 2:2:Ns]
Cdndag = Matrix(Cdn')
applyOps!(psi,dnsites,Cdndag,trail=F)

qpsi = makeqMPS(psi,Qlabels)

mu = -2.0
HubU = 4.0
t = 1.0

function H(i::Int64)
onsite(i::Int64) = mu * Ndens + HubU * Nup * Ndn #- Ne*exp(-abs(i-Ns/2)/2)*Ndens
        return [Id  O O O O O;
            -t*Cup' O O O O O;
            conj(t)*Cup  O O O O O;
            -t*Cdn' O O O O O;
            conj(t)*Cdn  O O O O O;
            onsite(i) Cup*F Cup'*F Cdn*F Cdn'*F Id]
    end

println("Making qMPO")

@time mpo = makeMPO(H,QS,Ns)
@time qmpo = makeqMPO(mpo,Qlabels)


println("#############")
println("QN version")
println("#############")

QNenergy = dmrg(qpsi,qmpo,maxm=45,sweeps=20,cutoff=1E-9)


println("#############")
println("nonQN version")
println("#############")

energy = dmrg(psi,mpo,maxm=45,sweeps=20,cutoff=1E-9)

println(QNenergy-energy)

