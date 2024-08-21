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
using TensorPACK

println("I recommend using tdvp_twosite")


Ns = 10

mu = -2.0
HubU = 4.0
t = 1.0

Cup,Cdn,F,Nup,Ndn,Ndens,O,Id = fermionOps()

function H(i::Int64)
        onsite = mu * Ndens + HubU * Nup * Ndn #- Ne*exp(-abs(i-Ns/2)/2)*Ndens
        return [Id  O O O O O;
            -t*Cup' O O O O O;
            conj(t)*Cup  O O O O O;
            -t*Cdn' O O O O O;
            conj(t)*Cdn  O O O O O;
            onsite Cup*F Cup'*F Cdn*F Cdn'*F Id]
    end
mpo = makeMPO(H,4,Ns)

Ne = Ns
Ne_up = ceil(Int64,div(Ne,2))
Ne_dn = Ne-Ne_up
QS = 4

psi = MPS(QS,Ns)
upsites = [i for i = 1:Ne_up]
Cupdag = Matrix(Cup')
applyOps!(psi,upsites,Cupdag,trail=F)


dnsites = [i for i = 1:Ne_dn]
Cdndag = Matrix(Cdn')
applyOps!(psi,dnsites,Cdndag,trail=F)

@makeQNs "fermion" U1 U1
Qlabels = [fermion(0,0),fermion(1,1),fermion(1,-1),fermion(2,0)]
qpsi,qmpo = MPS(Qlabels,psi,mpo)

energy = dmrg(psi,mpo,sweeps=100,m=100,cutoff=1E-9,silent=true)
QNenergy = dmrg(qpsi,qmpo,sweeps=100,m=100,cutoff=1E-9,silent=true)


psi0 = deepcopy(psi)
qpsi0 = deepcopy(qpsi)

ntime = 500
deltaT = 0.001




println("regular evolution")

function tMPO(i::Int64)
onsite(i::Int64) = (mu * Ndens + HubU * Nup * Ndn)*(i!=1 && i!= Ns ? 0.5 : 1)
    return [Id  O O O O O;
            Cup' O O O O O;
            Cup  O O O O O;
            Cdn' O O O O O;
            Cdn  O O O O O;
            onsite(i) -t*Cup*F conj(t)*Cup'*F -t*Cdn*F conj(t)*Cdn'*F Id]
end

tH = makeMPO(tMPO,4,Ns) #this form of the MPO is more stable for time evolution (factors of 1/2)

expgates = makeExpGates(tH,-im*deltaT)
phiT = Array{ComplexF64,1}(undef,ntime)
for j = 1:ntime
  global psi = tebd(psi,expgates,cutoff = 1E-8,m = 50)
  phiT[j] = expect(psi0,psi)
end
expT = phiT[ntime]


println("QN evolution")



times = [n*deltaT for n = 1:ntime]

checktimes = [n*deltaT for n = 1:ntime]
checkexp = [exp(-im*n*deltaT*energy) for n = 1:ntime]

tqMPO = makeqMPO(Qlabels,tH)

qexpgates = qmakeExpGates(tH,Qlabels,-im*deltaT)
qphiT = Array{ComplexF64,1}(undef,ntime)
for j = 1:ntime
  global qpsi = tebd(qpsi,qexpgates,cutoff = 1E-8,m = 50)
  qphiT[j] = expect(qpsi0,qpsi)
end
qexpT = qphiT[ntime]

finalE = expect(psi,psi,mpo)
QNfinalE = expect(qpsi,qpsi,qmpo)



#=
using Plots
plot(real.(phiT))
plot!(imag.(phiT))

plot!(real.(qphiT))
plot!(imag.(qphiT))
=#
println("See code for plotting checks")
