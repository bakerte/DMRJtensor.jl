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

Ns = 10

@makeQNs "fermion" U1 U1
Qlabels = [[fermion(0,0),fermion(1,1),fermion(1,-1),fermion(2,0)]]

Ne = Ns
Ne_up = ceil(Int64,div(Ne,2))
Ne_dn = Ne-Ne_up
QS = 4
Cup,Cdn,F,Nup,Ndn,Ndens,O,Id = fermionOps()

psi = MPS(QS,Ns)
upsites = [i for i = 1:2:Ne_up]
Cupdag = Matrix(Cup')
applyOps!(psi,upsites,Cupdag,trail=F)


dnsites = [i for i = 2:2:Ne_dn]
Cdndag = Matrix(Cdn')
applyOps!(psi,dnsites,Cdndag,trail=F)

qpsi = makeqMPS(Qlabels,psi)

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




nsteps = 100
DeltaT = -im*0.001

println("#############")
println("nonQN version")
println("#############")

energy = dmrg(psi,mpo,m=100,sweeps=200,goal=1E-10,cutoff=1E-9)

tpsi = MPS(ComplexF64,psi)
tmpo = MPO(ComplexF64,mpo)

time_exp = Array{ComplexF64,1}(undef,nsteps+1)
energies = similar(time_exp)

time_exp[1] = 1.
energies[1] = energy

for i = 1:nsteps
  tdvp_twosite(tpsi,tmpo,prefactor=DeltaT,m=100)
  time_exp[i+1] = expect(tpsi,psi)
  energies[i+1] = expect(tpsi,tmpo)
  println(i," ",time_exp[i+1]," ",exp(-i*DeltaT*energy)," ",energies[i+1])
end

true_exp = [exp(-(i-1)*DeltaT*energy) for i = 1:nsteps+1]

#=
using Plots
plot(imag.(time_exp))
plot!(imag.(true_exp))

plot!(real.(time_exp))
plot!(real.(true_exp))
=#
println("See code for plotting tests")

println()

println("#############")
println("QN version")
println("#############")

@time qmpo = makeqMPO(Qlabels,mpo)

qenergy = dmrg(qpsi,qmpo,m=45,sweeps=20,cutoff=1E-9)

tqpsi = MPS(ComplexF64,qpsi)
tqmpo = MPO(ComplexF64,qmpo)

qtime_exp = Array{ComplexF64,1}(undef,nsteps+1)
qenergies = similar(qtime_exp)

qtime_exp[1] = 1.
qenergies[1] = qenergy

for i = 1:nsteps
  tdvp_twosite(tqpsi,tqmpo,prefactor=DeltaT,m=100)
  qtime_exp[i+1] = expect(tqpsi,qpsi)
  qenergies[i+1] = expect(tqpsi,tqmpo)
  println(i," ",qtime_exp[i+1]," ",exp(-i*DeltaT*qenergy)," ",qenergies[i+1])
end

qtrue_exp = [exp(-(i-1)*DeltaT*qenergy) for i = 1:nsteps+1]

#=
using Plots
plot(imag.(qtime_exp))
plot!(imag.(qtrue_exp))

plot!(real.(qtime_exp))
plot!(real.(qtrue_exp))
=#
println("See code for plotting tests")
