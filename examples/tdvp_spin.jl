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

Ns = 12

psi = randMPS(2,Ns)

mpo = makeMPO(XXZ,2,Ns)

energy = dmrg(psi,mpo,m=100,sweeps=300,goal=1E-10,cutoff=1E-9)

DeltaT = -im*0.001

tpsi = MPS(ComplexF64,psi)
tmpo = MPO(ComplexF64,mpo)

nsteps = 100
time_exp = Array{ComplexF64,1}(undef,nsteps+1)
energies = similar(time_exp)

time_exp[1] = 1.
energies[1] = energy #expect(psi,mpo)

for i = 1:nsteps
  tdvp_twosite(tpsi,tmpo,prefactor=DeltaT,maxm=100)
  time_exp[i+1] = expect(tpsi,psi)
  energies[i+1] = expect(tpsi,tmpo)
  println(i," ",time_exp[i+1]," ",exp(-i*DeltaT*energy)," ",energies[i+1])
end

true_exp = [exp(-(i-1)*DeltaT*energy) for i = 1:nsteps+1]
#=
using Plots
plot(real.(time_exp))
plot!(real.(true_exp))

plot!(imag.(time_exp))
plot!(imag.(true_exp))
=#
println("See file for plotting checks")
