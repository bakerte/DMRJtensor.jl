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

Ns = 100
spinmag = 0.5

hereQS = convert(Int64,2*spinmag+1)

initTensor = [zeros(1,hereQS,1) for i=1:Ns]
for i = 1:Ns
   initTensor[i][1,i%2 == 1 ? 1 : 2,1] = 1.0
end

psi = MPS(initTensor)

Sx,Sy,Sz,Sp,Sm,O,Id = spinOps(s=spinmag)
function H(i::Int64)
    return [Id O O O O;
        Sp O O O O;
        Sm O O O O;
        Sz O O O O;
        O Sm/2 Sp/2 Sz Id]
    end

println("Making qMPO")
@time mpo = makeMPO(H,hereQS,Ns)


#Quantum number specification
@makeQNs "spin" U1
Qlabels = [[spin(1),spin(-1)]]

qpsi = makeqMPS(psi,Qlabels)
qmpo = makeqMPO(mpo,Qlabels)

println("#############")
println("QN version")
println("#############")

@time energy = dmrg(qpsi,qmpo,maxm=45,sweeps=20,cutoff=1E-9,method="twosite")

println("#############")
println("nonQN version")
println("#############")

@time energy = dmrg(psi,mpo,maxm=45,sweeps=20,cutoff=1E-9,method="twosite")

