println("#            +-------------+")
println("#>-----------|  MPtask.jl  |-----------<")
println("#            +-------------+")
fulltest = true

import LinearAlgebra

msize = 4
Ns = 10
C = [rand(i == 1 ? 1 : msize,2,i == Ns ? 1 : msize) for i = 1:Ns]

psi = MPS(C,oc=3,regtens=true)
testval = psi.oc == 3
testval &= psi.A == C
fulltest &= testfct(testval,"MPS(vector{Array},oc=)")

C = [tens(rand(i == 1 ? 1 : msize,2,i == Ns ? 1 : msize)) for i = 1:Ns]

psi = MPS(C,oc=3)
testval = psi.oc == 3
testval &= length(C) == sum([makeArray(psi.A[i]) == makeArray(C[i]) for i = 1:length(C)])
fulltest &= testfct(testval,"MPS(vector{Tens},oc=)")

X = MPS(psi)
testval = X.oc == 3
testval &= length(C) == sum([makeArray(X.A[i]) == makeArray(C[i]) for i = 1:length(C)])
fulltest &= testfct(testval,"MPS(MPS,oc=)")

D = MPS(ComplexF64,C,oc=3)
testval = D.oc == 3
testval &= eltype(D[1]) <: ComplexF64
fulltest &= testfct(testval,"MPS(ComplexF64,MPS,oc=)")

println()

C = [rand(i == 1 ? 1 : msize,2,2,i == Ns ? 1 : msize) for i = 1:Ns]

mpo = MPO(C,regtens=true)
testval = mpo.H == C
fulltest &= testfct(testval,"MPO(vector{Array})")

C = [tens(rand(i == 1 ? 1 : msize,2,2,i == Ns ? 1 : msize)) for i = 1:Ns]

mpo = MPO(ComplexF64,C)
testval = eltype(mpo) == ComplexF64
fulltest &= testfct(testval,"MPO(ComplexF64,vector{Tens})")

mpo = MPO(C)
testval = mpo.H == C
fulltest &= testfct(testval,"MPO(vector{Tens})")

B = MPO(mpo)
testval = mpo.H == B.H
fulltest &= testfct(testval,"MPO(MPO)")

Y = MPO(ComplexF64,mpo)
testval = eltype(Y) == ComplexF64
fulltest &= testfct(testval,"MPO(ComplexF64,MPO)")


println()


Z = environment(C...)
testval = Z.V == C
fulltest &= testfct(testval,"environment(vector)")

B = environment(C[1],Ns)
testval = length(B.V) == Ns
testval &= eltype(B.V) == eltype(C)
fulltest &= testfct(testval,"environment(Array,Ns)")

B = environment(tens(C[1]),Ns)
testval = length(B.V) == Ns
testval &= eltype(B.V) == eltype(C)
fulltest &= testfct(testval,"environment(tens,Ns)")

B = environment(psi)
testval = Ns == length(B.V)
fulltest &= testfct(testval,"environment(MPS)")

B = environment(mpo)
testval = Ns == length(B.V)
fulltest &= testfct(testval,"environment(MPO)")

println()

testval = elnumtype(X,Y,Z) <: ComplexF64
testval &= elnumtype(X,Z) <: Float64
fulltest &= testfct(testval,"elnumtype(MPS,MPO,Env)")

println()

testval = size(X) == (Ns,)
testval &= size(X,5) == size(X[5])
fulltest &= testfct(testval,"size(MPS[,int])")

testval = size(Y) == (Ns,)
testval &= size(Y,5) == size(Y[5])
fulltest &= testfct(testval,"size(MPO[,int])")

testval = size(Z) == (Ns,)
testval &= size(Z,5) == size(Z[5])
fulltest &= testfct(testval,"size(Env[,int])")

println()

testval = length(X) == Ns
fulltest &= testfct(testval,"length(MPS)")

testval = length(Y) == Ns
fulltest &= testfct(testval,"length(MPO)")

testval = length(Z) == Ns
fulltest &= testfct(testval,"length(Env)")

println()

testval = eltype(X) == Float64
fulltest &= testfct(testval,"eltype(MPS)")

testval = eltype(Y) == ComplexF64
fulltest &= testfct(testval,"eltype(MPO)")

testval = eltype(Z) == Float64
fulltest &= testfct(testval,"eltype(Env)")

println()

testval = X[5] == X.A[5]
fulltest &= testfct(testval,"getindex(MPS,integer)")

testval = Y[5] == Y.H[5]
fulltest &= testfct(testval,"getindex(MPO,integer)")

testval = Z[5] == Z.V[5]
fulltest &= testfct(testval,"getindex(Env,integer)")

B = X[4:6].A
C = X.A[4:6]
testval = makeArray(B[1]) == makeArray(C[1])
testval &= makeArray(B[2]) == makeArray(C[2])
testval &= makeArray(B[3]) == makeArray(C[3])
fulltest &= testfct(testval,"getindex(MPS,unitrange)")

testval = Y[4:6].H == Y.H[4:6]
fulltest &= testfct(testval,"getindex(MPO,unitrange)")

testval = Z[4:6].V == Z.V[4:6]
fulltest &= testfct(testval,"getindex(Env,unitrange)")

println()

testval = X[end] == X.A[end]
fulltest &= testfct(testval,"lastindex(MPS)")

testval = Y[end] == Y.H[end]
fulltest &= testfct(testval,"lastindex(MPO)")

testval = Z[end] == Z.V[end]
fulltest &= testfct(testval,"lastindex(Env)")

println()

B = tens(rand(20,2,40))
C = tens(rand(ComplexF64,20,2,40))

X[4] = B
testval = X[4] == B
fulltest &= testfct(testval,"setindex!(MPS)")

Y[4] = C
testval = Y[4] == C
fulltest &= testfct(testval,"setindex!(MPO)")

Z[4] = B
testval = Z[4] == B
fulltest &= testfct(testval,"setindex!(Env)")

println()

copypsi = copy(psi)
testval = sum(w->makeArray(copypsi.A[w]) == makeArray(psi.A[w]),1:Ns) == Ns
fulltest &= testfct(testval,"copy(MPS)")

copympo = copy(mpo)
testval = sum(w->makeArray(copympo.H[w]) == makeArray(mpo.H[w]),1:Ns) == Ns
fulltest &= testfct(testval,"copy(MPO)")

C = environment([psi[i] for i = 1:length(psi)])

copyenv = copy(C)
testval = sum(w->makeArray(copyenv.V[w]) == makeArray(psi.A[w]),1:Ns) == Ns
fulltest &= testfct(testval,"copy(Env)")

println()

conjD = conj!(copy(D))
testval = isapprox(norm(conj(conjD[1])-D[1]),0)
fulltest &= testfct(testval,"conj!(MPS)")

println()

conjD = conj(D)
testval = isapprox(norm(conj(conjD[1])-D[1]),0)
fulltest &= testfct(testval,"conj(MPS)")

println()

C = randMPS(2,Ns,m=msize)
testval = size(C[3],1) == msize
fulltest &= testfct(testval,"randMPS(integer,integer)")

B = [2 for i = 1:Ns]
C = randMPS(B,m=msize)
testval = size(C[3],1) == msize
fulltest &= testfct(testval,"randMPS(vector)")

C = randMPS([2,2,2],Ns,m=msize)
testval = size(C[3],1) == msize && length(C) == Ns
fulltest &= testfct(testval,"randMPS(vector,integer)")

C = randMPS(psi,m=msize)
testval = size(C[3],1) == msize && length(C) == Ns
fulltest &= testfct(testval,"randMPS(mps,integer)")

C = randMPS(mpo,m=msize)
testval = size(C[3],1) == msize && length(C) == Ns
fulltest &= testfct(testval,"randMPS(mpo,integer)")

println()

Ns = 10
spinmag = 0.5

hereQS = convert(Int64,2*spinmag+1)
QS = cld(hereQS,2)

initTensor = [zeros(1,hereQS,1) for i=1:Ns]
for i = 1:Ns
   initTensor[i][1,i%2 == 1 ? 1 : 2,1] = 1.0
end

psi = MPS(initTensor)

newpsi = psi/3
testval = isapprox(expect(psi)/expect(newpsi),3^2)
fulltest &= testfct(testval,"/(MPS,number)")

println()

newpsi = div!(copy(psi),3)
testval = isapprox(expect(psi)/expect(newpsi),3^2)
fulltest &= testfct(testval,"div!(MPS,number)")

println()

newpsi = psi*3
testval = isapprox(expect(newpsi)/expect(psi),3^2)
fulltest &= testfct(testval,"*(MPS,number)")

newpsi = psi*3
testval = isapprox(expect(newpsi)/expect(psi),3^2)
fulltest &= testfct(testval,"*(number,MPS)")

println()

newpsi = mult!(copy(psi),3)
testval = isapprox(expect(newpsi)/expect(psi),3^2)
fulltest &= testfct(testval,"mult!(MPS,number)")

newpsi = mult!(3,copy(psi))
testval = isapprox(expect(newpsi)/expect(psi),3^2)
fulltest &= testfct(testval,"mult!(number,MPS)")

println()


try
  global C = [heisenbergMPO(i) for i = 1:Ns]
  global mpo = makeMPO(C,[2])
  global mpo = makeMPO(C,2)
  global mpo = makeMPO(heisenbergMPO(1),2,Ns)
  global mpo = makeMPO(heisenbergMPO(1),[2],Ns)
  global mpo = makeMPO(heisenbergMPO,2,Ns)
  global mpo = makeMPO(heisenbergMPO,[2],Ns)
  global testval = true
catch
  global testval = false
end

fulltest &= testfct(true,"makeMPO([arguments...check energies below])")

println()

C = [heisenbergMPO(i) for i = 1:Ns]
mpo = makeMPO(C,[2])

B = fullH(mpo)
D,U = eigen(B)

testval = isapprox(D[1,1],-4.258035207282883)
fulltest &= testfct(testval,"fullH(MPO)")

println()

testpsi = makeMPS(makeArray(U[:,1]),2)
testval = isapprox(expect(testpsi,mpo),-4.258035207282883)
fulltest &= testfct(testval,"makeMPS(vect,integer)")

testpsi = makeMPS(U[:,1],[2,2])
testval = isapprox(expect(testpsi,mpo),-4.258035207282883)
fulltest &= testfct(testval,"makeMPS(vect,vector)")

println()

C = fullpsi(testpsi)

testval = isapprox(makeArray(U[:,1]),makeArray(C))
fulltest &= testfct(testval,"fullpsi(psi)")

println()

#Quantum number MPS

@makeQNs "testMPS" U1

Qlabels = [[testMPS(1),testMPS(-1)] for i = 1:Ns]


try
  global A = MPS(Qlabels[1],Ns)
  global testval = length(A) == Ns
catch
  global testval = false
end
fulltest &= testfct(testval,"MPS(Vector{Qnum},integer)")

println()

try
  global A = MPS(Qlabels,Ns)
  global testval = length(A) == Ns
catch
  global testval = false
end
fulltest &= testfct(testval,"MPS(Vector{Vector{Qnum}},integer)")

try
  global A = MPS(Qlabels)
  global testval = length(A) == Ns
catch
  global testval = false
end
fulltest &= testfct(testval,"MPS(Vector{Vector{Qnum}})")



println()


#randMPS(Qlabels)


println()

A = makeqMPS(psi,Qlabels[1],silent=true)
testval = isapprox(expect(A),expect(psi))
fulltest &= testfct(testval,"makeqMPS(MPS,Vector{Qnum})")

A = makeqMPS(psi,Qlabels,silent=true)
testval = isapprox(expect(A),expect(psi))
fulltest &= testfct(testval,"makeqMPS(MPS,Vector{Vector{Qnum}})")

A = makeqMPS(psi.A,Qlabels,silent=true)
testval = isapprox(expect(A),expect(psi))
fulltest &= testfct(testval,"makeqMPS(vector,Vector{Vector{Qnum}})")

println()

A = makeqMPO(mpo,Qlabels[1])
testvalvec = [true]
for i = 1:length(A)
  testvalvec[1] &= isapprox(makeArray(A[i]),makeArray(mpo[i]))
end
fulltest &= testfct(testvalvec[1],"makeqMPO(MPO,Vector{Qnum})")

A = makeqMPO(mpo,Qlabels)
testvalvec = [true]
for i = 1:length(A)
  testvalvec[1] &= isapprox(makeArray(A[i]),makeArray(mpo[i]))
end
fulltest &= testfct(testvalvec[1],"makeqMPO(MPO,Vector{Vector{Qnum}})")
#=
A = makeqMPO(mpo.H,Qlabels)
testvalvec = [true]
for i = 1:length(A)
  testvalvec[1] &= isapprox(makeArray(A[i]),makeArray(mpo[i]))
end
fulltest &= testfct(testvalvec[1],"makeqMPO(vector,Vector{Vector{Qnum}})")
=#