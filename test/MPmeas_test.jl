
println("#            +-------------+")
println("#>-----------|  MPmeas.jl  |-----------<")
println("#            +-------------+")
fulltest = true

Ns = 10
C = [heisenbergMPO(i) for i = 1:Ns]
mpo = makeMPO(C,[2])

B = fullH(mpo)
D,U = eigen(B)

testpsi = makeMPS(U[:,1],2)
A = fullpsi(testpsi)

En = (A'*B*A)[1]



X,Y,Z,truncerr = moveL(testpsi[testpsi.oc-1],testpsi[testpsi.oc])

moveL!(testpsi)
testval = isapprox(norm(X),norm(testpsi[testpsi.oc]))
testval &= isapprox(norm(Y),norm(testpsi[testpsi.oc+1]))
fulltest &= testfct(testval,"moveL(vect,vector) && moveL(MPS) [implicit: moveL!]")

println()

X,Y,Z,truncerr = moveR(testpsi[testpsi.oc],testpsi[testpsi.oc+1])

moveR!(testpsi)
testval = isapprox(norm(Y),norm(testpsi[testpsi.oc]))
testval &= isapprox(norm(X),norm(testpsi[testpsi.oc-1]))
fulltest &= testfct(testval,"moveR(vect,vector) && moveR(MPS) [implicit: moveR!]")

println()



println()

move!(testpsi,1)
move!(testpsi,5)

testval = isapprox(expect(testpsi),1)
testval &= testpsi.oc == 5
fulltest &= testfct(testval,"move!(MPS) [implicit: movecenter!]")

println()

A = move(testpsi,2)

testval = isapprox(expect(A),1)
testval &= A.oc == 2
fulltest &= testfct(testval,"move(MPS)")

println()

newpsi,D,V = leftnormalize(copy(testpsi))
testvec = [true]
for i = 1:length(newpsi)-1
  testvec[1] &= abs(ccontract(newpsi[i]) - size(newpsi[i],3)) < 1E-5
end
testval = testvec[1] && abs(ccontract(newpsi[end]) - 1) < 1E-5
testval &= newpsi.oc == length(testpsi) + 1
fulltest &= testfct(testval,"leftnormalize(MPS)")

println()

newpsi,D,V = leftnormalize!(copy(testpsi))
testvec = [true]
for i = 1:length(newpsi)-1
  testvec[1] &= abs(ccontract(newpsi[i]) - size(newpsi[i],3)) < 1E-5
end
testval = testvec[1] && abs(ccontract(newpsi[end]) - 1) < 1E-5
testval &= newpsi.oc == length(testpsi) + 1
fulltest &= testfct(testval,"leftnormalize!(MPS)")

println()

U,D,newpsi = rightnormalize(copy(testpsi))
testvec = [true]
for i = 2:length(newpsi)
  testvec[1] &= abs(ccontract(newpsi[i]) - size(newpsi[i],1)) < 1E-5
end
testval = testvec[1] && abs(ccontract(newpsi[1]) - 1) < 1E-5
testval &= newpsi.oc == 0
fulltest &= testfct(testval,"rightnormalize(MPS)")


println()

U,D,newpsi = rightnormalize!(copy(testpsi))
testvec = [true]
for i = 2:length(newpsi)
  testvec[1] &= abs(ccontract(newpsi[i]) - size(newpsi[i],1)) < 1E-5
end
testval = testvec[1] && abs(ccontract(newpsi[1]) - 1) < 1E-5
testval &= newpsi.oc == 0
fulltest &= testfct(testval,"rightnormalize!(MPS)")

println()

psi = randMPS(2,Ns)

B = makeBoundary(psi,psi,mpo,mpo)
testval = B.size == [1, 1, 1, 1]
testval &= B.T == [1.]
fulltest &= testfct(testval,"makeBoundary(dualpsi,psi,mpo...)")

println()

A = DMRjulia.defaultBoundary(B)
testval = typeof(A) <: TensType
testval &= norm(A) == 0
fulltest &= testfct(testval,"defaultBoundary(TensType)")


@makeQNs "test3" U1
Qlabels = [[test3(-1),test3(1)]]
C = Qtens(Qlabels)

A = DMRjulia.defaultBoundary(C)
testval = typeof(A) <: qarray
testval &= norm(A) == 0
fulltest &= testfct(testval,"defaultBoundary(qarray)")

println()

B = DMRjulia.makeEdgeEnv(psi,psi,mpo,mpo)
testval = B.size == [1, 1, 1, 1]
testval &= B.T == [1.]
fulltest &= testfct(testval,"makeEdgeEnv(dualpsi,psi,mpo...)")

println()

B,C = makeEnds(psi,psi,mpo,mpo)
testval = B.size == [1, 1, 1, 1]
testval &= B.T == [1.]
testval &= C.size == [1, 1, 1, 1]
testval &= C.T == [1.]
fulltest &= testfct(testval,"makeEnds(dualpsi,psi,mpo...)")

B,C = makeEnds(psi,psi,mpo,mpo)
testval = B.size == [1, 1, 1, 1]
testval &= B.T == [1.]
testval &= C.size == [1, 1, 1, 1]
testval &= C.T == [1.]
fulltest &= testfct(testval,"makeEnds(psi,mpo...)")

println()

move!(psi,7)

Lenv,Renv = makeEnv(psi,psi,mpo)

B = contract(Lenv[psi.oc],3,psi[psi.oc],1)
B = contract(B,[2,3],mpo[psi.oc],[1,2])
B = contract(B,[2,4],Renv[psi.oc],[1,2])
C = ccontract(psi[psi.oc],B)
testval = isapprox(expect(psi,mpo),C)
fulltest &= testfct(testval,"makeEnv(psi,psi,mpo...)")


Lenv,Renv = makeEnv(psi,mpo)

B = contract(Lenv[psi.oc],3,psi[psi.oc],1)
B = contract(B,[2,3],mpo[psi.oc],[1,2])
B = contract(B,[2,4],Renv[psi.oc],[1,2])
C = ccontract(psi[psi.oc],B)
testval = isapprox(expect(psi,mpo),C)
fulltest &= testfct(testval,"makeEnv(psi,mpo...)")


println()

B = Lupdate(Lenv[4],psi[4],psi[4],mpo[4])
testval = isapprox(B.T,Lenv[5].T)
fulltest &= testfct(testval,"Lupdate(TensType,psi,psi,mpo...)")

println()

C = Rupdate(Renv[9],psi[9],psi[9],mpo[9])
testval = isapprox(C.T,Renv[8].T)
fulltest &= testfct(testval,"Rupdate(TensType,psi,psi,mpo...)")

println()

Lupdate!(4,Lenv,psi,psi,mpo)
testval = isapprox(B.T,Lenv[5].T)
fulltest &= testfct(testval,"Lupdate!(TensType,psi,psi,mpo...)")

Lupdate!(4,Lenv,psi,mpo)
testval = isapprox(B.T,Lenv[5].T)
fulltest &= testfct(testval,"Lupdate!(TensType,psi,mpo...)")

println()

Rupdate!(9,Renv,psi,psi,mpo)
testval = isapprox(C.T,Renv[8].T)
fulltest &= testfct(testval,"Rupdate!(TensType,psi,psi,mpo...)")

Rupdate!(9,Renv,psi,mpo)
testval = isapprox(C.T,Renv[8].T)
fulltest &= testfct(testval,"Rupdate!(TensType,psi,mpo...)")

println()

Lenv,Renv = makeEnv(psi,mpo)

boundaryMove!(psi,4,Lenv,Renv,mpo)

B = contract(Lenv[psi.oc],3,psi[psi.oc],1)
B = contract(B,[2,3],mpo[psi.oc],[1,2])
B = contract(B,[2,4],Renv[psi.oc],[1,2])
C = ccontract(psi[psi.oc],B)
testval = isapprox(expect(psi,mpo),C)
fulltest &= testfct(testval,"boundaryMove!(psi,integer,Env,Env,mpo...)")

boundaryMove!(psi,6,Lenv,Renv,mpo)

B = contract(Lenv[psi.oc],3,psi[psi.oc],1)
B = contract(B,[2,3],mpo[psi.oc],[1,2])
B = contract(B,[2,4],Renv[psi.oc],[1,2])
C = ccontract(psi[psi.oc],B)
testval = isapprox(expect(psi,mpo),C)
fulltest &= testfct(testval,"boundaryMove!(dualpsi,psi,integer,Env,Env,mpo...)")

println()

psi,Lenv,Renv = boundaryMove(psi,4,Lenv,Renv,mpo)

B = contract(Lenv[psi.oc],3,psi[psi.oc],1)
B = contract(B,[2,3],mpo[psi.oc],[1,2])
B = contract(B,[2,4],Renv[psi.oc],[1,2])
C = ccontract(psi[psi.oc],B)
testval = isapprox(expect(psi,mpo),C)
fulltest &= testfct(testval,"boundaryMove(psi,integer,Env,Env,mpo...)")

psi,Lenv,Renv = boundaryMove(psi,6,Lenv,Renv,mpo)

B = contract(Lenv[psi.oc],3,psi[psi.oc],1)
B = contract(B,[2,3],mpo[psi.oc],[1,2])
B = contract(B,[2,4],Renv[psi.oc],[1,2])
C = ccontract(psi[psi.oc],B)
testval = isapprox(expect(psi,mpo),C)
fulltest &= testfct(testval,"boundaryMove(dualpsi,psi,integer,Env,Env,mpo...)")

println()

spinmag = 0.5

hereQS = convert(Int64,2*spinmag+1)
QS = cld(hereQS,2)

initTensor = [zeros(1,hereQS,1) for i=1:Ns]
for i = 1:Ns
   initTensor[i][1,i%2 == 1 ? 1 : 2,1] = 1.0
end

psi = MPS(initTensor)

Sp,Sm,Sz,Sy,Sx,O,Id = spinOps()
upsites = [2,4]
applyOps!(psi,upsites,Sp)

testval = isapprox(expect(psi,mpo),-0.25)
fulltest &= testfct(testval,"applyOps!(psi,sites,Op)")

println()

psi = MPS(initTensor)

Sp,Sm,Sz,Sy,Sx,O,Id = spinOps()
upsites = [2,4]
newpsi = applyOps(psi,upsites,Sp)

testval = isapprox(expect(newpsi,mpo),-0.25)
testval &= !(newpsi === psi)
fulltest &= testfct(testval,"applyOps(psi,sites,Op)")

println()

D,U = eigen(mpo)
psi = makeMPS(U[:,1],2)

#dmrg(psi,mpo,sweeps=20,m=45,cutoff=1E-9,silent=true)

testval = isapprox(expect(psi,mpo),-4.2580352071064205)
fulltest &= testfct(testval,"expect(psi,mpo)")

testval = isapprox(expect(psi,psi,mpo),-4.2580352071064205)
fulltest &= testfct(testval,"expect(psi,psi,mpo)")

println()

xHpsi = applyMPO(psi,mpo)
testval = abs(expect(psi,mpo) - expect(psi,xHpsi)) < 1E-2
fulltest &= testfct(testval,"applyMPO(psi,mpo)")

HHpsi = applyMPO(psi,mpo,mpo)

testval = abs(expect(psi,mpo,mpo) - expect(psi,HHpsi)) < 1E-2
fulltest &= testfct(testval,"applyMPO(psi,mpo,mpo)")

println()

M = correlationmatrix(psi,Sp,Sm)
checkM = [0.5000000000076139 -0.43872967633621973 0.13174799941792897 -0.14826632734925407 0.07219933185092689 -0.08743338281026562 0.04675324967326652 -0.06105775323552169 0.029789565383917464 -0.045003006594778676; -0.43872967633621973 0.4999999999947903 -0.19227042752001863 0.11155390189083765 -0.058340400052541465 0.06034206989865802 -0.0331044765849888 0.04085646471438276 -0.02009702139402716 0.029789565383917457; 0.13174799941792897 -0.19227042752001863 0.5000000000085532 -0.38562466592197114 0.12336933756144136 -0.13867658178596565 0.06781601638585225 -0.08616038961612817 0.0408564647143827 -0.0610577532355216; -0.14826632734925407 0.11155390189083765 -0.38562466592197114 0.499999999992767 -0.21453514266806917 0.11944507953765461 -0.06403763496332761 0.06781601638585201 -0.03310447658498866 0.04675324967326632; 0.07219933185092689 -0.058340400052541465 0.12336933756144136 -0.21453514266806917 0.5000000000084245 -0.37637031153183914 0.11944507953765447 -0.13867658178596512 0.0603420698986578 -0.0874333828102654; -0.08743338281026562 0.06034206989865802 -0.13867658178596565 0.11944507953765461 -0.37637031153183914 0.4999999999916104 -0.214535142668069 0.12336933756144121 -0.05834040005254136 0.07219933185092689; 0.04675324967326652 -0.0331044765849888 0.06781601638585225 -0.06403763496332761 0.11944507953765447 -0.214535142668069 0.500000000007405 -0.3856246659219711 0.11155390189083797 -0.14826632734925468; -0.06105775323552169 0.04085646471438276 -0.08616038961612817 0.06781601638585201 -0.13867658178596512 0.12336933756144121 -0.3856246659219711 0.499999999991303 -0.1922704275200191 0.13174799941792925; 0.029789565383917464 -0.02009702139402716 0.0408564647143827 -0.03310447658498866 0.0603420698986578 -0.05834040005254136 0.11155390189083797 -0.1922704275200191 0.5000000000053961 -0.43872967633621973; -0.045003006594778676 0.029789565383917457 -0.0610577532355216 0.04675324967326632 -0.0874333828102654 0.07219933185092689 -0.14826632734925468 0.13174799941792925 -0.43872967633621973 0.49999999999213623]

testval = norm(checkM-M)/length(checkM) < 1E-5
fulltest &= testfct(testval,"correlationmatrix(psi,Sp,Sm)")

M = correlationmatrix(psi,psi,Sp,Sm)
testval = norm(checkM-M)/length(checkM) < 1E-5
fulltest &= testfct(testval,"correlationmatrix(psi,psi,Sp,Sm)")

println()

move!(psi,5)

L,T,R = localizeOp(psi,[Sp,Sm],[3,6])
import LinearAlgebra
testval = isapprox(Array(T),zeros(2,2) + LinearAlgebra.I)

A = contract(L,2,psi[psi.oc],1)
B = contract(A,3,R,1)
C = ccontract(psi[psi.oc],B)

testval &= isapprox(C,M[3,6])
fulltest &= testfct(testval,"localizeOp(psi,Operators,sites)")



L,T,R = localizeOp(psi,[Sp,Sm],[3,6],trail=(Id,Id))
import LinearAlgebra
testval = isapprox(Array(T),zeros(2,2) + LinearAlgebra.I)

A = contract(L,2,psi[psi.oc],1)
B = contract(A,3,R,1)
C = ccontract(psi[psi.oc],B)

testval &= isapprox(C,M[3,6])
fulltest &= testfct(testval,"localizeOp(psi,Operators,sites,trail=...)")


A = localizeOp(psi,mpo)
testval = isapprox(ccontract(psi[psi.oc],A),expect(psi,mpo))
fulltest &= testfct(testval,"localizeOp(mps,mpo)")

println()

A = [3,3,4]
B = [4,4,4]
DMRjulia.operator_in_order!(A,B)

testval = A == [3,4,4]
fulltest &= testfct(testval,"operator_in_order!(vector,vector)")

println()

testval = length(permutations(4)) == 24
fulltest &= testfct(testval,"permutations(integer))")

println()

M = correlation(psi,Sp,Sm)
testval = sum([abs(checkM[w]-M[w]) < 1E-5 for w = 1:length(M)]) == length(M)
fulltest &= testfct(testval,"correlations(psi,Sp,Sm)")

M = correlation(psi,psi,Sp,Sm)
testval = sum([abs(checkM[w]-M[w]) < 1E-5 for w = 1:length(M)]) == length(M)
fulltest &= testfct(testval,"correlations(psi,psi,Sp,Sm)")

println()

Qlabel = [test3(2),test3(-2)]
qpsi,qmpo = MPS(Qlabel,psi,mpo)
En = dmrg(qpsi,qmpo,sweeps=20,m=45,cutoff=1E-9,silent=true)

xlambda = 0.01

penalty!(mpo,xlambda,psi,compress=false)
shiftedEn = dmrg(psi,mpo,sweeps=20,m=45,cutoff=1E-9,silent=true)
testval = abs(abs(En-shiftedEn) - xlambda) < 1E-6
fulltest &= testfct(testval,"penalty!(mpo,Real,mps)")

En = expect(qpsi,qmpo)

penalty!(qmpo,xlambda,qpsi,compress=false)
shiftedEn = dmrg(qpsi,qmpo,sweeps=20,m=45,cutoff=1E-9,silent=true)
testval = abs(abs(En-shiftedEn) - xlambda) < 1E-6
fulltest &= testfct(testval,"penalty!(mpo,Real,mps) [Quantum number types]")

println()

testval = ndims(transfermatrix(psi,2,5)) == 4
fulltest &= testfct(testval,"transfermatrix(psi,integer,integer)")

testval = ndims(transfermatrix(psi,psi,2,5)) == 4
fulltest &= testfct(testval,"transfermatrix(psi,psi,integer,integer)")


