
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
for i = 1:length(newpsi)
  testvec[1] &= abs(ccontract(newpsi[i]) - size(newpsi[i],3)) < 1E-5
end
testval = testvec[1]
testval &= newpsi.oc == length(testpsi) + 1
fulltest &= testfct(testval,"leftnormalize(MPS)")

println()

newpsi,D,V = leftnormalize!(copy(testpsi))
testvec = [true]
for i = 1:length(newpsi)
  testvec[1] &= abs(ccontract(newpsi[i]) - size(newpsi[i],3)) < 1E-5
end
testval = testvec[1]
testval &= newpsi.oc == length(testpsi) + 1
fulltest &= testfct(testval,"leftnormalize!(MPS)")

println()

U,D,newpsi = rightnormalize(copy(testpsi))
testvec = [true]
for i = 1:length(newpsi)
  testvec[1] &= abs(ccontract(newpsi[i]) - size(newpsi[i],1)) < 1E-5
end
testval = testvec[1]
testval &= newpsi.oc == 0
fulltest &= testfct(testval,"rightnormalize(MPS)")


println()

U,D,newpsi = rightnormalize!(copy(testpsi))
testvec = [true]
for i = 1:length(newpsi)
  testvec[1] &= abs(ccontract(newpsi[i]) - size(newpsi[i],1)) < 1E-5
end
testval = testvec[1]
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

dmrg(psi,mpo,sweeps=20,m=45,cutoff=1E-9,silent=true)



testval = isapprox(expect(psi,mpo),-4.2580352071064205)
fulltest &= testfct(testval,"expect(psi,mpo)")

testval = isapprox(expect(psi,psi,mpo),-4.2580352071064205)
fulltest &= testfct(testval,"expect(psi,psi,mpo)")

println()

Hpsi = applyMPO(psi,mpo)

testval = abs(expect(psi,mpo) - expect(psi,Hpsi)) < 1E-5
fulltest &= testfct(testval,"applyMPO(psi,mpo)")

HHpsi = applyMPO(psi,mpo,mpo)

testval = abs(expect(psi,mpo,mpo) - expect(psi,HHpsi)) < 1E-2
fulltest &= testfct(testval,"applyMPO(psi,mpo,mpo)")

println()

M = correlationmatrix(psi,Sp,Sm)
checkM = [0.5000000057609738 -0.438729673232345 0.13174799658153824 -0.1482663248336932 0.0721993231264475 -0.08743334143048859 0.04675318965634688 -0.061058632442812184 0.029791646809763095 -0.04500418422539056; -0.438729673232345 0.4999999985905336 -0.19227042473346861 0.11155390355586084 -0.05834040835687454 0.06034207860141116 -0.03310468886188692 0.040859402736994635 -0.02010215207385281 0.029791962375457993; 0.13174799658153824 -0.19227042473346861 0.5000000016246995 -0.3856246682632819 0.12336934891734136 -0.13867662091080504 0.06781633692375989 -0.08616279683715307 0.04085991865040792 -0.061059090313612736; -0.1482663248336932 0.11155390355586084 -0.3856246682632819 0.5000000000244338 -0.21453515851649485 0.11944513721711392 -0.06403777454172464 0.06781642760062827 -0.033104919397969695 0.046753377197649874; 0.0721993231264475 -0.05834040835687454 0.12336934891734136 -0.21453515851649485 0.5000000015536458 -0.37637036310937433 0.11944517069652068 -0.13867662440901915 0.060342170201218 -0.08743345853628713; -0.08743334143048859 0.06034207860141116 -0.13867662091080504 0.11944513721711392 -0.37637036310937433 0.5000000003300575 -0.21453514661829318 0.12336942706836512 -0.058340560624859185 0.07219938981596252; 0.04675318965634688 -0.03310468886188692 0.06781633692375989 -0.06403777454172464 0.11944517069652068 -0.21453514661829318 0.5000000015847765 -0.38562487003142315 0.11155389892199896 -0.14826611613195526; -0.061058632442812184 0.040859402736994635 -0.08616279683715307 0.06781642760062827 -0.13867662440901915 0.12336942706836512 -0.38562487003142315 0.49999999758786623 -0.1922702279928199 0.1317478943165531; 0.029791646809763095 -0.02010215207385281 0.04085991865040792 -0.033104919397969695 0.060342170201218 -0.058340560624859185 0.11155389892199896 -0.1922702279928199 0.49999999846634874 -0.43872977449262035; -0.04500418422539056 0.029791962375457993 -0.061059090313612736 0.046753377197649874 -0.08743345853628713 0.07219938981596252 -0.14826611613195526 0.1317478943165531 -0.43872977449262035 0.4999999944766882]

testval = norm(checkM-M) < 1E-5
fulltest &= testfct(testval,"correlationmatrix(psi,Sp,Sm)")

M = correlationmatrix(psi,psi,Sp,Sm)
testval = norm(checkM-M) < 1E-5
fulltest &= testfct(testval,"correlationmatrix(psi,psi,Sp,Sm)")

println()

move!(psi,5)

L,T,R = localizeOp(psi,[Sp,Sm],[3,6])
import LinearAlgebra
testval = isapprox(makeArray(T),zeros(2,2) + LinearAlgebra.I)

A = contract(L,2,psi[psi.oc],1)
B = contract(A,3,R,1)
C = ccontract(psi[psi.oc],B)

testval &= isapprox(C,M[3,6])
fulltest &= testfct(testval,"localizeOp(psi,Operators,sites)")



L,T,R = localizeOp(psi,[Sp,Sm],[3,6],trail=(Id,Id))
import LinearAlgebra
testval = isapprox(makeArray(T),zeros(2,2) + LinearAlgebra.I)

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

En = expect(psi,mpo)

lambda = 0.1

penalty!(mpo,lambda,psi,compress=false)
shiftedEn = dmrg(psi,mpo,sweeps=20,m=45,cutoff=1E-9,silent=true)
testval = isapprox(abs(En-shiftedEn),lambda)
fulltest &= testfct(testval,"penalty!(mpo,Real,mps")

println()

testval = ndims(transfermatrix(psi,2,5)) == 4
fulltest &= testfct(testval,"transfermatrix(psi,integer,integer)")

testval = ndims(transfermatrix(psi,psi,2,5)) == 4
fulltest &= testfct(testval,"transfermatrix(psi,psi,integer,integer)")


