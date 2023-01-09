

println("#            +---------------------+")
println("#>-----------|  decompositions.jl  |-----------<")
println("#            +---------------------+")
fulltest = true

D = sort(rand(100),rev=true)
D /= norm(D)
mag = 0. #norm(D)
m = 20
minm = 2

cutoff=1E-1
effZero=1E-12
nozeros = true
power = 2
keepdeg = true

thism,sizeD,truncerr,sumD = DMRjulia.findnewm(D,m,minm,mag,cutoff,effZero,nozeros,power,keepdeg)

testval = thism == max(min(thism,length(D)),minm)
testval &= sizeD == length(D)
testval &= isapprox(sumD,sum(D .^ power))
testval &= isapprox(truncerr,sum(i->abs(D[i])^power,thism+1:length(D)))

fulltest &= testfct(testval,"findnewm")

println()

import LinearAlgebra

a = 200
b = 100
A = rand(a,b)

U,D,V = LinearAlgebra.svd(A)
Vt = Array(V)

Utrunc,Dtrunc,Vtrunc,sumD = DMRjulia.svdtrunc(a,b,U,D,Vt,m,minm,mag,cutoff,effZero,nozeros,power,keepdeg)

B = Utrunc*Dtrunc*Vtrunc
testval = size(B) == size(A) && size(Dtrunc) == (m,m)
fulltest &= testfct(testval,"svdtrunc")

println()

#missing recursive_SVD
#missing safesvd

println()

B = A/norm(A)
U,D,V,truncerr,sumD = svd(B)

testval = isapprox(U*D*V,B)
testval &= truncerr == 0. && isapprox(sumD,1)
fulltest &= testfct(testval,"svd(Array)")

tB = tens(B)
U,D,V,truncerr,sumD = svd(tB)

testval = isapprox((U*D*V).T,tB.T)
testval = size(U,1)*size(V,2) == prod(size(tB))
testval &= truncerr == 0. && isapprox(sumD,1)
fulltest &= testfct(testval,"svd(tens)")

println()


B = reshape(A,10,20,10,10)

rAA,Lsizes,Rsizes,a,b = DMRjulia.getorder(B,[[1,2],[3,4]])

testval = size(rAA) == (200,100) == (a,b)
testval &= Lsizes == [10,20]
testval &= Rsizes == [10,10]
fulltest &= testfct(testval,"getorder(Array)")

println()

a = DMRjulia.findsize(B,[1,2])
testval = a == 200

fulltest &= testfct(testval,"findsize")

println()

U,D,V = svd(B,[[1,2],[3,4]])

testval = size(U,1) == 10 && size(U,2) == 20 && ndims(U) == 3
testval &= size(V,2) == 10 && size(V,3) == 10 && ndims(V) == 3

fulltest &= testfct(testval,"svd(tens,Array{Array})")

U,D,V = svd(B,[[2,1],[3,4]])

testval = size(U) == (20,10,100)

fulltest &= testfct(testval,"svd(tens,Array{Array}) [permute inputs]")

println()

U,D,V = svd(A)
newD = svdvals(A)

testval = isapprox(LinearAlgebra.Diagonal(newD),D)
fulltest &= testfct(testval,"svdvals(Array)")

newD = svdvals(copy(A))

testval = isapprox(LinearAlgebra.Diagonal(newD),D)
fulltest &= testfct(testval,"svdvals!(Array)")

println()

A = rand(200,200)
A += A'
B = A/norm(A)
D,U,truncerr,sumD = eigen(B)

testval = isapprox(U*D*U',B)
#testval &= truncerr == 0. && isapprox(sumD,1)
fulltest &= testfct(testval,"eigen(Array)")

tB = tens(B)
D,U,truncerr,sumD = eigen(tB)

testval = isapprox((U*D*U').T,tB.T)
#testval = size(U,1)*size(V,2) == prod(size(tB))
#testval &= truncerr == 0. && isapprox(sumD,1)
fulltest &= testfct(testval,"eigen(tens)")

D,U,truncerr,sumD = eigen!(copy(tB))

testval = isapprox((U*D*U').T,tB.T)
#testval = size(U,1)*size(V,2) == prod(size(tB))
#testval &= truncerr == 0. && isapprox(sumD,1)
fulltest &= testfct(testval,"eigen!(tens)")

println()

tB = reshape(tB,10,20,20,10)
checkD,U = eigen(tB,[[1,2],[3,4]])

testval = isapprox(checkD,D)
fulltest &= testfct(testval,"eigen(tens)")

tB = reshape(tB,10,20,20,10)
checkD,U = eigen!(copy(tB),[[1,2],[3,4]])

testval = isapprox(checkD,D)
fulltest &= testfct(testval,"eigen!(tens)")

println()

A = rand(200,200)
A += A'
tB = tens(A)

D,U = LinearAlgebra.eigen(A)

checkD = eigvals(tB)
testval = isapprox(checkD,D)
fulltest &= testfct(testval,"eigvals(tens)")

checkD = eigvals(copy(tB))
testval = isapprox(checkD,D)
fulltest &= testfct(testval,"eigvals!(tens)")

println()

A = rand(200,100)
Q,R = qr(A)

testval = isapprox(Q*R,A)
testval *= isapprox(sum(Q'*Q),100)
outsum = [0.]
for y = 1:size(R,1)-1
  @inbounds @simd for x = y+1:size(R,1)
    outsum[1] += R[x,y]
  end
end
testval &= isapprox(outsum[1],0)
fulltest &= testfct(testval,"qr(Array)")

tA = tens(A)
Q,R = qr(tA)

testval = isapprox((Q*R).T,tA.T)

testval &= isapprox(sum(Q'*Q),100)
fulltest &= testfct(testval,"qr(tens)")

println()

A = rand(200,100)
Q,R = qr!(copy(A))

testval = isapprox(Q*R,A)
testval *= isapprox(sum(Q'*Q),100)
outsum = [0.]
for y = 1:size(R,1)-1
  @inbounds @simd for x = y+1:size(R,1)
    outsum[1] += R[x,y]
  end
end
testval &= isapprox(outsum[1],0)
fulltest &= testfct(testval,"qr!(Array)")

tA = tens(A)
Q,R = qr!(copy(tA))

testval = isapprox((Q*R).T,tA.T)

testval &= isapprox(sum(Q'*Q),100)
fulltest &= testfct(testval,"qr!(tens)")

println()

A = tens(rand(10,20,10,10))
Q,R = qr(A,[[1,2],[3,4]])
QR = contract(Q,ndims(Q),R,1)
testval = isapprox(QR.T,A.T)
testval &= size(QR) == (10,20,10,10)
fulltest &= testfct(testval,"qr(tens,Array{Array})")

println()

A = tens(rand(10,20,10,10))
Q,R = qr!(copy(A),[[1,2],[3,4]])
QR = contract(Q,ndims(Q),R,1)
testval = isapprox(QR.T,A.T)
testval &= size(QR) == (10,20,10,10)
fulltest &= testfct(testval,"qr!(tens,Array{Array})")










println()

A = rand(200,100)
L,Q = lq(A)

testval = isapprox(L*Q,A)
testval *= isapprox(sum(Q'*Q),100)
outsum = [0.]
for y = 2:size(L,2)
  @inbounds @simd for x = 1:y-1
    outsum[1] += L[x,y]
  end
end
testval &= isapprox(outsum[1],0)
fulltest &= testfct(testval,"lq(Array)")

tA = tens(A)
L,Q = lq(tA)

testval = isapprox((L*Q).T,tA.T)

testval &= isapprox(sum(Q'*Q),100)
fulltest &= testfct(testval,"lq(tens)")

println()

A = rand(200,100)
L,Q = lq!(copy(A))

testval = isapprox(L*Q,A)
testval &= isapprox(sum(Q'*Q),100)
outsum = [0.]
for y = 2:size(L,2)
  @inbounds @simd for x = 1:y-1
    outsum[1] += L[x,y]
  end
end
testval &= isapprox(outsum[1],0)
fulltest &= testfct(testval,"lq!(Array)")

tA = tens(A)
L,Q = lq!(copy(tA))

testval = isapprox((L*Q).T,tA.T)

testval &= isapprox(sum(Q'*Q),100)
fulltest &= testfct(testval,"lq!(tens)")

println()

A = tens(rand(10,20,10,10))
L,Q = lq(A,[[1,2],[3,4]])
LQ = contract(L,ndims(L),Q,1)
testval = isapprox(LQ.T,A.T)
testval &= size(LQ) == (10,20,10,10)
fulltest &= testfct(testval,"lq(tens,Array{Array})")

println()

A = tens(rand(10,20,10,10))
L,Q = lq!(copy(A),[[1,2],[3,4]])
LQ = contract(L,ndims(L),Q,1)
testval = isapprox(LQ.T,A.T)
testval &= size(LQ) == (10,20,10,10)
fulltest &= testfct(testval,"lq!(tens,Array{Array})")

println()

A = tens(rand(10,20,10,10))

U,D,V = svd(A,[[1,2],[3,4]])

UDU,UV = polar(A,[[1,2],[3,4]],right=false)
testval = isapprox(contract(UDU,ndims(UDU),UV,1).T,A.T)

UV,VDV = polar(A,[[1,2],[3,4]])
testval &= isapprox(contract(UV,ndims(UV),VDV,1).T,A.T)

fulltest &= testfct(testval,"polar(tens) [left and right]")

println()

m = 200

A = rand(m,m)
A = A + A'

D,U = eigen(A)

testval = isapprox(U*D*U',A)
fulltest &= testfct(testval,"regular eigen [base check]")

D,Ut = eigen(A,transpose=true)
testval = isapprox(norm(U-Ut'),0) && isapprox(Ut'*D*Ut,A)
fulltest &= testfct(testval,"regular eigen [transpose]")

@makeQNs "testxyz" U1

Qlabels = [[testxyz(-2),testxyz(0),testxyz(),testxyz(2)] for i = 1:8]

B = rand(Qlabels,[false,false,false,false,true,true,true,true])
for q = 1:length(B.T)
  B.T[q] += B.T[q]'
end

println()

C = makeArray(B)
rC = reshape(C,[[1,2,3,4],[5,6,7,8]])
rD,rU = eigen(rC)
D,U = eigen(C,[[1,2,3,4],[5,6,7,8]])

newC = contractc(contract(U,ndims(U),D,1),ndims(U),U,ndims(U))

testval = norm(newC-C) < 1E-12 && norm(rC-reshape(C,[[1,2,3,4],[5,6,7,8]])) < 1E-12
fulltest &= testfct(testval,"quantum eigen")


qD,qU = eigen(B,[[1,2,3,4],[5,6,7,8]])

P = makeArray(qD)
testval = isapprox(sort([P[i,i] for i = 1:size(P,1)]),sort([D[i,i] for i = 1:size(D,1)]))
fulltest &= testfct(testval,"quantum eigen [comparison with dense version]")

println()

newqC = contractc(contract(qU,ndims(qU),qD,1),ndims(qU),qU,ndims(qU))


testvalvec = [true]
for q = 1:length(newqC.T)
  testvalvec[1] &= isapprox(qU.T[q]*qD.T[q]*qU.T[q]',B.T[q])
end

testvalvec[1] &= norm(makeArray(newqC)-newC) < 1E-12
fulltest &= testfct(testvalvec[1],"reconstruct U*D*U' [quantum number version]")


qDt,qUt = eigen(B,[[1,2,3,4],[5,6,7,8]],transpose=true)

testvalvec = [true]
for q = 1:length(newqC.T)
  testvalvec[1] &= isapprox(qUt.T[q]'*qD.T[q]*qUt.T[q],B.T[q])
end

newqCt = contract(contractc(qDt,1,qUt,1),1,qUt,1)
testvalvec[1] &= norm(makeArray(newqCt)-newC) < 1E-12
fulltest &= testfct(testvalvec[1],"reconstruct U'*D*U [quantum number version, transposed]")
