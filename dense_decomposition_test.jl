path = "../"
include(join([path,"DMRjulia.jl"]))

function testfct(test::Bool,message::String)
#  try test
  if test
    println("PASS "*message)
  else
    println("FAIL "*message)
  end
#  catch
#    error(message)
#  end
  return test
end

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

println("All tests passed? ",testval)