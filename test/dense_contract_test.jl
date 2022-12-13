

println("#            +-------------------+")
println("#>-----------|  contractions.jl  |-----------<")
println("#            +-------------------+")
fulltest = true

A = rand(10,20,30,10,5)
tA = tens(A)
iA = [1,2,3,4]
iB = (iA...,)

testval = DMRjulia.permq(A,iA)[1] && DMRjulia.permq(A,iB)[1]
fulltest &= testfct(testval,"permq(Array,Vector || Tuple)")

testval = DMRjulia.permq(tA,iA)[1] && DMRjulia.permq(tA,iB)[1]
fulltest &= testfct(testval,"permq(tens,Vector || Tuple)")

A = rand(10,20,30,10,5)
iA = [2,3,4,5]
iB = (iA...,)

testval = DMRjulia.permq(A,iA)[2] && DMRjulia.permq(A,iB)[2]
fulltest &= testfct(testval,"permq(Array,Vector || Tuple)")

testval = DMRjulia.permq(A,iA)[2] && DMRjulia.permq(A,iB)[2]
fulltest &= testfct(testval,"permq(tens,Vector || Tuple)")

println()

A = DMRjulia.willperm(true,ComplexF64,true,false)
testval = true == A[1] && 'C' == A[2]
A = DMRjulia.willperm(true,ComplexF64,false,true)
testval &= true == A[1] && 'N' == A[2]
A = DMRjulia.willperm(true,Float64,true,false)
testval &= true == A[1] && 'T' == A[2]
A = DMRjulia.willperm(false,Float64,true,false)
testval &= true == A[1] && 'T' == A[2]

fulltest &= testfct(testval,"willperm")

println()

A = rand(ComplexF64,10,20,30,40,10,20)
iA = [1,2,3,4]
B = DMRjulia.prepareT(A,iA,[5,6],false)

testval = isapprox(norm(A),norm(B))

A = rand(ComplexF64,10,20,30,40,10,20)
B = DMRjulia.prepareT(A,[1,2,3,4],[5,6],true)

testval &= isapprox(norm(A),norm(B)')

fulltest &= testfct(testval,"prepareT")

println()

B = zeros(intType,6)

Lsize,innersizeL = DMRjulia.getsizes(A,iA,B,4)
testval = B[5] == size(A,5) && B[6] == size(A,6)
testval &= Lsize == size(A,1)*size(A,2)
testval &= innersizeL == prod(w->size(A,w),3:6)

fulltest &= testfct(testval,"getsizes")

println()

A = rand(20,40)
B = rand(40,20)
C = rand(20,20)

iA = [2]
iB = [1]

checkC = DMRjulia.maincontractor(false,false,A,iA,B,iB,alpha=2.)

testval = isapprox(checkC,2. * A*B)#+2.*C)
fulltest &= testfct(testval,"alpha*A*B [Array]")


checkC = DMRjulia.maincontractor(false,false,A,iA,B,iB,copy(C),alpha=2.,beta=3.)

testval = isapprox(checkC,2. * A*B + 3. * C)
fulltest &= testfct(testval,"alpha*A*B + beta*C [Array]")

checkC = DMRjulia.maincontractor(false,false,tens(A),iA,tens(B),iB,alpha=2.)

testval = isapprox(checkC.T,reshape(2. * A*B,400))#+2.*C)
fulltest &= testfct(testval,"alpha*A*B [tens]")


checkC = DMRjulia.maincontractor(false,false,tens(A),iA,tens(B),iB,tens(copy(C)),alpha=2.,beta=3.)

testval = isapprox(checkC.T,reshape(2. * A*B + 3. * C,400))
fulltest &= testfct(testval,"alpha*A*B + beta*C [tens]")

println()

A = rand(20,40)
B = rand(20,40)

testval = isapprox(dot(A,B),dot(tens(A),tens(B)))
fulltest &= testfct(testval,"dot(A,B) [Array || tens]")

A = rand(40)
C = rand(40,40)
B = rand(40)

tB = adjoint(B)
res = dot(A,C,B)
testval = isapprox(A'*C*B,res)
testval &= isapprox(res,dot(tens(A),tens(C),tens(B)))
fulltest &= testfct(testval,"dot(A,B,C) [Array || tens]")

println()

A = rand(20,40,30,10)
B = rand(10,30,20,40)

C = A*B
tC = tens(A)*tens(B)

testval = size(C) == (20,40,30,30,20,40)
testval &= size(tC) == (20,40,30,30,20,40)
testval = isapprox(reshape(C,prod(size(C))),tC.T)
fulltest &= testfct(testval,"*(A,B) [Array || tens]")

println()

import LinearAlgebra

A = rand(10,2,10)
B = LinearAlgebra.Diagonal(rand(10))

C = A*B
D = B*A

tC = tens(A)*B
tD = B*tens(A)

testval = isapprox(reshape(C,prod(size(C))),tC.T)
testval = isapprox(reshape(D,prod(size(D))),tD.T)

fulltest &= testfct(testval,"*(A,diagonal) [Array || tens, commutes]")

println()

C = contract(copy(A),3,B,1)
tC = rmul!(copy(tens(A)),B)

testval = isapprox(reshape(C,prod(size(C))),tC.T)
fulltest &= testfct(testval,"rmul!(A,diagonal)")

C = 3. * copy(A)
tC = rmul!(3.,copy(tens(A)))

testval = isapprox(reshape(C,prod(size(C))),tC.T)
fulltest &= testfct(testval,"rmul!(tens,number)")

println()

D = contract(B,2,copy(A),1)
tD = lmul!(B,copy(tens(A)))

testval = isapprox(reshape(D,prod(size(D))),tD.T)
fulltest &= testfct(testval,"lmul!(diagonal,tens)")

D = copy(A) * 3.
tD = lmul!(copy(tens(A)),3.)

testval = isapprox(reshape(D,prod(size(D))),tD.T)
fulltest &= testfct(testval,"lmul!(number,tens)")


println()

A = rand(ComplexF64,10,10)
B = rand(ComplexF64,10,10)

testval = isapprox(contract(A,2,B,1),A*B)
fulltest &= testfct(testval,"contract(Array,vec,Array,vec)")

testval = isapprox(ccontract(A,2,B,1),conj(A)*B)
fulltest &= testfct(testval,"ccontract(Array,vec,Array,vec)")

testval = isapprox(contractc(A,2,B,1),A*conj(B))
fulltest &= testfct(testval,"contractc(Array,vec,Array,vec)")

testval = isapprox(ccontractc(A,2,B,1),conj(A)*conj(B))
fulltest &= testfct(testval,"ccontractc(Array,vec,Array,vec)")

A = tens(A)
B = tens(B)

testval = isapprox(contract(A,2,B,1).T,(A*B).T)
fulltest &= testfct(testval,"contract(tens,vec,tens,vec)")

testval = isapprox(ccontract(A,2,B,1).T,(conj(A)*B).T)
fulltest &= testfct(testval,"ccontract(tens,vec,tens,vec)")

testval = isapprox(contractc(A,2,B,1).T,(A*conj(B)).T)
fulltest &= testfct(testval,"contractc(tens,vec,tens,vec)")

testval = isapprox(ccontractc(A,2,B,1).T,conj(A*B).T)
fulltest &= testfct(testval,"ccontractc(tens,vec,tens,vec)")

println()



A = rand(ComplexF64,10,20,30,40)
B = rand(ComplexF64,10,20,30,40)

C = transpose(reshape(A,prod(size(A))))*reshape(B,prod(size(B)))

testval = isapprox(contract(A,B),C)
fulltest &= testfct(testval,"contract(Array,Array)")

C = conj(transpose(reshape(A,prod(size(A)))))*reshape(B,prod(size(B)))

testval = isapprox(ccontract(A,B),C)
fulltest &= testfct(testval,"ccontract(Array,Array)")

C = (transpose(reshape(A,prod(size(A)))))*conj(reshape(B,prod(size(B))))

testval = isapprox(contractc(A,B),C)
fulltest &= testfct(testval,"contractc(Array,Array)")

C = conj(transpose(reshape(A,prod(size(A)))))*conj(reshape(B,prod(size(B))))

testval = isapprox(ccontractc(A,B),C)
fulltest &= testfct(testval,"ccontractc(Array,Array)")

A = tens(A)
B = tens(B)

C = dot(A,B,Lfct=identity,Rfct=identity)

testval = isapprox(contract(A,B),C)
fulltest &= testfct(testval,"contract(tens,tens)")

C = dot(A,B,Lfct=adjoint,Rfct=identity)

testval = isapprox(ccontract(A,B),C)
fulltest &= testfct(testval,"ccontract(tens,tens)")


C = dot(A,B,Lfct=identity,Rfct=adjoint)

testval = isapprox(contractc(A,B),C)
fulltest &= testfct(testval,"contractc(tens,tens)")


C = dot(A,B,Lfct=adjoint,Rfct=adjoint)

testval = isapprox(ccontractc(A,B),C)
fulltest &= testfct(testval,"ccontractc(tens,tens)")

println()

testval = contract(A,(1,2,3,4),B,(1,2,3,4))[1] == contract(A,(1,2,3,4),B)[1] == contract(A,B,(1,2,3,4))[1] 

fulltest &= testfct(testval,"contract(tens,[vec,]tens[,vec])")

testval = ccontract(A,(1,2,3,4),B,(1,2,3,4))[1] == ccontract(A,(1,2,3,4),B)[1] == ccontract(A,B,(1,2,3,4))[1] 

fulltest &= testfct(testval,"ccontract(tens,[vec,]tens[,vec])")

testval = contractc(A,(1,2,3,4),B,(1,2,3,4))[1] == contractc(A,(1,2,3,4),B)[1] == contractc(A,B,(1,2,3,4))[1] 

fulltest &= testfct(testval,"contractc(tens,[vec,]tens[,vec])")

testval = ccontractc(A,(1,2,3,4),B,(1,2,3,4))[1] == ccontractc(A,(1,2,3,4),B)[1] == ccontractc(A,B,(1,2,3,4))[1] 

fulltest &= testfct(testval,"ccontractc(tens,[vec,]tens[,vec])")


println()

A = rand(10,20,10,30#=,40,40=#);
B = trace(copy(A),[[1,3]#=,[5,6]=#])

C = makeId(copy(A),[1])
checkB = contract(copy(A),[1,3#=,5,6=#],C,[1,2#=,3,4=#])

testval = size(B) == (20,30) && isapprox(checkB.T,B.T)
fulltest &= testfct(testval,"trace(Array,[indices])")



A = rand(10,20,10,30,40,40);
B = trace(copy(A),[[1,3],[5,6]])

A = tens(A)
tB = trace(A,[[1,3],[5,6]])

tC = makeId(A,[1,5])
checkB = contract(A,[1,3,5,6],tC,[1,2,3,4])

testval = size(B) == (20,30) && isapprox(tB.T,checkB.T)
fulltest &= testfct(testval,"trace(tens,[indices])")

println()

A = rand(10,20,10,30#=,40,40=#);
B = trace!(copy(A),[[1,3]#=,[5,6]=#])

C = makeId(copy(A),[1])
checkB = contract(copy(A),[1,3#=,5,6=#],C,[1,2#=,3,4=#])

testval = size(B) == (20,30) && isapprox(checkB.T,B.T)
fulltest &= testfct(testval,"trace!(Array,[indices])")



A = rand(10,20,10,30,40,40);
B = trace!(copy(A),[[1,3],[5,6]])

A = tens(A)
tB = trace!(copy(A),[[1,3],[5,6]])

tC = makeId(A,[1,5])
checkB = contract(A,[1,3,5,6],tC,[1,2,3,4])

testval = size(B) == (20,30) && isapprox(tB.T,checkB.T)
fulltest &= testfct(testval,"trace!(tens,[indices])")
