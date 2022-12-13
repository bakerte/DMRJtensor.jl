println("#            +-------------+")
println("#>-----------|  Qlinearalgebra.jl  |-----------<")
println("#            +-------------+")
fulltest = true

import LinearAlgebra

@makeQNs "test5" U1
QS = 2

chi = [test5(2),test5(0),test5(0),test5(-2)]

numdims = 10

QNs = [chi for g = 1:numdims]
S = div(numdims,2)
Arrows = vcat([false for g = 1:S],[true for g = 1:S])

d = size(chi,1)

msize = convert(Int64,d^(size(QNs,1)/2))

A = rand(QNs,Arrows)
B = rand(QNs,Arrows)

Ltup = [i for i = 1:S]
Rtup = [i+S for i = 1:S]

tA = tens(makeArray(A))
tB = tens(makeArray(B))

@time C = contract(A,Ltup,B,Rtup)
@time tC = contract(tA,Ltup,tB,Rtup)

testval = isapprox(makeArray(C),makeArray(tC))
fulltest &= testfct(testval,"contract on dense and Qtens")

println()

@time C = ccontract(A,B)
@time tC = ccontract(tA,tB)

testval = isapprox(C,tC)
fulltest &= testfct(testval,"dot on dense and Qtens")

println()

@time U,D,V = svd(A,[Ltup,Rtup],nozeros=false)
@time tU,tD,tV = svd(tA,[Ltup,Rtup],nozeros=false)

C = reshape(makeArray(tA),length(chi)^length(Ltup),length(chi)^length(Rtup))
@time checkU,checkD,checkV = LinearAlgebra.svd(C)

rA = U*D*V
rtA = tU*tD*tV

#testval = isapprox(makeArray(A),makeArray(tA))
testval = isapprox(makeArray(rA),makeArray(rtA))
#testval &= isapprox(makeArray(),makeArray(tA))
fulltest &= testfct(testval,"svd on dense and Qtens")

println()

@time U,D,V = svd(A,[Ltup,Rtup])
@time tU,tD,tV = svd(tA,[Ltup,Rtup])

C = reshape(makeArray(tA),length(chi)^length(Ltup),length(chi)^length(Rtup))
@time checkU,checkD,checkV = LinearAlgebra.svd(C)

rA = U*D*V
rtA = tU*tD*tV

#testval = isapprox(makeArray(A),makeArray(tA))
testval = isapprox(makeArray(rA),makeArray(rtA))
#testval &= isapprox(makeArray(),makeArray(tA))
fulltest &= testfct(testval,"svd on dense and Qtens [nozeros=true]")

println()

@time U,D,V = svd(A,m=10,[Ltup,Rtup])
@time tU,tD,tV = svd(tA,m=10,[Ltup,Rtup])

rA = U*D*V
rtA = tU*tD*tV

testval = isapprox(makeArray(rA),makeArray(rtA))
fulltest &= testfct(testval,"truncating svd on dense and Qtens")
