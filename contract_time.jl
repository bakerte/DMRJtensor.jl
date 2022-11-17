path = "../"
include(join([path,"DMRjulia.jl"]))

using BenchmarkTools

@makeQNs "spin" U1
QS = 2

chi = [spin(2),spin(0),spin(0),spin(-2)]
#chi = qindtype[-2 0 0 2]
QNs = [chi,chi]
Arrows = [false,true]

d = size(chi,1)

msize = convert(Int64,d^(size(QNs,1)/2))

println("#####################################")
println("#####################################")
println("       $msize x $msize        (two rank ",size(QNs,1),"s)")
println("#####################################")
println("#####################################")

A = rand(QNs,Arrows)
W = deepcopy(A) #B = rand(QNs,Arrows)
#checkflux(A)

@time B = contract(A,[2],W,[1])
@btime B = contract(A,[2],W,[1])
println()

QNs = [chi,chi,chi,chi]
Arrows = [false,false,true,true]

msize = convert(Int64,d^(size(QNs,1)/2))

println("#####################################")
println("#####################################")
println("       $msize x $msize        (two rank ",size(QNs,1),"s)")
println("#####################################")
println("#####################################")

C = rand(QNs,Arrows)
G = deepcopy(C)
#checkflux(C)

#H = rand(QNs,Arrows)
@time D = contract(C,[3,4],G,[1,2])
@btime D = contract(C,[3,4],G,[1,2])

println()

QNs = [chi,chi,chi,chi,chi,chi]
Arrows = [false,false,false,true,true,true]

msize = convert(Int64,d^(size(QNs,1)/2))

println("#####################################")
println("#####################################")
println("       $msize x $msize        (two rank ",size(QNs,1),"s)")
println("#####################################")
println("#####################################")

E = rand(QNs,Arrows)
P = deepcopy(E)

@time F = contract(E,[4,5,6],P,[1,2,3])
@btime F = contract(E,[4,5,6],P,[1,2,3])
@btime F = contract(P,[1,2,3],E,[4,5,6])
println()

QNs = [chi,chi,chi,chi,chi,chi,chi,chi]
Arrows = [false,false,false,false,true,true,true,true]

msize = convert(Int64,d^(size(QNs,1)/2))

println("#####################################")
println("#####################################")
println("       $msize x $msize        (two rank ",size(QNs,1),"s)")
println("#####################################")
println("#####################################")

G = rand(QNs,Arrows)
M = deepcopy(G) #rand(QNs,Arrows)

@time H = contract(G,[5,6,7,8],M,[1,2,3,4])
@btime H = contract(G,[5,6,7,8],M,[1,2,3,4])
@btime H = contract(M,[1,2,3,4],G,[5,6,7,8])
println()

QNs = [chi,chi,chi,chi,chi,chi,chi,chi,chi,chi]
Arrows = [false,false,false,false,false,true,true,true,true,true]

msize = convert(Int64,d^(size(QNs,1)/2))

println("#####################################")
println("#####################################")
println("       $msize x $msize        (two rank ",size(QNs,1),"s)")
println("#####################################")
println("#####################################")

J = rand(QNs,Arrows)
R = deepcopy(J)
#Z = rand(QNs,Arrows)

@time K = contract(J,[6,7,8,9,10],R,[1,2,3,4,5])
@btime K = contract(J,[6,7,8,9,10],R,[1,2,3,4,5])

@btime K = contract(R,[1,2,3,4,5],J,[6,7,8,9,10])
println()

QNs = [chi,chi,chi,chi,chi,chi,chi,chi,chi,chi,chi,chi]
Arrows = [false,false,false,false,false,false,true,true,true,true,true,true]

msize = convert(Int64,d^(size(QNs,1)/2))

println("#####################################")
println("#####################################")
println("       $msize x $msize        (two rank ",size(QNs,1),"s)")
println("#####################################")
println("#####################################")

L = rand(QNs,Arrows)
S = deepcopy(L)
#F = rand(QNs,Arrows)

@time M = contract(L,[7,8,9,10,11,12],S,[1,2,3,4,5,6])
@btime M = contract(L,[7,8,9,10,11,12],S,[1,2,3,4,5,6])

@btime M = contract(S,[1,2,3,4,5,6],L,[7,8,9,10,11,12])
println()


QNs = [chi,chi,chi,chi,chi,chi,chi,chi,chi,chi,chi,chi,chi,chi]
Arrows = [false,false,false,false,false,false,false,true,true,true,true,true,true,true]

msize = convert(Int64,d^(size(QNs,1)/2))

println("#####################################")
println("#####################################")
println("       $msize x $msize        (two rank ",size(QNs,1),"s)")
println("#####################################")
println("#####################################")

N = rand(QNs,Arrows)
U = deepcopy(N)

@time P = contract(N,[8,9,10,11,12,13,14],U,[1,2,3,4,5,6,7])
@btime P = contract(N,[8,9,10,11,12,13,14],U,[1,2,3,4,5,6,7])
println()

@time aP = contract(U,[1,2,3,4,5,6,7],N,[8,9,10,11,12,13,14])
@btime aP = contract(U,[1,2,3,4,5,6,7],N,[8,9,10,11,12,13,14])
