
using BenchmarkTools

@makeQNs "spin" U1
QS = 2

chi = [spin(2),spin(0),spin(0),spin(-2)]

permuting = false

ninds = 12

QNs = [chi for x = 1:ninds]
halfsize = (ninds ÷ 2)
halfbool = [true for x = 1:halfsize]
nothalfbool = [false for x = 1:halfsize]
Arrows = vcat(nothalfbool,halfbool)

msize = length(chi)^halfsize

println("#####################################")
println("#####################################")
println("       $msize x $msize        (two rank ",size(QNs,1),"s)")
println("#####################################")
println("#####################################")

A = rand(QNs,Arrows)

Linds = [i for i = (halfsize+1):ninds]
Rinds = [i for i = 1:halfsize]

if permuting
  Rinds,Linds = Linds,Rinds
end

rA = reshape!(A,prod(size(A)[Linds]),prod(size(A)[Rinds]))

@time U,D,V = svd(rA)
@btime svd(rA)
println()
println("...now truncating svd")
@time U,D,V = svd(rA,m=40)
@btime U,D,V = svd(rA,m=40)

#U,D,V = svd(A,[Linds,Rinds])
#@btime svd(A,[Linds,Rinds])

