
ndim = max(round(Int64,rand()*10),2)
Asize = ntuple(w->max(2,round(Int64,rand()*10)),ndim)
A = rand(Asize...)

println("#            +--------------+")
println("#>-----------|  Qtensor.jl  |-----------<")
println("#            +--------------+")
fulltest = true

import LinearAlgebra


@makeQNs "test" U1 U1 Zn{2}

nqdim = 4
Qlabels = [[inv(test(2,-2,1)),test(),test(),test(2,-2,1)] for i = 1:nqdim]

A = Qtens(Qlabels)

testval = A.size == [[i] for i = 1:nqdim]
testval &= typeof(A.T) <: Union{Array{Array{Float64,2},1},Array{LinearAlgebra.Diagonal{Float64,Vector{Float64}},1}}
testval &= typeof(A.ind) <: Vector{Tuple{Matrix{intType}, Matrix{intType}}}
testval &= typeof(A.currblock) <: NTuple{2,Array{intType,1}}
testval &= typeof(A.QnumMat) <: Array{Array{intType,1},1}
testval &= typeof(A.QnumSum) <: Array{Array{test,1},1}
testval &= typeof(A.flux) <: test

fulltest &= testfct(testval,"Qtens(Qlabels)")

println()

QNsummary,leftSummary,rightSummary,newQblocksum = DMRjulia.makeQNsummaries(Qlabels,[1,2],[3,4],false,test())

testval = typeof(QNsummary) <: Vector{test}
testval &= typeof(leftSummary) <: Vector{test}
testval &= typeof(rightSummary) <: Vector{test}
testval &= typeof(newQblocksum) <: Vector{Tuple{test,test}}
fulltest &= testfct(testval,"makeQNsummaries(Qlabels)")

QNsummary,leftSummary,rightSummary,newQblocksum = DMRjulia.makeQNsummaries(A,[1,2],[3,4],false)

testval = typeof(QNsummary) <: Vector{test}
testval &= typeof(leftSummary) <: Vector{test}
testval &= typeof(rightSummary) <: Vector{test}
testval &= typeof(newQblocksum) <: Vector{Tuple{test,test}}
fulltest &= testfct(testval,"makeQNsummaries(Qtensor)")

println()

B,C = DMRjulia.convertQnumMat(Qlabels)
testval = A.QnumSum == C
testval &= A.QnumMat == B
fulltest &= testfct(testval,"convertQnumMat(Qlabels)")

B = DMRjulia.convertQnumMat(Qlabels,A.QnumSum)
testval &= A.QnumMat == B
fulltest &= testfct(testval,"convertQnumMat(Qlabels,QnumSum)")

println()

B = DMRjulia.undefMat(ComplexF64,10,10)
testval = size(B) == (10,10)
testval &= typeof(B) <: Array{ComplexF64,2}
fulltest &= testfct(testval,"undefMat(DataType,x,y)")

println()

B = Qtens(Qlabels,[false for i = 1:length(Qlabels)])

newQlabels = fullQnumMat(B)
global testval2 = true
for i = 1:length(Qlabels)
  global testval2 &= newQlabels[i] + Qlabels[i] == [test() for g = 1:length(newQlabels[i])]
end
fulltest &= testfct(testval2,"Qtens(Qlabels,arrows)")

println()

testval = ([1,2],[3,4]) == DMRjulia.equalblocks(Qlabels)
fulltest &= testfct(testval,"equalblocks(Array{Array{Qnum,1},1})")

testval = ([1,2],[3,4]) == DMRjulia.equalblocks(size(A))
fulltest &= testfct(testval,"equalblocks(Tuple)")

testval = ([1,2],[3,4]) == DMRjulia.equalblocks(A)
fulltest &= testfct(testval,"equalblocks(Qtens)")

println()

B = DMRjulia.recoverQNs(1,A.QnumMat,A.QnumSum)
testval = B == Qlabels[1]
fulltest &= testfct(testval,"recoverQNs(1,QnumMat,QnumSum)")

B = DMRjulia.recoverQNs(1,A)
testval = B == Qlabels[1]
fulltest &= testfct(testval,"recoverQNs(1,Qtens)")

println()

B = fullQnumMat(A.QnumMat,A.QnumSum)
testval = B == Qlabels
fulltest &= testfct(testval,"fullQnumMat(QnumMat,QnumSum)")

B = fullQnumMat(A)
testval = B == Qlabels
fulltest &= testfct(testval,"fullQnumMat(Qtens)")

println()

B = rand(4,4,4,4)
C = Qtens(B,A)
#don't test norms since they will not be equal...B was not set up for blocks ideall, so some elements were excluded
testval = C.QnumMat == A.QnumMat
testval &= C.QnumSum == A.QnumSum
fulltest &= testfct(testval,"Qtens(Array,Qtens)")

B = rand(4,4,4,4)
C = Qtens(tens(B),A)
#don't test norms since they will not be equal...B was not set up for blocks ideall, so some elements were excluded
testval = C.QnumMat == A.QnumMat
testval &= C.QnumSum == A.QnumSum
fulltest &= testfct(testval,"Qtens(tens,Qtens)")

C = Qtens(B,Qlabels,[true for i = 1:length(Qlabels)])
testval = C.QnumMat == A.QnumMat
testval &= C.QnumSum == A.QnumSum
fulltest &= testfct(testval,"Qtens(tens,Qlabels)")

println()

C,C2 = Qtens(Qlabels,B,B)
testval = C.QnumMat == A.QnumMat
testval &= C.QnumSum == A.QnumSum
testval &= C2.QnumMat == A.QnumMat
testval &= C2.QnumSum == A.QnumSum
fulltest &= testfct(testval,"Qtens(Qlabels,Op...)")

B = rand(4,4)
C,C2 = Qtens(Qlabels[1],B,B)
D = fullQnumMat(C)
testval = D == [Qlabels[1],inv.(Qlabels[1])]
fulltest &= testfct(testval,"Qtens(Qlabel,Op...)")

println()

C = Qtens(tens(B),[Qlabels[1],inv.(Qlabels[1])])
testval = C2.QnumMat == C.QnumMat
testval &= C2.QnumSum == C.QnumSum
fulltest &= testfct(testval,"Qtens(tens,Op...)")

C = Qtens(tens(B),[Qlabels[1],inv.(Qlabels[1])],[false,false])
D = fullQnumMat(C)
testval = fullQnumMat(C2) == [inv.(D[i]) for i = 1:length(D)]
fulltest &= testfct(testval,"Qtens(operator,Qlabels,arrows)")

println()

B = Qtens(A)

testval = B.size == A.size
testval &= B.Qblocksum == A.Qblocksum
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"Qtens(Qtens)")

println()

B = rand(Qlabels)

testval = B.size == A.size
testval &= B.Qblocksum == A.Qblocksum
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"rand(Qlabels)")



B = rand(ComplexF64,Qlabels)

testval = eltype(B) <: ComplexF64
fulltest &= testfct(testval,"rand(ComplexF64,Qlabels)")


B = rand(Qlabels,[false for i = 1:length(Qlabels)])
C = fullQnumMat(B)

testval = Qlabels == [inv.(C[i]) for i = 1:length(Qlabels)]
fulltest &= testfct(testval,"rand(Qlabels,arrows)")

B = rand(A)
testval = eltype(B) == eltype(A)
testval &= B.Qblocksum == A.Qblocksum
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"rand(Qtens)")

B = rand(ComplexF64,A)
testval = eltype(B) != eltype(A)
testval &= B.Qblocksum == A.Qblocksum
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"rand(ComplexF64,Qtens)")

println()

B = basesize(Qlabels)

testval = B == ntuple(i->length(Qlabels[i]),nqdim)
fulltest &= testfct(testval,"basesize(Qlabels)")

C = basesize(A)
testval = C == ntuple(i->length(Qlabels[i]),nqdim)
fulltest &= testfct(testval,"basesize(Qtens)")

println()

B = DMRjulia.basedims(A)
testval = B == nqdim
fulltest &= testfct(testval,"basedims(Qtens)")

println()

B = zeros(Qlabels)
testval = isapprox(norm(B),0)
testval &= B.Qblocksum == A.Qblocksum
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"zeros(Qlabels)")

B = zeros(Qlabels,[false for i = 1:nqdim])
C = fullQnumMat(B)
testval = Qlabels == [inv.(C[i]) for i = 1:length(Qlabels)]
fulltest &= testfct(testval,"rand(Qlabels,arrows)")

B = zeros(ComplexF64,Qlabels)
testval = isapprox(norm(B),0) && eltype(B) <: ComplexF64
testval &= B.Qblocksum == A.Qblocksum
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"zeros(ComplexF64,Qlabels)")

B = zeros(A)
testval = isapprox(norm(B),0)
testval &= B.Qblocksum == A.Qblocksum
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"zeros(Qtens)")

B = zeros(ComplexF64,A)
testval = isapprox(norm(B),0) && eltype(B) <: ComplexF64
testval &= B.Qblocksum == A.Qblocksum
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"zeros(ComplexF64,Qtens)")

println()

B = zero(A)
testval = isapprox(norm(B),0)
testval &= B.Qblocksum == A.Qblocksum
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"zeros(Qtens)")

println()

B = Qtens{ComplexF64,test}()
testval = eltype(B) <: ComplexF64
testval &= typeof(B.flux) <: test
fulltest &= testfct(testval,"Qtens{ComplexF64,Qnum}())")

println()

mA,mB = checkType(A,B)
testval = eltype(mA) == eltype(mB)
fulltest &= testfct(testval,"checkType(Qtens,Qtens))")

println()

B = convertTens(ComplexF64,A)
testval = eltype(mB) <: ComplexF64
fulltest &= testfct(testval,"convertTens(Qtens,Qtens))")

println()

B = convertTens(ComplexF64,A)
testval = eltype(mB) <: ComplexF64
fulltest &= testfct(testval,"convertTens(Qtens,Qtens))")

println()

B = copy!(A)
testval = A.size == B.size
testval &= B.Qblocksum == A.Qblocksum
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"copy!(Qtens))")

println()

B = copy(A)
testval = A.size == B.size
testval &= B.Qblocksum == A.Qblocksum
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"copy(Qtens))")

println()

QNsummary,leftSummary,rightSummary,newQblocksum = DMRjulia.makeQNsummaries(Qlabels,[1,2],[3,4],false,test())

leftQNs,Lbigtosub,rows,Lindexes,rightQNs,Rbigtosub,columns,Rindexes = DMRjulia.QnumList(A,[1,2],[3,4],leftSummary,rightSummary)

testval = leftQNs == [5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1]
testval &= rightQNs == [1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5]
testval &= Lbigtosub == [1, 1, 2, 1, 3, 2, 3, 1, 4, 4, 5, 2, 6, 3, 4, 1]
testval &= Rbigtosub == [1, 1, 2, 1, 3, 2, 3, 1, 4, 4, 5, 2, 6, 3, 4, 1]
testval &= rows == [1, 4, 6, 4, 1]
testval &= columns == [1, 4, 6, 4, 1]
testval &= Lindexes == [A.ind[i][1] for i = 1:length(A.ind)]
testval &= Rindexes == [A.ind[i][2] for i = 1:length(A.ind)]

fulltest &= testfct(testval,"QnumList(Qtensor,Linds,Rinds,leftSummary,rightSummary)")



Lsizes = [length(A.QnumMat[p]) for p in [1,2]]
Rsizes = [length(A.QnumMat[p]) for p in [3,4]]

leftQNs,Lbigtosub,rows,Lindexes,rightQNs,Rbigtosub,columns,Rindexes = DMRjulia.QnumList(Lsizes,Rsizes,A.QnumMat,A.QnumSum,[1,2],[3,4],leftSummary,rightSummary)

testval = leftQNs == [5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1]
testval &= rightQNs == [1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5]
testval &= Lbigtosub == [1, 1, 2, 1, 3, 2, 3, 1, 4, 4, 5, 2, 6, 3, 4, 1]
testval &= Rbigtosub == [1, 1, 2, 1, 3, 2, 3, 1, 4, 4, 5, 2, 6, 3, 4, 1]
testval &= rows == [1, 4, 6, 4, 1]
testval &= columns == [1, 4, 6, 4, 1]
testval &= Lindexes == [A.ind[i][1] for i = 1:length(A.ind)]
testval &= Rindexes == [A.ind[i][2] for i = 1:length(A.ind)]

fulltest &= testfct(testval,"QnumList(Lsizes,Rsizes,QnumMat,QnumSum,Linds,Rinds,leftSummary,rightSummary)")

println()

nonzero_sizes = [true for i = 1:nqdim]

DMRjulia.loadindex(nonzero_sizes,2,[4,4],Lindexes,leftQNs,Lbigtosub,rows)

testval = Lindexes == [A.ind[i][1] for i = 1:length(A.ind)]
fulltest &= testfct(testval,"loadindex(nonzero_sizes,Linds,Lsizes,Lindexes,leftQNs,Lbigtosub,rows)")

println()

leftQNs,Lbigtosub,rows = DMRjulia.QnumList(Lsizes,A.QnumMat,A.QnumSum,[1,2],leftSummary)

testval = leftQNs == [5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1]
testval &= Lbigtosub == [1, 1, 2, 1, 3, 2, 3, 1, 4, 4, 5, 2, 6, 3, 4, 1]
testval &= rows == [1, 4, 6, 4, 1]
testval &= Lindexes == [A.ind[i][1] for i = 1:length(A.ind)]

fulltest &= testfct(testval,"QnumList(Lsizes,QnumMat,QnumSum,Linds,leftSummary)")

println()

leftQNs,Lbigtosub,rows,Lindexes,rightQNs,Rbigtosub,columns,Rindexes = DMRjulia.QnumList(A,[1,2],[3,4],leftSummary,rightSummary)

Lindexes,Rindexes,leftQNs,rightQNs,rows,columns = DMRjulia.makeIndexes([1,2],[4,4],leftQNs,Lbigtosub,rows,[3,4],[4,4],rightQNs,Rbigtosub,columns)

testval = leftQNs == [5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1]
testval &= rightQNs == [1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5]
testval &= rows == [1, 4, 6, 4, 1]
testval &= columns == [1, 4, 6, 4, 1]
testval &= Lindexes == [A.ind[i][1] for i = 1:length(A.ind)]
testval &= Rindexes == [A.ind[i][2] for i = 1:length(A.ind)]

fulltest &= testfct(testval,"makeIndexes(Lsizes,QnumMat,QnumSum,Linds,leftSummary)")

println()

B = multi_indexsummary(A.QnumSum,[1,2])

testval = B == reverse(leftSummary)
fulltest &= testfct(testval,"multi_indexsummary(QnumSum,vec)")

C = multi_indexsummary(A,[1,2])

testval = B == C
fulltest &= testfct(testval,"multi_indexsummary(QnumSum,vec)")

println()

B = DMRjulia.findsizes(A,[1,2])
testval = B == (2,2,true)
fulltest &= testfct(testval,"findsizes(Qtens,vec)")

println()

B = DMRjulia.checkorder!(copy(A),1)
testval = B == [A.ind[i][1] for i = 1:length(A.ind)]
fulltest &= testfct(testval,"checkorder!(Qtens,integer)")

C = copy(A)
DMRjulia.checkorder!(C)
testval = C.ind == A.ind
fulltest &= testfct(testval,"checkorder!(Qtens)")

println()

B = DMRjulia.LRsummary_invQ(leftSummary,A.flux)
testval = B == (leftSummary,rightSummary)
fulltest &= testfct(testval,"LRsummary_invQ(QNsummary,flux)")

println()

A = rand(A)

prenorm = norm(A)
C = changeblock(A,[1,2,3],[4])
testval = isapprox(prenorm,norm(C))

B = changeblock(C,[1,2],[3,4])

testval = isapprox(norm(B),prenorm)
testval &= B.size == A.size
testval &= sort([B.Qblocksum[i][1] for i = 1:length(B.Qblocksum)]) == sort([A.Qblocksum[i][1] for i = 1:length(A.Qblocksum)])
testval &= sort([B.Qblocksum[i][2] for i = 1:length(B.Qblocksum)]) == sort([A.Qblocksum[i][2] for i = 1:length(A.Qblocksum)])
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"changeblock(Qtens,vec,vec)")



C = changeblock(A,[[1,2,3],[4]])
testval = isapprox(prenorm,norm(C))

B = changeblock(C,[[1,2],[3,4]])

testval = isapprox(norm(B),prenorm)
testval &= B.size == A.size
testval &= sort([B.Qblocksum[i][1] for i = 1:length(B.Qblocksum)]) == sort([A.Qblocksum[i][1] for i = 1:length(A.Qblocksum)])
testval &= sort([B.Qblocksum[i][2] for i = 1:length(B.Qblocksum)]) == sort([A.Qblocksum[i][2] for i = 1:length(A.Qblocksum)])
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"changeblock(Qtens,[vec,vec])")



C = changeblock(A,([1,2,3],[4]))
testval = isapprox(prenorm,norm(C))

B = changeblock(C,([1,2],[3,4]))

testval = isapprox(norm(B),prenorm)
testval &= B.size == A.size
testval &= sort([B.Qblocksum[i][1] for i = 1:length(B.Qblocksum)]) == sort([A.Qblocksum[i][1] for i = 1:length(A.Qblocksum)])
testval &= sort([B.Qblocksum[i][2] for i = 1:length(B.Qblocksum)]) == sort([A.Qblocksum[i][2] for i = 1:length(A.Qblocksum)])
testval &= B.currblock == A.currblock
testval &= B.QnumMat == A.QnumMat
testval &= B.QnumSum == A.QnumSum
testval &= B.flux == A.flux
fulltest &= testfct(testval,"changeblock(Qtens,(vec,vec))")

println()

testval = 10 == DMRjulia.AAzeropos2ind([1,2,3],(4,5,6),(1,2))
fulltest &= testfct(testval,"AAzeropos2ind(vec,tup,tup)")

println()

B = rand(Int64,3,10)
C = rand(Int64,3)
order = [1,3,2]

DMRjulia.innerloadpos!(4,3,C,order,B)
testval = C == B[order,4]
fulltest &= testfct(testval,"innerloadpos!(int,int,vec,vec,Matrix)")

println()

#innerloop()

#doubleloop_right()

#doubleloop_left()

#doubleloop_reg()

#reblock!()

#checkzeroblocks!()

#countload(

#newindexsizeone!()

println()


testval = size(A) == (4,4,4,4)

B = reshape!(copy(A),[[1,2],[3,4]])
C = reshape!(copy(A),16,16)

testval = size(B) == size(C) == (16,16) && C.size == B.size
fulltest &= testfct(testval,"reshape!(...)")

println()

testval = size(A) == (4,4,4,4)

B = reshape(copy(A),[[1,2],[3,4]])
C = reshape(copy(A),16,16)

testval = size(B) == size(C) == (16,16) && C.size == B.size
fulltest &= testfct(testval,"reshape(...)")

println()

testval = A.QnumSum[1][1] == getQnum(1,1,A.QnumMat,A.QnumSum)
fulltest &= testfct(testval,"getQnum(x,y,QnumMat,QnumSum)")

testval = A.QnumSum[1][1] == getQnum(1,1,A)
fulltest &= testfct(testval,"getQnum(x,y,Qtens)")

println()
#makenewindsL
#makenewindsR
#mergeQNloop!()
#mergereshape!
#mergereshape()

println()

#unreshape!(
#unreshape(


println()
#lastindex()


#matchblocks()

#loadtup!()
#findextrablocks()

println()

B = -A
testval = isapprox(norm(A+B),0)
fulltest &= testfct(testval,"-(Qtens)")

println()

#tensorcombination!()

#tensorcombination()

#invmat!()
#exp!()
#exp()
#metricdistance
#sum
#norm
#eltype
#elnumtype
#conj
#conj!
#ndims
#size
#recoverShape
#permutedims
#permutedims!

#isapprox

println()
#evaluate_keep
#truncate_replace_inds

#get_ranges

println()

#isinteger

println()

#innerjoinloop
#getindex
#getindex!

#findmatch
#loadpos!

#setindex!

#findqsector
#scaninds
#searchindex
#getblockrows
#AAind2zeropos

#joinloop!
#firstloop!
#matchloop

#Bextraloop!
#makerowcol
#rowcolsort
#orderloop!


println()

#joinindex
#applylocalF!


#applylocalF

#getinds_loop_one
#getinds_loop_two

#getinds

#Idhelper

#makeId

#swapgate
#makedens
#makeArray


#showQtens

#print
#println

#checkflux

