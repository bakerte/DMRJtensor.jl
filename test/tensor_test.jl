

ndim = max(round(Int64,rand()*10),2)
Asize = ntuple(w->max(2,round(Int64,rand()*10)),ndim)
A = rand(Asize...)

println("#            +-------------+")
println("#>-----------|  tensor.jl  |-----------<")
println("#            +-------------+")
fulltest = true

println("Array size (rank $ndim): ",Asize)

B = tA = tens(A)

sizetest = (B.size...,) == Asize
fulltest &= testfct(sizetest,"denstens .size test")
normtest = isapprox(norm(B.T),norm(A))
fulltest &= testfct(normtest,"denstens .A field")

println()
println("input tests; default input auto-pass, tens(Array), by this point")

B = tens(type=ComplexF64)
loadtest1 = B.size == Array{intType,1}(undef,0) && B.T == Array{ComplexF64,1}(undef,0)
fulltest &= testfct(loadtest1,"tens(;type=)")

B = tens(ComplexF64)
loadtest2 = B.size == Array{intType,1}(undef,0) && B.T == Array{ComplexF64,1}(undef,0)
fulltest &= testfct(loadtest2,"tens(type)")


import LinearAlgebra
size_vec = rand(ndim)
C = LinearAlgebra.Diagonal(size_vec)
B = tens(ComplexF64,C)

testval = B.size == [ndim,ndim] && isapprox(B.T,reshape(Array(C),ndim^2))
fulltest &= testfct(testval,"tens(Type,AbstractArray)")

B = tens(C)
testval = B.size == [ndim,ndim] && isapprox(B.T,reshape(Array(C),ndim^2))
fulltest &= testfct(testval,"tens(AbstractArray)")

B = tens(ComplexF64,A)
testval = B.size == tA.size && isapprox(B.T,convert(Array{ComplexF64,1},reshape(A,prod(size(A)))))
fulltest &= testfct(testval,"tens(Type,Array{DiffType})")

B = tens(Float64,A)
testval = B.size == tA.size && isapprox(B.T,convert(Array{ComplexF64,1},tA.T))
fulltest &= testfct(testval,"tens(Type,Array{Type})")


B = tens{ComplexF64}(A)
testval = B.size == tA.size && isapprox(B.T,convert(Array{ComplexF64,1},tA.T))
fulltest &= testfct(testval,"tens{DiffType}(Array{Type})")

B = tens{ComplexF64}(tA)
testval = B.size == tA.size && isapprox(B.T,convert(Array{ComplexF64,1},tA.T))
fulltest &= testfct(testval,"tens{DiffType}(tens{Type})")

B = tens{Float64}(tA)
testval = B.size == tA.size && isapprox(B.T,convert(Array{Float64,1},tA.T))
fulltest &= testfct(testval,"tens{Type}(tens{Type})")

println()

B = rand(tA)
testval = B.size == tA.size
fulltest &= testfct(testval,"rand(denstens)")

B = rand(C)
testval = size(B) == size(C)
fulltest &= testfct(testval,"rand(AbstractArray")

println()

B = zeros(C)
testval = size(B) == size(C)
fulltest &= testfct(testval,"zeros(AbstractArray)")

B = zeros(tA)
testval = size(B) == size(tA)
fulltest &= testfct(testval,"zeros(denstens)")

B = zeros(ComplexF64,C)
testval = size(B) == size(C) && eltype(B) == ComplexF64
fulltest &= testfct(testval,"zeros(ComplexF64,AbstractArray)")

B = zeros(ComplexF64,tA)
testval = size(B) == size(tA) && eltype(B) == ComplexF64
fulltest &= testfct(testval,"zeros(ComplexF64,denstens)")

println()

B = zero(tA)
testval = sum(B.T) == 0.0 && size(B) == size(tA)
fulltest &= testfct(testval,"zero(denstens)")

println()

ldim = round(Int64,rand(5:40,1)[1])
rdim = round(Int64,rand(5:40,1)[1])
println("  (ldim,rdim) = (",ldim,",",rdim,")")
A = DMRjulia.makeIdarray(ComplexF64,ldim,rdim)
mindim = min(ldim,rdim)
testval = size(A) == (ldim,rdim) && isapprox(sum(A),mindim) && eltype(A) == ComplexF64 && isapprox(A[1:mindim,1:mindim],zeros(mindim,mindim) + LinearAlgebra.I)
fulltest &= testfct(testval,"makeIdarray")

println()

A = makeId(ComplexF64,ldim,addone=true,addRightDim=false)
testval = size(A) == (1,ldim,ldim) && isapprox(sum(A),ldim) && eltype(A) == ComplexF64 && isapprox(makeArray(A[1,1:mindim,1:mindim]),zeros(mindim,mindim) + LinearAlgebra.I)
fulltest &= testfct(testval,"makeId (square, left)")

A = makeId(ComplexF64,ldim,rdim,addone=true,addRightDim=true)
testval = size(A) == (ldim,rdim,1) && isapprox(sum(A),mindim) && eltype(A) == ComplexF64 && isapprox(makeArray(A[1:mindim,1:mindim,1]),zeros(mindim,mindim) + LinearAlgebra.I)
fulltest &= testfct(testval,"makeId (square, right)")

A = makeId(ComplexF64,ldim,rdim)
testval = size(A) == (ldim,rdim) && isapprox(sum(A),mindim) && eltype(A) == ComplexF64 && isapprox(makeArray(A[1:mindim,1:mindim]),zeros(mindim,mindim) + LinearAlgebra.I)
fulltest &= testfct(testval,"makeId (rectangular)")

println()
ldim,rdim = rdim,ldim
println("  (ldim,rdim) = (",ldim,",",rdim,")")
A = DMRjulia.makeIdarray(ComplexF64,ldim,rdim)
mindim = min(ldim,rdim)
testval = size(A) == (ldim,rdim) && isapprox(sum(A),mindim) && eltype(A) == ComplexF64 && isapprox(A[1:mindim,1:mindim],zeros(mindim,mindim) + LinearAlgebra.I)
fulltest &= testfct(testval,"makeIdarray")

println()

A = makeId(ComplexF64,ldim,addone=true,addRightDim=false)
testval = size(A) == (1,ldim,ldim) && isapprox(sum(A),ldim) && eltype(A) == ComplexF64 && isapprox(makeArray(A[1,1:mindim,1:mindim]),zeros(mindim,mindim) + LinearAlgebra.I)
fulltest &= testfct(testval,"makeId (square, left)")

A = makeId(ComplexF64,ldim,rdim,addone=true,addRightDim=true)
testval = size(A) == (ldim,rdim,1) && isapprox(sum(A),mindim) && eltype(A) == ComplexF64 && isapprox(makeArray(A[1:mindim,1:mindim,1]),zeros(mindim,mindim) + LinearAlgebra.I)
fulltest &= testfct(testval,"makeId (square, right)")

A = makeId(ComplexF64,ldim,rdim)
testval = size(A) == (ldim,rdim) && isapprox(sum(A),mindim) && eltype(A) == ComplexF64 && isapprox(makeArray(A[1:mindim,1:mindim]),zeros(mindim,mindim) + LinearAlgebra.I)
fulltest &= testfct(testval,"makeId (rectangular)")

A = rand(ldim,rdim,50,10)
B = makeId(A,[1,2,3])

testval = isapprox(sum(B),ldim*rdim*50) && size(B) == (ldim,ldim,rdim,rdim,50,50)

fulltest &= testfct(testval,"makeId (index inputs, [1,2,3])")

println()

B = convertTens(ComplexF64,tens(A))
testval = eltype(B) == ComplexF64 && size(B) == size(A) && norm(B) == norm(A) && typeof(B) <: denstens
fulltest &= testfct(testval,"convertTens(denstens)")

B = convertTens(ComplexF64,A)
testval = eltype(B) == ComplexF64 && size(B) == size(A) && norm(B) == norm(A)
fulltest &= testfct(testval,"convertTens(Array)")

println()

A = convIn([1,2,3])
testval = typeof(A) <: Tuple && length(A) == 3 && norm([A[i]-i for i = 1:3]) == 0
fulltest &= testfct(testval,"convIn(Array)")

A = convIn([1;2;3])
testval = typeof(A) <: Tuple && length(A) == 3 && norm([A[i]-i for i = 1:3]) == 0
fulltest &= testfct(testval,"convIn(Matrix)")

B = convIn(1)
testval = typeof(B) <: Tuple
fulltest &= testfct(testval,"convIn(Integer)")

A = (1,2,3)
B = convIn(A)
testval = A === B
fulltest &= testfct(testval,"convIn(Tuple)")

println()

nA = rand(9:100,1)[1]
vect = [rand(Bool) for i = 1:nA]
iA = Vector{Int64}(undef,sum(vect))
othervec = Vector{Int64}(undef,nA-length(iA))
counter = [0]
anticounter = [0]
for i = 1:nA
  if vect[i]
    counter[1] += 1
    iA[counter[1]] = i
  else
    anticounter[1] += 1
    othervec[anticounter[1]] = i
  end
end
B = findnotcons(nA,(iA...,))
testval = length(B) == nA-length(iA) && othervec == B
fulltest &= testfct(testval,"findnotcons")

println()

A = LinearAlgebra.Diagonal(rand(10))
B = tens(rand(ComplexF64,20,40,3))

C,D = checkType(A,B)
testval = typeof(C) <: denstens && eltype(C) == ComplexF64 && typeof(D) <: denstens && eltype(D) == ComplexF64
fulltest &= testfct(testval,"checkType(Diagonal,denstens)")

C,D = checkType(B,A)
testval = typeof(C) <: denstens && eltype(C) == ComplexF64 && typeof(D) <: denstens && eltype(D) == ComplexF64
fulltest &= testfct(testval,"checkType(denstens,Diagonal)")

C = checkType(A)
testval = typeof(C) <: Array && isapprox(norm(A),norm(C))
fulltest &= testfct(testval,"checkType(Diagonal)")

C = checkType(B)
testval = C === B
fulltest &= testfct(testval,"checkType(tens)")

A = tens(rand(20,3,40))
B = tens(rand(ComplexF64,20,40,3))
C,D = checkType(A,B)
testval = typeof(C) <: denstens && eltype(C) == ComplexF64 && typeof(D) <: denstens && eltype(D) == ComplexF64
fulltest &= testfct(testval,"checkType(denstens,denstens)")

println()

pos = makepos(ndim)
B = vcat([0],ones(Int64,ndim-1))

testval = pos == B
fulltest &= testfct(testval,"makepos(Integer)")

println()

S = ntuple(g->rand(5:10,1)[1],3)
vecS = [S[i] for i = 1:3]
currposvec = [makepos(3) for i = 1:prod(S)]

x = [i for i = 1:prod(S)]
y = Vector{intType}(undef,prod(S))

testval = [true]

for k = 1:length(currposvec)
  DMRjulia.ind2pos!(currposvec,k,x,k,S)
  val = pos2ind((currposvec[k]...,),S)
  val2 = pos2ind((currposvec[k]...,),vecS)
  pos2ind!(y,k,currposvec[k],S)
  testval[1] &= val == x[k] == val2 && y[k] == x[k]
end
testval = currposvec[end] == vecS && testval[1]
fulltest &= testfct(testval,"ind2pos! && pos2ind(Vector) && pos2ind(Tuple) && pos2ind!")

println()

pos = makepos!(pos)
B = vcat([0],ones(Int64,ndim-1))
testval = pos == B
fulltest &= testfct(testval,"makepos!(Vector)")

println()

S = (10,20,10,8,10,11)
cols = (:,[1,2,3,4],(1,2,3,4,5),1:5,1:5,1)
fullcols = DMRjulia.get_denseranges(S,cols...)

testval = fullcols == [1:10,[1,2,3,4],[1,2,3,4,5],1:5,1:5,1]
fulltest &= testfct(testval,"get_denseranges(genColTypes)")

println()

A = rand(ComplexF64,10,20)
tA = tens(A)
B = transpose(tA)

testval = isapprox(norm(transpose(A) - makeArray(B)),0)
fulltest &= testfct(testval,"adjoint")

println()

A = rand(ComplexF64,10,20)
tA = tens(A)
B = adjoint(tA)

testval = isapprox(norm(A' - makeArray(B)),0)
fulltest &= testfct(testval,"adjoint")

println()

Asize = [rand(3:10,1)[1] for i = 1:ndim]
A = tens(rand(Asize...))
B = copy(A)
testval = !(A===B) && A.size == B.size && A.T == B.T && eltype(A) == eltype(B)
fulltest &= testfct(testval,"copy(denstens)")

println()

testval = length(A) == prod(A.size)
fulltest &= testfct(testval,"length(denstens)")

println()

testval = size(A) == (A.size...,)
fulltest &= testfct(testval,"size(denstens)")

testval = ntuple(n->size(A,n),length(A.size)) == size(A)
fulltest &= testfct(testval,"size(denstens,int)")

println()

A = rand(ComplexF64,Asize...)
tA = tens(A)
sumA = sum(A)
testval = isapprox(sumA,sum(tA))
fulltest &= testfct(testval,"sum(denstens)")

println()

testval = isapprox(norm(A),norm(tA))
fulltest &= testfct(testval,"norm(denstens)")

println()

testval = isapprox(sum(conj(tA)),sumA')
fulltest &= testfct(testval,"conj(denstens)")
testval = isapprox(sum(conj!(tA)),sumA')
fulltest &= testfct(testval,"conj!(denstens)")

println()

testval = length(tA.size) == ndims(tA)
fulltest &= testfct(testval,"ndims(denstens)")

println()

testval = lastindex(tA,1) == size(tA,1)
fulltest &= testfct(testval,"lastindex(denstens,i)")

println()

testval = eltype(tA) <: ComplexF64
fulltest &= testfct(testval,"eltype(denstens)")

println()

testval = elnumtype(tA) <: ComplexF64
fulltest &= testfct(testval,"enumtype(denstens)")

println()

S = (10,20,10,8,10,11)
A = rand(S...)
tA = tens(A)
testval = makeArray(tA[cols...]) == A[fullcols...]
fulltest &= testfct(testval,"getindex(genColtype)")

testval = tA[S...] == A[S...]
fulltest &= testfct(testval,"getindex(integer...)")
#=
A = LinearAlgebra.Diagonal(rand(10))
testval = A[:,[1,2,3]] == A[:,1:3]
fulltest &= testfct(testval,"getindex()")
=#
println()

B = searchindex(tA,1,2,3,4,5,6)
testval = B == searchindex(A,(1,2,3,4,5,6)) == A[1,2,3,4,5,6]
fulltest &= testfct(testval,"searchindex(denstens,integer...)")

println()

B = rand(2,2)
tB = tens(B)

C = copy(tA)
tA[1:2,[2,3],3,4,5,6] = tB
C[1:2,[2,3],3,4,5,6] = B
A[1:2,2:3,3,4,5,6] = tB
testval = B == makeArray(tA[1:2,2:3,3,4,5,6]) == makeArray(C[1:2,[2,3],3,4,5,6]) == A[1:2,2:3,3,4,5,6]
fulltest &= testfct(testval,"setindex(denstens,integer...)")

println()

A = rand(100,100)
B = zeros(100,100)
DMRjulia.loadM!(B,A)
testval = A == B
fulltest &= testfct(testval,"loadM!")

println()

A = rand(10,20,30,10)
tA = tens(A)
B = rand(10,20,30,10)
tB = tens(B)
checkC = 2*A+2*B

testval = makeArray(tensorcombination((2.,2.),tA,tB)) == tensorcombination(A,B,alpha=(2.,2.)) == checkC
fulltest &= testfct(testval,"tensorcombination(denstens || Array)")

println()

A = rand(10,20,30,10);
tA = tens(copy(A));
B = rand(10,20,30,10);
tB = tens(copy(B));
checkC = 2*A+2*B;
checktC = 2*tA+2*tB;

testtC = tensorcombination!((2.,2.),tA,tB)
testC = tensorcombination!(A,B,alpha=(2.,2.))

testval = norm(tA - checktC) == 0 && norm(tA - testtC) == 0 && norm(checkC-testC) == 0  # == A
fulltest &= testfct(testval,"tensorcombination!(denstens || Array)")

println()

B = mult!(3,tA)
C = mult!(3,A)

testval = norm(makeArray(B)-C) == 0
fulltest &= testfct(testval,"mult!(denstens || Array)")

println()

tA = tens(copy(A))
tC = tens(copy(C))
testval = isapprox(norm(add!(copy(A),C,7) - (A+7*C)),0) == isapprox(norm(makeArray(add!(copy(tA),tC,7)) - (A+7*C)),0)
fulltest &= testfct(testval,"add!(denstens || Array,number)")

tA = tens(copy(A))
tC = tens(copy(C))
testval = isapprox(norm(add!(copy(A),C) - (A+C)),0) == isapprox(norm(makeArray(add!(copy(tA),tC)) - (A+C)),0)
fulltest &= testfct(testval,"add!(denstens || Array)")

println()

tA = tens(copy(A))
tC = tens(copy(C))
testval = isapprox(norm(sub!(copy(A),C,7) - A-7*C),0) == isapprox(norm(makeArray(sub!(copy(tA),tC,7)) - A-7*C),0)
fulltest &= testfct(testval,"sub!(denstens || Array,number)")

tA = tens(copy(A))
tC = tens(copy(C))
testval = isapprox(norm(sub!(copy(A),C) - A-C),0) == isapprox(norm(makeArray(sub!(copy(tA),tC)) - A-C),0)
fulltest &= testfct(testval,"sub!(denstens || Array)")

println()

f = rand()
B = div!(copy(A),f)
C = A/f
testval = isapprox(norm(B-C),0)
fulltest &= testfct(testval,"div!(denstens || Array)")

println()

A = norm!(A)
testval = isapprox(norm(A),1)
fulltest &= testfct(testval,"norm!(TensType)")

println()

tA = tens(A)
tC = tens(C)
testval = isapprox(makeArray(tA+tC),A+C) && isapprox(makeArray(A+tC),makeArray(tA+C))
fulltest &= testfct(testval,"+(A,B))")

println()

testval = isapprox(makeArray(tA-tC),A-C) && isapprox(makeArray(A-tC),makeArray(tA-C))
fulltest &= testfct(testval,"-(A,B))")

println()

testval = isapprox(makeArray(tA/8),makeArray((1/8)*tA))
fulltest &= testfct(testval,"/(A,c)")

println()

testval = isapprox(sum(sqrt!(LinearAlgebra.Diagonal(ones(ndim)/ndim))^2),1)
fulltest &= testfct(testval,"sqrt!(Diagonal)")

println()

testval = isapprox(sum(sqrt(LinearAlgebra.Diagonal(ones(ndim)/ndim))^2),1)
fulltest &= testfct(testval,"sqrt(Diagonal)")

println()

testval = isapprox(sum(invmat(LinearAlgebra.Diagonal(ones(ndim)/ndim))),ndim^2)
fulltest &= testfct(testval,"invmat(Diagonal)")

testval = isapprox(sum(invmat(tens(LinearAlgebra.Diagonal(ones(ndim)/ndim)))),ndim^2)
fulltest &= testfct(testval,"invmat(denstens)")

testval = isapprox(sum(invmat(makeArray(tens(LinearAlgebra.Diagonal(ones(ndim)/ndim))))),ndim^2)
fulltest &= testfct(testval,"invmat(Array)")

println()

A = rand(10,10)
B = exp(A)
testval = isapprox(exp(copy(A)),B)
fulltest &= testfct(testval,"exp(Array)")

testval = isapprox(makeArray(exp(tens(copy(A)))),B)
fulltest &= testfct(testval,"exp(denstens)")

A = rand(10,10)
B = exp(2*A)
testval = isapprox(exp!(copy(A),2),B) 
fulltest &= testfct(testval,"exp(Array,prefactor)")

testval = isapprox(exp(A,2),B)
fulltest &= testfct(testval,"exp(denstens,prefactor)")

println()

alpha = rand(10)
beta = rand(9)
testval = isapprox(exp(2*LinearAlgebra.SymTridiagonal(alpha,beta)),exp(alpha,beta,2))
fulltest &= testfct(testval,"exp(alpha,beta,prefactor)")

testval = isapprox(exp(LinearAlgebra.SymTridiagonal(alpha,beta)),exp(alpha,beta))
fulltest &= testfct(testval,"exp(alpha,beta)")

testval = isapprox(exp(2*LinearAlgebra.SymTridiagonal(alpha,beta)),exp(LinearAlgebra.SymTridiagonal(alpha,beta),2))
fulltest &= testfct(testval,"exp(alpha,beta,prefactor)")

println()

A = rand(20,40,30,50)

rA = reshape(A,20,1200,50)
testval = isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)
fulltest &= testfct(testval,"reshape(Array,integer)")

rA = reshape(A,(20,1200,50))
testval = isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)
fulltest &= testfct(testval,"reshape(Array,tuple)")

rA = reshape(A,[20,1200,50])
testval = isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)
fulltest &= testfct(testval,"reshape(Array,Vector)")

tA = tens(rand(20,40,30,50))

rA = reshape(tA,20,1200,50)
testval = isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)
fulltest &= testfct(testval,"reshape(tens,integer)")

rA = reshape(tA,(20,1200,50))
testval = isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)
fulltest &= testfct(testval,"reshape(tens,tuple)")

rA = reshape(tA,[20,1200,50])
testval = isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)
fulltest &= testfct(testval,"reshape(tens,Vector)")

rA = reshape(copy(A),20,1200,50)
testval = isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)
fulltest &= testfct(testval,"reshape(Array,integer)")

rA = reshape(copy(A),(20,1200,50))
testval = isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)
fulltest &= testfct(testval,"reshape(Array,tuple)")

rA = reshape(copy(A),[20,1200,50])
testval = isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)
fulltest &= testfct(testval,"reshape(Array,Vector)")

rA = reshape(copy(tA),[[1],[2,3],[4]])
testval = isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)
fulltest &= testfct(testval,"reshape(tens,[[indexes]])")



tA = tens(rand(20,40,30,50))

rA = reshape(copy(tA),20,1200,50)
testval = isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)
fulltest &= testfct(testval,"reshape(tens,integer)")

rA = reshape(copy(tA),(20,1200,50))
testval = isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)
fulltest &= testfct(testval,"reshape!(tens,tuple)")

rA = reshape!(copy(tA),[20,1200,50])
testval = isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)
fulltest &= testfct(testval,"reshape!(tens,Vector)")

rA = reshape!(copy(tA),[[1],[2,3],[4]])
testval = isapprox(rA.T,tA.T) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)
fulltest &= testfct(testval,"reshape!(tens,[[indexes]])")

println()

rA = unreshape(copy(A),20,1200,50)
testval = isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)
fulltest &= testfct(testval,"unreshape(Array,Vector)")

rA = unreshape(copy(tA),20,1200,50)
testval = isapprox(norm(rA.T),norm(tA.T)) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)
fulltest &= testfct(testval,"unreshape(tens,Vector)")

rA = unreshape!(copy(A),20,1200,50)
testval = isapprox(norm(rA),norm(A)) && size(rA) == (20,1200,50) && size(A) == (20,40,30,50)
fulltest &= testfct(testval,"unreshape!(Array,Vector)")

rA = unreshape!(copy(tA),20,1200,50)
testval = isapprox(norm(rA.T),norm(tA.T)) && size(rA) == (20,1200,50) && size(tA) == (20,40,30,50)
fulltest &= testfct(testval,"unreshape!(tens,Vector)")

println()

order = [1,4,3,2]
B = permutedims!(copy(tA),order)
testval = isapprox(sum([size(B,i) - size(tA,order[i]) for i = 1:length(order)]),0) && isapprox(norm(tA),norm(B))
fulltest &= testfct(testval,"permutedims!(tens,Vector)")

B = permutedims!(copy(tA),(order...,))
testval = isapprox(sum([size(B,i) - size(tA,order[i]) for i = 1:length(order)]),0) && isapprox(norm(tA),norm(B))
fulltest &= testfct(testval,"permutedims!(tens,Tuple)")

C = LinearAlgebra.Diagonal(rand(10))

order = [2,1]
B = permutedims!(copy(C),order)
testval = isapprox(sum([size(B,i) - size(C,order[i]) for i = 1:length(order)]),0) && isapprox(norm(C),norm(B))
fulltest &= testfct(testval,"permutedims!(Diagonal,Vector)")

B = permutedims!(copy(C),(order...,))
testval = isapprox(sum([size(B,i) - size(C,order[i]) for i = 1:length(order)]),0) && isapprox(norm(C),norm(B))
fulltest &= testfct(testval,"permutedims!(Diagonal,Tuple)")


order = [1,4,3,2]
B = permutedims(copy(tA),order)
testval = isapprox(sum([size(B,i) - size(tA,order[i]) for i = 1:length(order)]),0) && isapprox(norm(tA),norm(B))
fulltest &= testfct(testval,"permutedims(tens,Vector)")

B = permutedims(copy(tA),(order...,))
testval = isapprox(sum([size(B,i) - size(tA,order[i]) for i = 1:length(order)]),0) && isapprox(norm(tA),norm(B))
fulltest &= testfct(testval,"permutedims(tens,Tuple)")

C = LinearAlgebra.Diagonal(rand(10))

order = [2,1]
B = permutedims(copy(C),order)
testval = isapprox(sum([size(B,i) - size(C,order[i]) for i = 1:length(order)]),0) && isapprox(norm(C),norm(B))
fulltest &= testfct(testval,"permutedims(Diagonal,Vector)")

B = permutedims(copy(C),(order...,))
testval = isapprox(sum([size(B,i) - size(C,order[i]) for i = 1:length(order)]),0) && isapprox(norm(C),norm(B))
fulltest &= testfct(testval,"permutedims(Diagonal,Tuple)")

println()

A = rand(30,20,10)
B = rand(10,20,10)

C = joinindex(A,B,1)
testval = isapprox(sum(A)+sum(B),sum(C)) && size(C) == (40,20,10)
fulltest &= testfct(testval,"joinindex(Array,integer)")

C = joinindex(A,B,[1,2])
testval = isapprox(sum(A)+sum(B),sum(C)) && size(C) == (40,40,10)
fulltest &= testfct(testval,"joinindex(Array,Array)")

tA = tens(A)
tB = tens(B)

C = joinindex(tA,tB,1)
testval = isapprox(sum(tA)+sum(tB),sum(C)) && size(C) == (40,20,10)
fulltest &= testfct(testval,"joinindex(tens,integer)")

C = joinindex(tA,tB,[1,2])
testval = isapprox(sum(tA)+sum(tB),sum(C)) && size(C) == (40,40,10)
fulltest &= testfct(testval,"joinindex(tens,Array)")
