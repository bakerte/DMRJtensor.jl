
println("#            +----------------+")
println("#>-----------|  dmrg_test.jl  |-----------<")
println("#            +----------------+")
fulltest = true

algs = ["twosite","3S","2S"]

@makeQNs "testspin" U1
@makeQNs "testHubbard" U1 U1

testvals = Array{Bool,2}(undef,2,length(algs))


Cup,Cdn,F,Nup,Ndn,Ndens,O,Id = fermionOps()

for i = 1:2
  if i == 1
    global mpo = makeMPO(heisenbergMPO,2,100)
    global Qlabels = [testspin(-2),testspin(2)]
  else
    global mu = -2.0
    global HubU = 4.0
    global t = 1.0

    function H(i::Int64)
        onsite = mu * Ndens + HubU * Nup * Ndn #- Ne*exp(-abs(i-Ns/2)/2)*Ndens
        return [Id  O O O O O;
            -t*Cup' O O O O O;
            conj(t)*Cup  O O O O O;
            -t*Cdn' O O O O O;
            conj(t)*Cdn  O O O O O;
            onsite Cup*F Cup'*F Cdn*F Cdn'*F Id]
    end

    global mpo = makeMPO(H,4,10)
    global Qlabels = [testHubbard(0,0),testHubbard(1,1),testHubbard(1,-1),testHubbard(2,0)]
  end
  global Ns = length(mpo)

  for w = 1:length(algs)
    if i == 1
      print("spins"," ")
    else
      print("fermions"," ")
    end
    print(algs[w],"   ")#,"    Delta En = ")

    if i == 1
      global spinmag = 0.5

      global hereQS = convert(Int64,2*spinmag+1)
      global QS = cld(hereQS,2)
      
      global initTensor = [zeros(1,hereQS,1) for i=1:Ns]
      for i = 1:Ns
         initTensor[i][1,i%2 == 1 ? 1 : 2,1] = 1.0
      end
      
      global psi = MPS(initTensor)
    else
      global Ne = Ns
      global Ne_up = ceil(Int64,div(Ne,2))
      global Ne_dn = Ne-Ne_up
      global QS = 4

      global psi = MPS(QS,Ns)
      global upsites = [i for i = 1:Ne_up]
      global Cupdag = Matrix(Cup')
      applyOps!(psi,upsites,Cupdag,trail=F)


      global dnsites = [i for i = 1:Ne_dn]
      global Cdndag = Matrix(Cdn')
      applyOps!(psi,dnsites,Cdndag,trail=F)
    end

    global qpsi,qmpo = MPS(Qlabels,psi,mpo)

    dmrg(psi,mpo,sweeps=10,m=2,cutoff=1E-12,silent=true)

    energy = dmrg(psi,mpo,sweeps=100,m=45,cutoff=1E-12,silent=true,method=algs[w])

    energyQN = dmrg(qpsi,qmpo,sweeps=100,m=45,cutoff=1E-12,silent=true,method=algs[w])
    testvals[i,w] = abs(energy-energyQN) < 1E-4
    if testvals[i,w]
      printstyled(testvals[i,w],color=:green)
    else
      printstyled(testvals[i,w],color=:red)
    end
    println()
#    println(energy-energyQN)
  end
end

fulltest &= testfct(sum(testvals)==length(testvals),"dmrg functions (two site, 3S, 2S) [for spins and fermions]")


println()

#krylov
#optimize mps

#tdvp






#tebd


println()

Ns = 10

mu = -2.0
HubU = 4.0
t = 1.0

function H(i::Int64)
        onsite = mu * Ndens + HubU * Nup * Ndn #- Ne*exp(-abs(i-Ns/2)/2)*Ndens
        return [Id  O O O O O;
            -t*Cup' O O O O O;
            conj(t)*Cup  O O O O O;
            -t*Cdn' O O O O O;
            conj(t)*Cdn  O O O O O;
            onsite Cup*F Cup'*F Cdn*F Cdn'*F Id]
    end
mpo = makeMPO(H,4,Ns)

Ne = Ns
Ne_up = ceil(Int64,div(Ne,2))
Ne_dn = Ne-Ne_up
QS = 4
Cup,Cdn,F,Nup,Ndn,Ndens,O,Id = fermionOps()

psi = MPS(QS,Ns)
upsites = [i for i = 1:Ne_up]
Cupdag = Matrix(Cup')
applyOps!(psi,upsites,Cupdag,trail=F)


dnsites = [i for i = 1:Ne_dn]
Cdndag = Matrix(Cdn')
applyOps!(psi,dnsites,Cdndag,trail=F)

Qlabels = [testHubbard(0,0),testHubbard(1,1),testHubbard(1,-1),testHubbard(2,0)]
qpsi,qmpo = MPS(Qlabels,psi,mpo)

energy = dmrg(psi,mpo,sweeps=20,m=45,cutoff=1E-9,silent=true)
QNenergy = dmrg(qpsi,qmpo,sweeps=20,m=45,cutoff=1E-9,silent=true)



psi0 = deepcopy(psi)
qpsi0 = deepcopy(qpsi)

ntime = 500
deltaT = 0.001

function tMPO(i::Int64)
onsite(i::Int64) = (mu * Ndens + HubU * Nup * Ndn)*(i!=1 && i!= Ns ? 0.5 : 1)
    return [Id  O O O O O;
            Cup' O O O O O;
            Cup  O O O O O;
            Cdn' O O O O O;
            Cdn  O O O O O;
            onsite(i) -t*Cup*F conj(t)*Cup'*F -t*Cdn*F conj(t)*Cdn'*F Id]
end

tH = makeMPO(tMPO,4,Ns)

expgates = makeExpGates(tH,-im*deltaT)
phiT = Array{ComplexF64,1}(undef,ntime)
for j = 1:ntime
  global psi = tebd(psi,expgates,cutoff = 1E-8,m = 50)
  phiT[j] = expect(psi0,psi)
end
expT = phiT[ntime]




times = [n*deltaT for n = 1:ntime]

checktimes = [n*deltaT for n = 1:ntime]
checkexp = [exp(-im*n*deltaT*energy) for n = 1:ntime]

tqMPO = makeqMPO(Qlabels,tH)

qexpgates = qmakeExpGates(tH,Qlabels,-im*deltaT)
qphiT = Array{ComplexF64,1}(undef,ntime)
for j = 1:ntime
  global qpsi = tebd(qpsi,qexpgates,cutoff = 1E-8,m = 50)
  qphiT[j] = expect(qpsi0,qpsi)
end
qexpT = qphiT[ntime]

finalE = expect(psi,psi,mpo)
QNfinalE = expect(qpsi,qpsi,qmpo)


testval = abs(QNfinalE-finalE) < 1E-4
fulltest &= testfct(testval,"tebd (dense & symmetry)")

println()
println()
