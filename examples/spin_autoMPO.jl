#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker (2018)
# See accompanying license with this program
# This code is native to the julia programming language (v1.1.1) or (v1.5)
#


using DMRjulia


    using DelimitedFiles
    G = #readdlm("./XXXXXX.txt") #load in your file here.  Format:
#=

First column: site one
Second column: site two

[for current example]
Columns 3 and 4: couplings in a XXZ model

=#

    Ns = convert(Int64,maximum(vcat(G[:,1],G[:,2])))

    spinmag = 0.5
    Sx,Sy,Sz,Sp,Sm,O,Id = spinOps(s=spinmag)

    base = Array{Float64,2}[Id for i = 1:Ns] 

    function makeH()
        # X-terms
        xpos,ypos = convert(Int64,G[1,1]),convert(Int64,G[1,2])
        baseMPO = mpoterm(-G[1,4], [Sm,Sp], [xpos,ypos], base)
        baseMPO += mpoterm(-G[1,4], [Sp,Sm], [xpos,ypos], base)
        baseMPO += mpoterm(G[1,3], [Sz,Sz], [xpos,ypos], base)
        for a = 2:size(G,1)
          xpos,ypos = convert(Int64,G[a,1]),convert(Int64,G[a,2])
          baseMPO += mpoterm(-G[a,4], [Sm,Sp], [xpos,ypos], base)
          baseMPO += mpoterm(-G[a,4], [Sp,Sm], [xpos,ypos], base)
          baseMPO += mpoterm(G[a,3], [Sz,Sz], [xpos,ypos], base)
        end
        return baseMPO
    end
    mpo = makeH()

    hereQS = convert(Int64,2*spinmag+1)

    @makeQNs "spin" U1
    Qlabels = [[spin(i) for i = QS:-2:-QS]]
    qmpo = makeqMPO(mpo,Qlabels)
    compressMPO!(qmpo)

    
    psi = randMPS(hereQS,Ns)
    qpsi = makeqMPS(psi,Qlabels)

    dmrg(qpsi,qmpo,sweeps=1000,goal=1E-5,cutoff=1E-6,maxm=200)
    QNenergy = dmrg(qpsi,qmpo,sweeps=100,cutoff=1E-10,maxm=200,method="twosite")
