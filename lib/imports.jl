###############################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                               v1.0
#
###############################################################################
# Made by Thomas E. Baker and « les qubits volants » (2024)
# See accompanying license with this program
# This code is native to the julia programming language (v1.10.0+)
#


import LinearAlgebra
import Printf
import Serialization

import Distributed




import Base.display


import Base.println




import .Base: getindex,setindex!,copy,size,conj,conj!,eltype,lastindex,sum,permutedims!,+,*,-,/,length,adjoint

import .LinearAlgebra.norm



#using TensorPACK
using ..TENPACK


#import .TENPACK.TensType

import ..TENPACK: undefMat,checkType,tensorcombination,tensorcombination!,permutedims,invmat!,reshape!,unreshape!,unreshape,get_denseranges,sub!,add!,mult!,div!,joinindex!,getindex!,searchindex,convertTens

import ..TENPACK: libmult,libsvd,libsvd!,libeigen,libeigen!,libqr,liblq,libqr!,liblq!

#contracions
import ..TENPACK: contract,ccontract,contractc,ccontractc

import ..TENPACK: default_boundary

import ..TENPACK: dot,dmul!,diagcontract!


import ..TENPACK: permq,willperm,prepareT,getsizes,maincontractor,trace



import ..TENPACK: sqrt!,defzero,basedims,makeHpsi,root


import ..TENPACK: truncate,svd,svd!,getorder,findsize,svdvals,eigen,eigen!,eigvals,eigvals!,qr,qr!,lq,lq!,polar,eye


import ..TENPACK: network,TNnetwork


import ..TENPACK: largevector, tensor2disc, tensorfromdisc


import ..TENPACK: stdavg,testfct


import ..TENPACK: Qtens



#decompositions
#import ..TENPACK.libsvd,libsvd!


#import ..TENPACK.libeigen
#import ..TENPACK.libeigen!

#=
import TensorPACK.undefMat,TensorPACK.makeArray,TensorPACK.checkType,TensorPACK.tensorcombination,TensorPACK.tensorcombination!
import TensorPACK.makeId

import TensorPACK.reshape!,TensorPACK.convertTens

import TensorPACK.makedens
import TensorPACK.get_denseranges

import TensorPACK.sub!,TensorPACK.add!,TensorPACK.mult!,TensorPACK.div!
import TensorPACK.joinindex!
import TensorPACK.getindex!
import TensorPACK.searchindex

import TensorPACK.libmult,TensorPACK.libsvd,TensorPACK.libsvd!,TensorPACK.libeigen,TensorPACK.libeigen!,TensorPACK.libqr,TensorPACK.liblq,TensorPACK.libqr!,TensorPACK.liblq!
=#

#import .TENPACK.tens
#export tens





#import ..TENPACK.sin
#import ..TENPACK.cos
