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



using TensorPACK
#using TensorPACK




#import .TensorPACK.TensType

import TensorPACK: undefMat,checkType,tensorcombination,tensorcombination!,permutedims,invmat!,reshape!,unreshape!,unreshape,get_denseranges,sub!,add!,mult!,div!,joinindex!,getindex!,searchindex,convertTens

import TensorPACK: libmult,libsvd,libsvd!,libeigen,libeigen!,libqr,liblq,libqr!,liblq!

#contracions
import TensorPACK: contract,ccontract,contractc,ccontractc

import TensorPACK: default_boundary

import TensorPACK: dot,dmul!,diagcontract!


import TensorPACK: permq,willperm,prepareT,getsizes,maincontractor,trace



import TensorPACK: sqrt!,defzero,basedims,makeHpsi,root


import TensorPACK: truncate,svd,svd!,getorder,findsize,svdvals,eigen,eigen!,eigvals,eigvals!,qr,qr!,lq,lq!,polar,eye


import TensorPACK: network,TNnetwork


import TensorPACK: largevector, tensor2disc, tensorfromdisc


import TensorPACK: stdavg,testfct


import TensorPACK: Qtens



#decompositions
#import TensorPACK.libsvd,libsvd!


#import TensorPACK.libeigen
#import TensorPACK.libeigen!

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

#import .TensorPACK.tens
#export tens





#import TensorPACK.sin
#import TensorPACK.cos
