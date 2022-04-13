#########################################################################
#
#  Density Matrix Renormalization Group (and other methods) in julia (DMRjulia)
#                              v0.8
#
#########################################################################
# Made by Thomas E. Baker (2020)
# See accompanying license with this program
# This code is native to the julia programming language (v1.5.4+)
#


import LinearAlgebra.BLAS: BlasReal, BlasComplex, BlasFloat, BlasInt, DimensionMismatch, checksquare, axpy!, @blasfunc, libblastrampoline


for elty in (:Float32,:Float64,:ComplexF32,:ComplexF64)
  @eval begin

    #tridiagonal
    function libeigen!(dv::AbstractVector{$elty}, ev::AbstractVector{$elty},n::Integer; job::Char='V', rank::Integer=2,fct::Function=stev!)
      return fct(dv, ev,n, job=job, rank=rank)
    end

    function libeigen!(dv::AbstractVector{$elty}, ev::AbstractVector{$elty};job::Char='V', rank::Integer=2) #stev!
      return libeigen!(dv,ev,length(dv),job=job,rank=rank)
    end

    function libeigen(dv::AbstractVector{$elty}, ev::AbstractVector{$elty},n::Integer; job::Char='V', rank::Integer=2) #stev!
      return libeigen!(copy(dv),copy(ev),n,job=job,rank=rank)
    end

    function libeigen(dv::AbstractVector{$elty}, ev::AbstractVector{$elty};job::Char='V', rank::Integer=2) #stev!
      return libeigen!(copy(dv),copy(ev),length(dv),job=job,rank=rank)
    end

    #symmetric/Hermitian
    function libeigen!(A::Union{AbstractMatrix{$elty},Vector{$elty}},n::Integer;job::Char='V',uplo::Char='U',fct::Function=syev!)
      return fct(A,n,job=job,uplo=uplo)
    end

    function libeigen!(A::AbstractMatrix{$elty};job::Char='V',uplo::Char='U') #syev!
      return libeigen!(A,size(A,1),job=job,uplo=uplo)
    end

    function libeigen!(A::tens{$elty};job::Char='V',uplo::Char='U') #syev!
      return libeigen!(A.T,size(A,1),job=job,uplo=uplo)
    end

    function libeigen(A::AbstractMatrix{$elty};job::Char='V',uplo::Char='U') #syev!
      return libeigen!(copy(A),size(A,1),job=job,uplo=uplo)
    end

    function libeigen(A::tens{$elty};job::Char='V',uplo::Char='U') #syev!
      return libeigen!(copy(A.T),size(A,1),job=job,uplo=uplo)
    end

    #generalized
    function libeigen!(A::AbstractMatrix{$elty}, n::Integer, B::AbstractMatrix{$elty};jobvl::AbstractChar='V', jobvr::AbstractChar='V',fct::Function=ggev!)
      return fct(A,n,B,jobvl=jobvl,jobvr=jobvr)
    end


    function libsvd!(A::Union{AbstractMatrix{$elty},Vector{$elty}},m::Integer,n::Integer;job::Char='O',fct::Function=gesdd!)
      return fct(A,m,n,job=job)
    end

    function libsvd!(A::AbstractMatrix{$elty};job::Char='O')
        return libsvd!(A,size(A,1),size(A,2),job=job)
    end

    function libsvd!(A::tens{$elty};job::Char='O')
        return libsvd!(A.T,size(A,1),size(A,2),job=job)
    end

    function libsvd(A::Union{AbstractMatrix{$elty},Vector{$elty}},m::Integer,n::Integer;job::Char='S') #LinearAlgebra.LAPACK.gesdd!
        return libsvd!(copy(A),m,n,job=job)
    end

    function libsvd(A::AbstractMatrix{$elty};job::Char='S')
        return libsvd!(copy(A),size(A,1),size(A,2),job=job)
    end

    function libsvd(A::tens{$elty};job::Char='S')
        return libsvd!(copy(A.T),size(A,1),size(A,2),job=job)
    end
      
    #generalized
    function libsvd!(A::AbstractMatrix{$elty}, m::Integer, n::Integer, B::AbstractMatrix{$elty};jobu::AbstractChar='S', jobv::AbstractChar='S', jobq::AbstractChar='Q',fct::Function=ggsvd!)
      return fct(A, m, n, B;jobu=jobu, jobv=jobv, jobq=jobq)
    end
  end
end

#         +---------------+
#>--------|     mult!     |---------<
#         +---------------+

for (gemm, elty) in
  ((:dgemm_,:Float64),
   (:sgemm_,:Float32),
   (:zgemm_,:ComplexF64),
   (:cgemm_,:ComplexF32))
  @eval begin
        # SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
        # *     .. Scalar Arguments ..
        #       DOUBLE PRECISION ALPHA,BETA
        #       INTEGER K,LDA,LDB,LDC,M,N
        #       CHARACTER TRANSA,TRANSB
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
    function libmult!(transA::AbstractChar, transB::AbstractChar,
                  alpha::Union{($elty), Bool},
                  A::AbstractArray{$elty,N},
                  B::AbstractArray{$elty,M},
                  beta::Union{($elty), Bool},
                  C::AbstractArray{$elty,G},
                  m::Integer,ka::Integer,kb::Integer,n::Integer) where {N,M,G}
        lda = max(1,transA == 'N' ? m : ka)
        ldb = max(1,transB == 'N' ? ka : n)
        ldc = max(1,m)
        ccall((@blasfunc($gemm), libblastrampoline), Cvoid,
            (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
            Ref{BlasInt}, Ref{$elty}, Ptr{$elty}, Ref{BlasInt},
            Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ptr{$elty},
            Ref{BlasInt}, Clong, Clong),
            transA, transB, m, n,
            ka, alpha, A,  lda,
            B, ldb, beta, C,
            ldc, 1, 1)
        C
    end
    function libmult(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::AbstractArray{$elty,N},B::AbstractArray{$elty,M},m::Integer,ka::Integer,kb::Integer,n::Integer) where {N,M}
        C = Array{($elty),2}(undef,m,n)
        libmult!(transA, transB, alpha, A, B, zero($elty), C,m,ka,kb,n)
        return C
    end
    function libmult(transA::AbstractChar, transB::AbstractChar, A::Union{AbstractArray{$elty,N},tens{$elty}},B::Union{AbstractArray{$elty,M},tens{$elty}},m::Integer,ka::Integer,kb::Integer,n::Integer) where {N,M}
        libmult(transA, transB, one($elty), A, B,m,ka,kb,n)
    end

    function libmult(transA::AbstractChar, transB::AbstractChar, alpha::($elty), A::tens{$elty},B::tens{$elty},m::Integer,ka::Integer,kb::Integer,n::Integer) where {N,M}
      C = Array{($elty),1}(undef,m*n)
      libmult!(transA, transB, alpha, A.T, B.T, zero($elty), C,m,ka,kb,n)
      return C
    end
  end
end

#         +-----------------------------+
#>--------|     Eigen (tridiagonal)     |---------<
#         +-----------------------------+



## (ST) Symmetric tridiagonal - eigendecomposition
for (stev, stebz, stegr, stein, elty) in
  ((:dstev_,:dstebz_,:dstegr_,:dstein_,:Float64),
   (:sstev_,:sstebz_,:sstegr_,:sstein_,:Float32)
   )
  @eval begin

      function stev!(dv::AbstractVector{$elty}, ev::AbstractVector{$elty},n::Integer; job::Char='V', rank::Integer=2) #stev!
          Zmat = rank == 1 ? Array{$elty,1}(undef,job != 'N' ? n*n : 0) : Array{$elty,2}(undef,n,job != 'N' ? n : 0)
          work = Vector{$elty}(undef, max(1, 2n-2))
          info = Ref{BlasInt}()
          ccall((@blasfunc($stev), libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                 Ref{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Clong),
                job, n, dv, ev, Zmat, n, work, info, 1)
#          chklapackerror(info[])
          dv, Zmat
      end


      #*  DSTEBZ computes the eigenvalues of a symmetric tridiagonal
      #*  matrix T.  The user may ask for all eigenvalues, all eigenvalues
      #*  in the half-open interval (VL, VU], or the IL-th through IU-th
      #*  eigenvalues.
      function stebz!(range::AbstractChar, order::AbstractChar, vl::$elty, vu::$elty, il::Integer, iu::Integer, abstol::Real, dv::AbstractVector{$elty}, ev::AbstractVector{$elty})
          n = length(dv)
          if length(ev) != n - 1
              throw(DimensionMismatch("ev has length $(length(ev)) but needs one less than dv's length, $n)"))
          end
          m = Ref{BlasInt}()
          nsplit = Vector{BlasInt}(undef, 1)
          w = similar(dv, $elty, n)
          tmp = 0.0
          iblock = similar(dv, BlasInt,n)
          isplit = similar(dv, BlasInt,n)
          work = Vector{$elty}(undef, 4*n)
          iwork = Vector{BlasInt}(undef, 3*n)
          info = Ref{BlasInt}()
          ccall((@blasfunc($stebz), libblastrampoline), Cvoid,
              (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{$elty},
              Ref{$elty}, Ref{BlasInt}, Ref{BlasInt}, Ref{$elty},
              Ptr{$elty}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt},
              Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
              Ptr{BlasInt}, Ptr{BlasInt}, Clong, Clong),
              range, order, n, vl,
              vu, il, iu, abstol,
              dv, ev, m, nsplit,
              w, iblock, isplit, work,
              iwork, info, 1, 1)
#          chklapackerror(info[])
          w[1:m[]], iblock[1:m[]], isplit[1:nsplit[1]]
      end

      function stegr!(jobz::AbstractChar, range::AbstractChar, dv::AbstractVector{$elty}, ev::AbstractVector{$elty}, vl::Real, vu::Real, il::Integer, iu::Integer)
          n = length(dv)
          ne = length(ev)
          if ne == n - 1
              eev = [ev; zero($elty)]
          elseif ne == n
              eev = copy(ev)
              eev[n] = zero($elty)
          else
              throw(DimensionMismatch("ev has length $ne but needs one less than or equal to dv's length, $n)"))
          end

          abstol = Vector{$elty}(undef, 1)
          m = Ref{BlasInt}()
          w = similar(dv, $elty, n)
          ldz = jobz == 'N' ? 1 : n
          Z = similar(dv, $elty, ldz, range == 'I' ? iu-il+1 : n)
          isuppz = similar(dv, BlasInt, 2*size(Z, 2))
          work = Vector{$elty}(undef, 1)
          lwork = BlasInt(-1)
          iwork = Vector{BlasInt}(undef, 1)
          liwork = BlasInt(-1)
          info = Ref{BlasInt}()
          for i = 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
              ccall((@blasfunc($stegr), libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                  Ptr{$elty}, Ref{$elty}, Ref{$elty}, Ref{BlasInt},
                  Ref{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{$elty},
                  Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                  Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                  Clong, Clong),
                  jobz, range, n, dv,
                  eev, vl, vu, il,
                  iu, abstol, m, w,
                  Z, ldz, isuppz, work,
                  lwork, iwork, liwork, info,
                  1, 1)
#              chklapackerror(info[])
              if i == 1
                  lwork = BlasInt(work[1])
                  resize!(work, lwork)
                  liwork = iwork[1]
                  resize!(iwork, liwork)
              end
          end
          m[] == length(w) ? w : w[1:m[]], m[] == size(Z, 2) ? Z : Z[:,1:m[]]
      end

      function stein!(dv::AbstractVector{$elty}, ev_in::AbstractVector{$elty}, w_in::AbstractVector{$elty}, iblock_in::AbstractVector{BlasInt}, isplit_in::AbstractVector{BlasInt})
          require_one_based_indexing(dv, ev_in, w_in, iblock_in, isplit_in)
          chkstride1(dv, ev_in, w_in, iblock_in, isplit_in)
          n = length(dv)
          ne = length(ev_in)
          if ne == n - 1
              ev = [ev_in; zero($elty)]
          elseif ne == n
              ev = copy(ev_in)
              ev[n] = zero($elty)
          else
              throw(DimensionMismatch("ev_in has length $ne but needs one less than or equal to dv's length, $n)"))
          end
          ldz = n #Leading dimension
          #Number of eigenvalues to find
          if !(1 <= length(w_in) <= n)
              throw(DimensionMismatch("w_in has length $(length(w_in)), but needs to be between 1 and $n"))
          end
          m = length(w_in)
          #If iblock and isplit are invalid input, assume worst-case block partitioning,
          # i.e. set the block scheme to be the entire matrix
          iblock = similar(dv, BlasInt,n)
          isplit = similar(dv, BlasInt,n)
          w = similar(dv, $elty,n)
          if length(iblock_in) < m #Not enough block specifications
              iblock[1:m] = fill(BlasInt(1), m)
              w[1:m] = sort(w_in)
          else
              iblock[1:m] = iblock_in
              w[1:m] = w_in #Assume user has sorted the eigenvalues properly
          end
          if length(isplit_in) < 1 #Not enough block specifications
              isplit[1] = n
          else
              isplit[1:length(isplit_in)] = isplit_in
          end
          z = similar(dv, $elty,(n,m))
          work  = Vector{$elty}(undef, 5*n)
          iwork = Vector{BlasInt}(undef, n)
          ifail = Vector{BlasInt}(undef, m)
          info  = Ref{BlasInt}()
          ccall((@blasfunc($stein), libblastrampoline), Cvoid,
              (Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
              Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
              Ref{BlasInt}, Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt},
              Ptr{BlasInt}),
              n, dv, ev, m, w, iblock, isplit, z, ldz, work, iwork, ifail, info)
          chklapackerror(info[])
          if any(ifail .!= 0)
              # TODO: better error message / type
              error("failed to converge eigenvectors:\n$(findall(!iszero, ifail))")
          end
          z
      end
  end
end

#         +---------------+
#>--------|     Eigen     |---------<
#         +---------------+

for (syev, syevr, sygvd, elty, relty) in
  ((:dsyev_,:dsyevr_,:dsygvd_,:Float64,:Float64),
   (:ssyev_,:ssyevr_,:ssygvd_,:Float32,:Float32),
   (:zheev_,:zheevr_,:zhegvd_,:ComplexF64,:Float64),
   (:cheev_,:cheevr_,:chegvd_,:ComplexF32,:Float32))
  @eval begin

      #       SUBROUTINE DSYEV( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, INFO )
      # *     .. Scalar Arguments ..
      #       CHARACTER          JOBZ, UPLO
      #       INTEGER            INFO, LDA, LWORK, N
      # *     .. Array Arguments ..
      #       DOUBLE PRECISION   A( LDA, * ), W( * ), WORK( * )
      function syev!(A::Union{AbstractMatrix{$elty},Vector{$elty}},n::Integer;job::Char='V',uplo::Char='U') #syev!
#          chkstride1(A)
#          n = checksquare(A)
          W     = similar(A, $elty, n)
          work  = Vector{$elty}(undef, 1)
          lwork = BlasInt(-1)
          cmplx = eltype(A) <: Complex
          if cmplx
            rwork = Vector{$relty}(undef, max(1, 3n-2))
          end
          info  = Ref{BlasInt}()
          for i = 1:2  # first call returns lwork as work[1]
              if cmplx
                  ccall((@blasfunc($syev), libblastrampoline), Cvoid,
                        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt},
                        Clong, Clong),
                        job, uplo, n, A, max(1,n), W, work, lwork, rwork, info, 1, 1)
              else
                  ccall((@blasfunc($syev), libblastrampoline), Cvoid,
                        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
                        job, uplo, n, A, max(1,n), W, work, lwork, info, 1, 1)
              end
#              chklapackerror(info[])
              if i == 1
                  lwork = BlasInt(real(work[1]))
                  resize!(work, lwork)
              end
          end
          job == 'V' ? (W, A) : W
      end

      #       SUBROUTINE DSYEVR( JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU,
      #      $                   ABSTOL, M, W, Z, LDZ, ISUPPZ, WORK, LWORK,
      #      $                   IWORK, LIWORK, INFO )
      # *     .. Scalar Arguments ..
      #       CHARACTER          JOBZ, RANGE, UPLO
      #       INTEGER            IL, INFO, IU, LDA, LDZ, LIWORK, LWORK, M, N
      #       DOUBLE PRECISION   ABSTOL, VL, VU
      # *     ..
      # *     .. Array Arguments ..
      #       INTEGER            ISUPPZ( * ), IWORK( * )
      #       DOUBLE PRECISION   A( LDA, * ), W( * ), WORK( * ), Z( LDZ, * )
      function syevr!(jobz::AbstractChar, range::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty},
                      vl::AbstractFloat, vu::AbstractFloat, il::Integer, iu::Integer, abstol::AbstractFloat)                
          n = checksquare(A)
          lda = stride(A,2)
          m = Ref{BlasInt}()
          w = similar(A, $elty, n)
          ldz = n
          if jobz == 'N'
              Z = similar(A, $elty, ldz, 0)
          elseif jobz == 'V'
              Z = similar(A, $elty, ldz, n)
          end
          isuppz = similar(A, BlasInt, 2*n)
          work   = Vector{$elty}(undef, 1)
          lwork  = BlasInt(-1)

          cmplx = eltype(A) <: Complex
          if cmplx
            rwork  = Vector{$relty}(undef, 1)
            lrwork = BlasInt(-1)
          end

          iwork  = Vector{BlasInt}(undef, 1)
          liwork = BlasInt(-1)
          info   = Ref{BlasInt}()
          for i = 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
            if cmplx
                ccall((@blasfunc($syevr), libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                 Ref{BlasInt}, Ref{BlasInt}, Ref{$elty}, Ptr{BlasInt},
                 Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                 Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                 Clong, Clong, Clong),
                jobz, range, uplo, n,
                A, lda, vl, vu,
                il, iu, abstol, m,
                w, Z, ldz, isuppz,
                work, lwork, rwork, lrwork,
                iwork, liwork, info,
                1, 1, 1)
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                    lrwork = BlasInt(rwork[1])
                    resize!(rwork, lrwork)
                    liwork = iwork[1]
                    resize!(iwork, liwork)
                end
            else
              ccall((@blasfunc($syevr), libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                      Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                      Ref{BlasInt}, Ref{BlasInt}, Ref{$elty}, Ptr{BlasInt},
                      Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                      Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                      Ptr{BlasInt}, Clong, Clong, Clong),
                  jobz, range, uplo, n,
                  A, max(1,lda), vl, vu,
                  il, iu, abstol, m,
                  w, Z, max(1,ldz), isuppz,
                  work, lwork, iwork, liwork,
                  info, 1, 1, 1)
                  if i == 1
                      lwork = BlasInt(real(work[1]))
                      resize!(work, lwork)
                      liwork = iwork[1]
                      resize!(iwork, liwork)
                  end
            end
#              chklapackerror(info[])
          end
          w[1:m[]], Z[:,1:(jobz == 'V' ? m[] : 0)]
      end
      syevr!(jobz::AbstractChar, A::AbstractMatrix{$elty}) =
          syevr!(jobz, 'A', 'U', A, 0.0, 0.0, 0, 0, -1.0)

      # Generalized eigenproblem
      #           SUBROUTINE DSYGVD( ITYPE, JOBZ, UPLO, N, A, LDA, B, LDB, W, WORK,
      #      $                   LWORK, IWORK, LIWORK, INFO )
      # *     .. Scalar Arguments ..
      #       CHARACTER          JOBZ, UPLO
      #       INTEGER            INFO, ITYPE, LDA, LDB, LIWORK, LWORK, N
      # *     ..
      # *     .. Array Arguments ..
      #       INTEGER            IWORK( * )
      #       DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), W( * ), WORK( * )
      function sygvd!(itype::Integer, jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
          n, m = checksquare(A, B)
          lda = max(1, stride(A, 2))
          ldb = max(1, stride(B, 2))
          w = similar(A, $elty, n)
          work = Vector{$elty}(undef, 1)
          lwork = BlasInt(-1)
          iwork = Vector{BlasInt}(undef, 1)
          liwork = BlasInt(-1)
          cmplx = eltype(A) <: Complex
          if cmplx
            rwork = Vector{$relty}(undef, 1)
            lrwork = BlasInt(-1)
          end
          info = Ref{BlasInt}()
          for i = 1:2  # first call returns lwork as work[1] and liwork as iwork[1]
            if cmplx
                ccall((@blasfunc($sygvd), libblastrampoline), Cvoid,
                (Ref{BlasInt}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                 Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                 Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                 Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                 Clong, Clong),
                itype, jobz, uplo, n,
                A, lda, B, ldb,
                w, work, lwork, rwork,
                lrwork, iwork, liwork, info,
                1, 1)
                chkargsok(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                    liwork = iwork[1]
                    resize!(iwork, liwork)
                    lrwork = BlasInt(rwork[1])
                    resize!(rwork, lrwork)
                end
            else
              ccall((@blasfunc($sygvd), libblastrampoline), Cvoid,
                  (Ref{BlasInt}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                   Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
                  itype, jobz, uplo, n,
                  A, lda, B, ldb,
                  w, work, lwork, iwork,
                  liwork, info, 1, 1)
#              chkargsok(info[])
              if i == 1
                  lwork = BlasInt(work[1])
                  resize!(work, lwork)
                  liwork = iwork[1]
                  resize!(iwork, liwork)
              end
            end
          end
#          chkposdef(info[])
          w, A, B
      end
  end
end

for (geev, elty, relty) in
  ((:dgeev_,:Float64,:Float64),
   (:sgeev_,:Float32,:Float32),
   (:zgeev_,:ComplexF64,:Float64),
   (:cgeev_,:ComplexF32,:Float32))
  @eval begin
      #      SUBROUTINE DGEEV( JOBVL, JOBVR, N, A, LDA, WR, WI, VL, LDVL, VR,
      #      $                  LDVR, WORK, LWORK, INFO )
      # *     .. Scalar Arguments ..
      #       CHARACTER          JOBVL, JOBVR
      #       INTEGER            INFO, LDA, LDVL, LDVR, LWORK, N
      # *     .. Array Arguments ..
      #       DOUBLE PRECISION   A( LDA, * ), VL( LDVL, * ), VR( LDVR, * ),
      #      $                   WI( * ), WORK( * ), WR( * )
      function geev!(jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix{$elty})
          chkstride1(A)
          n = checksquare(A)
          chkfinite(A) # balancing routines don't support NaNs and Infs
          lvecs = jobvl == 'V'
          rvecs = jobvr == 'V'
          VL    = similar(A, $elty, (n, lvecs ? n : 0))
          VR    = similar(A, $elty, (n, rvecs ? n : 0))
          cmplx = eltype(A) <: Complex
          if cmplx
              W     = similar(A, $elty, n)
              rwork = similar(A, $relty, 2n)
          else
              WR    = similar(A, $elty, n)
              WI    = similar(A, $elty, n)
          end
          work  = Vector{$elty}(undef, 1)
          lwork = BlasInt(-1)
          info  = Ref{BlasInt}()
          for i = 1:2  # first call returns lwork as work[1]
              if cmplx
                  ccall((@blasfunc($geev), libblastrampoline), Cvoid,
                        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                         Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                         Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                         Ptr{$relty}, Ptr{BlasInt}, Clong, Clong),
                        jobvl, jobvr, n, A, max(1,stride(A,2)), W, VL, n, VR, n,
                        work, lwork, rwork, info, 1, 1)
              else
                  ccall((@blasfunc($geev), libblastrampoline), Cvoid,
                        (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                         Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                         Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                         Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
                        jobvl, jobvr, n, A, max(1,stride(A,2)), WR, WI, VL, n,
                        VR, n, work, lwork, info, 1, 1)
              end
              chklapackerror(info[])
              if i == 1
                  lwork = BlasInt(real(work[1]))
                  resize!(work, lwork)
              end
          end
          cmplx ? (W, VL, VR) : (WR, WI, VL, VR)
      end
  end
end


for (geevx, ggev, elty, relty) in
  ((:dgeevx_,:dggev_,:Float64,:Float64),
   (:sgeevx_,:sggev_,:Float32,:Float32),
   (:zgeevx_,:zggev_,:ComplexF64,:Float64),
   (:cgeevx_,:cggev_,:ComplexF32,:Float32))
  @eval begin
      #     SUBROUTINE ZGEEVX( BALANC, JOBVL, JOBVR, SENSE, N, A, LDA, W, VL,
      #                          LDVL, VR, LDVR, ILO, IHI, SCALE, ABNRM, RCONDE,
      #                          RCONDV, WORK, LWORK, RWORK, INFO )
      #
      #       .. Scalar Arguments ..
      #       CHARACTER          BALANC, JOBVL, JOBVR, SENSE
      #       INTEGER            IHI, ILO, INFO, LDA, LDVL, LDVR, LWORK, N
      #       DOUBLE PRECISION   ABNRM
      #       ..
      #       .. Array Arguments ..
      #       DOUBLE PRECISION   RCONDE( * ), RCONDV( * ), RWORK( * ),
      #      $                   SCALE( * )
      #       COMPLEX*16         A( LDA, * ), VL( LDVL, * ), VR( LDVR, * ),
      #      $                   W( * ), WORK( * )
      function geevx!(balanc::AbstractChar, jobvl::AbstractChar, jobvr::AbstractChar, sense::AbstractChar, A::AbstractMatrix{$elty}) #geevx!
          n = checksquare(A)
          chkfinite(A) # balancing routines don't support NaNs and Infs
          lda = max(1,stride(A,2))
          w = similar(A, $elty, n)
#            if balanc ∉ ['N', 'P', 'S', 'B']
#                throw(ArgumentError("balanc must be 'N', 'P', 'S', or 'B', but $balanc was passed"))
#            end
          ldvl = 0
          if jobvl == 'V'
              ldvl = n
          elseif jobvl == 'N'
              ldvl = 0
#            else
#                throw(ArgumentError("jobvl must be 'V' or 'N', but $jobvl was passed"))
          end
          VL = similar(A, $elty, ldvl, n)
          ldvr = 0
          if jobvr == 'V'
              ldvr = n
          elseif jobvr == 'N'
              ldvr = 0
#            else
#                throw(ArgumentError("jobvr must be 'V' or 'N', but $jobvr was passed"))
          end
#            if sense ∉ ['N','E','V','B']
#                throw(ArgumentError("sense must be 'N', 'E', 'V' or 'B', but $sense was passed"))
#            end
          VR = similar(A, $elty, ldvr, n)
          ilo = Ref{BlasInt}()
          ihi = Ref{BlasInt}()
          scale = similar(A, $relty, n)
          abnrm = Ref{$relty}()
          rconde = similar(A, $relty, n)
          rcondv = similar(A, $relty, n)
          work = Vector{$elty}(undef, 1)
          lwork = BlasInt(-1)
          cmplx = eltype(A) <: Complex
          if cmplx
            rwork = Vector{$relty}(undef, 2n)
          else
            iworksize = 0
            if sense == 'N' || sense == 'E'
                iworksize = 0
            elseif sense == 'V' || sense == 'B'
                iworksize = 2*n - 2
#            else
#                throw(ArgumentError("sense must be 'N', 'E', 'V' or 'B', but $sense was passed"))
            end
            iwork = Vector{BlasInt}(undef, iworksize)
            wr = w
            wi = similar(A, $elty, n)
          end
          info = Ref{BlasInt}()
          for i = 1:2  # first call returns lwork as work[1]
            if cmplx
              ccall((@blasfunc($geevx), libblastrampoline), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                     Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                     Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                     Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$relty}, Ptr{$relty},
                     Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                     Ptr{$relty}, Ptr{BlasInt}, Clong, Clong, Clong, Clong),
                     balanc, jobvl, jobvr, sense,
                     n, A, lda, w,
                     VL, max(1,ldvl), VR, max(1,ldvr),
                     ilo, ihi, scale, abnrm,
                     rconde, rcondv, work, lwork,
                     rwork, info, 1, 1, 1, 1)
            else
              ccall((@blasfunc($geevx), libblastrampoline), Cvoid,
              (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
               Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
               Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
               Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
               Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
               Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
               Clong, Clong, Clong, Clong),
               balanc, jobvl, jobvr, sense,
               n, A, lda, wr,
               wi, VL, max(1,ldvl), VR,
               max(1,ldvr), ilo, ihi, scale,
               abnrm, rconde, rcondv, work,
               lwork, iwork, info,
               1, 1, 1, 1)
            end
#                chklapackerror(info[])
              if i == 1
                  lwork = BlasInt(work[1])
                  resize!(work, lwork)
              end
          end
          if cmplx
            A, w, VL, VR, ilo[], ihi[], scale, abnrm[], rconde, rcondv
          else
            A, wr, wi, VL, VR, ilo[], ihi[], scale, abnrm[], rconde, rcondv
          end
      end


      # SUBROUTINE ZGGEV( JOBVL, JOBVR, N, A, LDA, B, LDB, ALPHA, BETA,
      #      $                  VL, LDVL, VR, LDVR, WORK, LWORK, RWORK, INFO )
      # *     .. Scalar Arguments ..
      #       CHARACTER          JOBVL, JOBVR
      #       INTEGER            INFO, LDA, LDB, LDVL, LDVR, LWORK, N
      # *     ..
      # *     .. Array Arguments ..
      #       DOUBLE PRECISION   RWORK( * )
      #       COMPLEX*16         A( LDA, * ), ALPHA( * ), B( LDB, * ),
      #      $                   BETA( * ), VL( LDVL, * ), VR( LDVR, * ),
      #      $                   WORK( * )
      function ggev!(A::AbstractMatrix{$elty}, n::Integer, B::AbstractMatrix{$elty};jobvl::AbstractChar='V', jobvr::AbstractChar='V')
#            require_one_based_indexing(A, B)
#            chkstride1(A, B)
#            n, m = checksquare(A, B)
#            if n != m
#                throw(DimensionMismatch("A has dimensions $(size(A)), and B has dimensions $(size(B)), but A and B must have the same size"))
#            end
          lda = max(1, n)#stride(A, 2))
          ldb = lda #max(1, stride(B, 2))
          alpha = similar(A, $elty, n)
          beta = similar(A, $elty, n)
          ldvl = 0
          if jobvl == 'V'
              ldvl = n
          elseif jobvl == 'N'
              ldvl = 1
#            else
#                throw(ArgumentError("jobvl must be 'V' or 'N', but $jobvl was passed"))
          end
          vl = similar(A, $elty, ldvl, n)
          ldvr = 0
          if jobvr == 'V'
              ldvr = n
          elseif jobvr == 'N'
              ldvr = 1
#            else
#                throw(ArgumentError("jobvr must be 'V' or 'N', but $jobvr was passed"))
          end
          vr = similar(A, $elty, ldvr, n)
          work = Vector{$elty}(undef, 1)
          lwork = BlasInt(-1)
          if eltype(A) <: Complex
            rwork = Vector{$relty}(undef, 8n)
          else
            alphar = alpha
            alphai = similar(A, $elty, n)
          end
          info = Ref{BlasInt}()
          for i = 1:2  # first call returns lwork as work[1]
            if cmplx
              ccall((@blasfunc($ggev), libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                   Ptr{BlasInt}, Clong, Clong),
                  jobvl, jobvr, n, A,
                  lda, B, ldb, alpha,
                  beta, vl, ldvl, vr,
                  ldvr, work, lwork, rwork,
                  info, 1, 1)
            else
              ccall((@blasfunc($ggev), libblastrampoline), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}, Clong, Clong),
                  jobvl, jobvr, n, A,
                  lda, B, ldb, alphar,
                  alphai, beta, vl, ldvl,
                  vr, ldvr, work, lwork,
                  info, 1, 1)
            end
#                chklapackerror(info[])
              if i == 1
                  lwork = BlasInt(work[1])
                  resize!(work, lwork)
              end
          end
          if cmplx
            alpha, beta, vl, vr
          else
            alphar, alphai, beta, vl, vr
          end
      end
  end
end

#         +---------------+
#>--------|      SVD      |---------<
#         +---------------+

for (gesvd, gesdd, ggsvd, elty, relty) in
  ((:dgesvd_,:dgesdd_,:dggsvd_,:Float64,:Float64),
   (:sgesvd_,:sgesdd_,:sggsvd_,:Float32,:Float32),
   (:zgesvd_,:zgesdd_,:zggsvd_,:ComplexF64,:Float64),
   (:cgesvd_,:cgesdd_,:cggsvd_,:ComplexF32,:Float32))
  @eval begin

      #    SUBROUTINE DGESDD( JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK,
      #                   LWORK, IWORK, INFO )
      #*     .. Scalar Arguments ..
      #      CHARACTER          JOBZ
      #      INTEGER            INFO, LDA, LDU, LDVT, LWORK, M, N
      #*     ..
      #*     .. Array Arguments ..
      #      INTEGER            IWORK( * )
      #      DOUBLE PRECISION   A( LDA, * ), S( * ), U( LDU, * ),
      #                        VT( LDVT, * ), WORK( * )
      function gesdd!(A::Union{AbstractMatrix{$elty},Vector{$elty}},m::Integer,n::Integer;job::Char='O')
          minmn  = min(m, n)
          
          if typeof(A) <: AbstractMatrix
            if job == 'A'
              U  = similar(A, $elty, (m, m))
              VT = similar(A, $elty, (n, n))
            elseif job == 'S'
              U  = similar(A, $elty, (m, minmn))
              VT = similar(A, $elty, (minmn, n))
            elseif job == 'O'
              U  = similar(A, $elty, (m, m >= n ? 0 : m))
              VT = similar(A, $elty, (n, m >= n ? n : 0))
            else #if job == 'N'
              U  = similar(A, $elty, (m, 0))
              VT = similar(A, $elty, (n, 0))
            end
          else
            if job == 'A'
              U  = similar(A, $elty, m*m)
              VT = similar(A, $elty, n*n)
            elseif job == 'S'
              U  = similar(A, $elty, m*minmn)
              VT = similar(A, $elty, minmn*n)
            elseif job == 'O'
              U  = similar(A, $elty, m*(m >= n ? 0 : m))
              VT = similar(A, $elty, n*(m >= n ? n : 0))
            else #if job == 'N'
              U  = similar(A, $elty, 0)
              VT = similar(A, $elty, 0)
            end
          end

          work   = Vector{$elty}(undef, 1)
          lwork  = BlasInt(-1)
          S      = similar(A, $relty, minmn)
          cmplx  = eltype(A)<:Complex
          if cmplx
              rwork = Vector{$relty}(undef, #=job == 'N' ? 7*minmn : =# minmn*max(5*minmn+7, 2*max(m,n)+2*minmn+1))
          end
          iwork  = Vector{BlasInt}(undef, 8*minmn)
          info   = Ref{BlasInt}()

          onestride = max(1,m)
          twostride = max(1,m)
          threestride = max(1,minmn)

          for i = 1:2  # first call returns lwork as work[1]
              if cmplx
                  ccall((@blasfunc($gesdd), libblastrampoline), Cvoid,
                        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                         Ref{BlasInt}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                         Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                         Ptr{$relty}, Ptr{BlasInt}, Ptr{BlasInt}, Clong),
                        job, m, n, A, onestride, S, U, twostride, VT, threestride,
                        work, lwork, rwork, iwork, info, 1)
              else
                  ccall((@blasfunc($gesdd), libblastrampoline), Cvoid,
                        (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                         Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                         Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                         Ptr{BlasInt}, Ptr{BlasInt}, Clong),
                        job, m, n, A, onestride, S, U, twostride, VT, threestride,
                        work, lwork, iwork, info, 1)
              end
              if i == 1
                  # Work around issue with truncated Float32 representation of lwork in
                  # sgesdd by using nextfloat. See
                  # http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=13&t=4587&p=11036&hilit=sgesdd#p11036
                  # and
                  # https://github.com/scipy/scipy/issues/5401
                  lwork = round(BlasInt, nextfloat(real(work[1])))
                  resize!(work, lwork)
              end
          end
          if job == 'O'
            if m >= n
              return (A, S, VT)
            else
              return (U, S, A)
            end
          end
          return (U, S, VT)
      end

      #       SUBROUTINE ZGGSVD( JOBU, JOBV, JOBQ, M, N, P, K, L, A, LDA, B,
      #      $                   LDB, ALPHA, BETA, U, LDU, V, LDV, Q, LDQ, WORK,
      #      $                   RWORK, IWORK, INFO )
      # *     .. Scalar Arguments ..
      #       CHARACTER          JOBQ, JOBU, JOBV
      #       INTEGER            INFO, K, L, LDA, LDB, LDQ, LDU, LDV, M, N, P
      # *     ..
      # *     .. Array Arguments ..
      #       INTEGER            IWORK( * )
      #       DOUBLE PRECISION   ALPHA( * ), BETA( * ), RWORK( * )
      #       COMPLEX*16         A( LDA, * ), B( LDB, * ), Q( LDQ, * ),
      #      $                   U( LDU, * ), V( LDV, * ), WORK( * )
      function ggsvd!(A::AbstractMatrix{$elty}, m::Integer, n::Integer, B::AbstractMatrix{$elty};jobu::AbstractChar='S', jobv::AbstractChar='S', jobq::AbstractChar='Q') #
#        require_one_based_indexing(A, B)
#        chkstride1(A, B)
#        m, n = size(A)
#        minmn = min(m,n)
#        if size(B, 2) != n
#            throw(DimensionMismatch("B has second dimension $(size(B,2)) but needs $n"))
#        end
        p = m #size(B, 1)
        k = Vector{BlasInt}(undef, 1)
        l = Vector{BlasInt}(undef, 1)
        lda = max(1,m)
        ldb = ldb #max(1,stride(B, 2))
        alpha = similar(A, $relty, n)
        beta = similar(A, $relty, n)
        ldu = max(1, m)
        U = jobu == 'U' ? similar(A, $elty, ldu, m) : similar(A, $elty, 0)
        ldv = max(1, p)
        V = jobv == 'V' ? similar(A, $elty, ldv, p) : similar(A, $elty, 0)
        ldq = max(1, n)
        Q = jobq == 'Q' ? similar(A, $elty, ldq, n) : similar(A, $elty, 0)
        work = Vector{$elty}(undef, max(3n, m, p) + n)
        cmplx = eltype(A) <: Complex
        if cmplx
            rwork = Vector{$relty}(undef, 2n)
        end
        iwork = Vector{BlasInt}(undef, n)
        info = Ref{BlasInt}()
        if cmplx
            ccall((@blasfunc($ggsvd), libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ptr{$relty}, Ptr{BlasInt}, Ptr{BlasInt},
                Clong, Clong, Clong),
                jobu, jobv, jobq, m,
                n, p, k, l,
                A, lda, B, ldb,
                alpha, beta, U, ldu,
                V, ldv, Q, ldq,
                work, rwork, iwork, info,
                1, 1, 1)
        else
            ccall((@blasfunc($ggsvd), libblastrampoline), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                Ptr{$elty}, Ptr{BlasInt}, Ptr{BlasInt},
                Clong, Clong, Clong),
                jobu, jobv, jobq, m,
                n, p, k, l,
                A, lda, B, ldb,
                alpha, beta, U, ldu,
                V, ldv, Q, ldq,
                work, iwork, info,
                1, 1, 1)
        end
#        chklapackerror(info[])
        if m - k[1] - l[1] >= 0
#            triu(A[1:k[1] + l[1],n - k[1] - l[1] + 1:n])
            R = Array{$elty,2}(undef,k[1]+l[1],k[1] + l[1] - 1)
            for y = 1:size(R,2)
              Ay = n-k[1]-l[1]+y
              tempind = size(R,1)*(y-1)
              for x = 1:k[1]+l[1]
                R[x + tempind] = A[x,Ay]
              end
            end
        else
#            R = triu([A[1:m, n - k[1] - l[1] + 1:n]; B[m - k[1] + 1:l[1], n - k[1] - l[1] + 1:n]])
            R = Array{$elty,2}(undef,m,k[1] + l[1] - 1)
            for y = 1:size(R,2)
              Ay = n-k[1]-l[1]+y
              tempind = size(R,1)*(y-1)
              for x = 1:m
                R[x + tempind] = A[x,Ay]
              end
            end
        end
        U, V, Q, alpha, beta, k[1], l[1], R
    end


    # SUBROUTINE DGESVD( JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, INFO )
    # *     .. Scalar Arguments ..
    #       CHARACTER          JOBU, JOBVT
    #       INTEGER            INFO, LDA, LDU, LDVT, LWORK, M, N
    # *     .. Array Arguments ..
    #       DOUBLE PRECISION   A( LDA, * ), S( * ), U( LDU, * ),
    #      $                   VT( LDVT, * ), WORK( * )
    function gesvd!(jobu::AbstractChar, jobvt::AbstractChar, A::AbstractMatrix{$elty})
#      require_one_based_indexing(A)
#      chkstride1(A)
      m, n   = size(A)
      minmn  = min(m, n)
      S      = similar(A, $relty, minmn)
      U      = similar(A, $elty, jobu  == 'A' ? (m, m) : (jobu  == 'S' ? (m, minmn) : (m, 0)))
      VT     = similar(A, $elty, jobvt == 'A' ? (n, n) : (jobvt == 'S' ? (minmn, n) : (n, 0)))
      work   = Vector{$elty}(undef, 1)
      cmplx  = eltype(A) <: Complex
      if cmplx
          rwork = Vector{$relty}(undef, 5minmn)
      end
      lwork  = BlasInt(-1)
      info   = Ref{BlasInt}()
      for i in 1:2  # first call returns lwork as work[1]
          if cmplx
              ccall((@blasfunc($gesvd), libblastrampoline), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                      Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{$elty},
                      Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                      Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt}, Clong, Clong),
                    jobu, jobvt, m, n, A, max(1,stride(A,2)), S, U, max(1,stride(U,2)), VT, max(1,stride(VT,2)),
                    work, lwork, rwork, info, 1, 1)
          else
              ccall((@blasfunc($gesvd), libblastrampoline), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                      Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                      Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                      Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
                    jobu, jobvt, m, n, A, max(1,stride(A,2)), S, U, max(1,stride(U,2)), VT, max(1,stride(VT,2)),
                    work, lwork, info, 1, 1)
          end
          chklapackerror(info[])
          if i == 1
              lwork = BlasInt(real(work[1]))
              resize!(work, lwork)
          end
      end
      if jobu == 'O'
          return (A, S, VT)
      elseif jobvt == 'O'
          return (U, S, A)
      else
          return (U, S, VT)
      end
    end


  end
end

#         +---------------+
#>--------|     QR/LQ     |---------<
#         +---------------+



for (orglq, orgqr, orgql, orgrq,  elty) in
  ((:dorglq_,:dorgqr_,:dorgql_,:dorgrq_,:Float64),
   (:sorglq_,:sorgqr_,:sorgql_,:sorgrq_,:Float32),
   (:zunglq_,:zungqr_,:zungql_,:zungrq_,:ComplexF64),
   (:cunglq_,:cungqr_,:cungql_,:cungrq_,:ComplexF32))
  @eval begin
      # SUBROUTINE DORGLQ( M, N, K, A, LDA, TAU, WORK, LWORK, INFO )
      # *     .. Scalar Arguments ..
      #       INTEGER            INFO, K, LDA, LWORK, M, N
      # *     .. Array Arguments ..
      #       DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
      function orglq!(A::Union{AbstractMatrix{$elty},Vector{$elty}}, m::Integer, n::Integer, tau::AbstractVector{$elty}, k::Integer = length(tau))
          mindim = min(m,n) #length(tau) #min(m, n)
          
          work  = Vector{$elty}(undef, 1)
          lwork = BlasInt(-1)
          info  = Ref{BlasInt}()
          for i = 1:2  # first call returns lwork as work[1]
              ccall((@blasfunc($orglq), libblastrampoline), Cvoid,
                    (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                     mindim, n, k, A, max(1,m), tau, work, lwork, info)
              #chklapackerror(info[])
              if i == 1
                  lwork = BlasInt(real(work[1]))
                  resize!(work, lwork)
              end
          end
          if mindim < m
            if typeof(A) <: Vector
              outA = Array{$elty,1}(undef,mindim*n)
              for y = 1:n
                thisind = mindim*(y-1)
                thisotherind = m*(y-1)
                @inbounds @simd for x = 1:mindim
                  outA[x + thisind] = A[x + thisotherind]
                end
              end
              outA
            else
              A[1:mindim,:]
            end
          else
              A
          end
      end

      # SUBROUTINE DORGQR( M, N, K, A, LDA, TAU, WORK, LWORK, INFO )
      # *     .. Scalar Arguments ..
      #       INTEGER            INFO, K, LDA, LWORK, M, N
      # *     .. Array Arguments ..
      #       DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
      function orgqr!(A::Union{AbstractMatrix{$elty},Vector{$elty}}, m::Integer, n::Integer, tau::AbstractVector{$elty}, k::Integer = length(tau))
          mindim = min(m, n)

          work  = Vector{$elty}(undef, 1)
          lwork = BlasInt(-1)
          info  = Ref{BlasInt}()
          for i = 1:2  # first call returns lwork as work[1]
              ccall((@blasfunc($orgqr), libblastrampoline), Cvoid,
                    (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                    m, mindim, k, A,
                    max(1,m), tau, work, lwork,
                    info)
              #chklapackerror(info[])
              if i == 1
                  lwork = BlasInt(real(work[1]))
                  resize!(work, lwork)
              end
          end
          if mindim < n
            if typeof(A) <: Vector
              A[1:m*mindim]
            else
              A[:,1:mindim]
            end
          else
              A
          end
      end
        # SUBROUTINE DORGQL( M, N, K, A, LDA, TAU, WORK, LWORK, INFO )
        # *     .. Scalar Arguments ..
        #       INTEGER            INFO, K, LDA, LWORK, M, N
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
        function orgql!(A::Union{AbstractMatrix{$elty},Vector{$elty}}, m::Integer, n::Integer, tau::AbstractVector{$elty}, k::Integer = length(tau))
          mindim = min(m,n)
          work  = Vector{$elty}(undef, 1)
          lwork = BlasInt(-1)
          info  = Ref{BlasInt}()
          for i = 1:2  # first call returns lwork as work[1]
              ccall((@blasfunc($orgql), libblastrampoline), Cvoid,
                    (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                    mindim, n, k, A,
                    max(1,m), tau, work, lwork,
                    info)
#              chklapackerror(info[])
              if i == 1
                  lwork = BlasInt(real(work[1]))
                  resize!(work, lwork)
              end
          end
          if mindim < n
            if typeof(A) <: Vector
              A[1:m*mindim]
            else
              A[:,1:mindim]
            end
          else
              A
          end
      end

      # SUBROUTINE DORGRQ( M, N, K, A, LDA, TAU, WORK, LWORK, INFO )
      # *     .. Scalar Arguments ..
      #       INTEGER            INFO, K, LDA, LWORK, M, N
      # *     .. Array Arguments ..
      #       DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
      function orgrq!(A::Union{AbstractMatrix{$elty},Vector{$elty}}, m::Integer, n::Integer, tau::AbstractVector{$elty}, k::Integer = length(tau))
          mindim = min(m,n)
          work  = Vector{$elty}(undef, 1)
          lwork = BlasInt(-1)
          info  = Ref{BlasInt}()
          for i = 1:2  # first call returns lwork as work[1]
              ccall((@blasfunc($orgrq), libblastrampoline), Cvoid,
                    (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                    m, mindim, k, A,
                    max(1,m), tau, work, lwork,
                    info)
              chklapackerror(info[])
              if i == 1
                  lwork = BlasInt(real(work[1]))
                  resize!(work, lwork)
              end
          end
          if mindim < m
            if typeof(A) <: Vector
              outA = Array{$elty,1}(undef,mindim*n)
              for y = 1:n
                thisind = mindim*(y-1)
                thisotherind = m*(y-1)
                @inbounds @simd for x = 1:mindim
                  outA[x + thisind] = A[x + thisotherind]
                end
              end
              outA
            else
              A[1:mindim,:]
            end
          else
              A
          end
      end
  end
end

for (gebrd,gelqf, geqrf, elty, relty) in
  ((:dgebrd_,:dgelqf_,:dgeqrf_,:Float64,:Float64),
   (:sgebrd_,:sgelqf_,:sgeqrf_,:Float32,:Float32),
   (:zgebrd_,:zgelqf_,:zgeqrf_,:ComplexF64,:Float64),
   (:cgebrd_,:cgelqf_,:cgeqrf_,:ComplexF32,:Float32))
  @eval begin

    #bidiagonal form P*B*Q'

      # SUBROUTINE DGEBRD( M, N, A, LDA, D, E, TAUQ, TAUP, WORK, LWORK,
      #                    INFO )
      # .. Scalar Arguments ..
      # INTEGER            INFO, LDA, LWORK, M, N
      # .. Array Arguments ..
      #  DOUBLE PRECISION   A( LDA, * ), D( * ), E( * ), TAUP( * ),
      #           TAUQ( * ), WORK( * )
      function gebrd!(A::AbstractMatrix{$elty}) #gebrd!
#          require_one_based_indexing(A)
#          chkstride1(A)
          m, n  = size(A)
          k     = min(m, n)
          d     = similar(A, $relty, k) #diagonal of B
          e     = similar(A, $relty, k) #off diagonal of B
          tauq  = similar(A, $elty, k) #reflector storage for Q
          taup  = similar(A, $elty, k) #reflector storage for P
          work  = Vector{$elty}(undef, 1)
          lwork = BlasInt(-1)
          info  = Ref{BlasInt}()
          for i = 1:2  # first call returns lwork as work[1]
              ccall((@blasfunc($gebrd), libblastrampoline), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ptr{$elty},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                    m, n, A, max(1,m),
                    d, e, tauq, taup,
                    work, lwork, info)
#              chklapackerror(info[])
              if i == 1
                  lwork = BlasInt(real(work[1]))
                  resize!(work, lwork)
              end
          end
          A, d, e, tauq, taup
      end

      # SUBROUTINE DGELQF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
      # *     .. Scalar Arguments ..
      #       INTEGER            INFO, LDA, LWORK, M, N
      # *     .. Array Arguments ..
      #       DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
      function gelqf!(A::Union{AbstractMatrix{$elty},Vector{$elty}}, m::Integer, n::Integer, tau::AbstractVector{$elty})
          lda   = BlasInt(max(1,m))
          
          lwork = BlasInt(-1)
          work  = Vector{$elty}(undef, 1)
          info  = Ref{BlasInt}()
          for i = 1:2  # first call returns lwork as work[1]
              ccall((@blasfunc($gelqf), libblastrampoline), Cvoid,
                    (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                     Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                    m, n, A, lda, tau, work, lwork, info)
              #chklapackerror(info[])
              if i == 1
                  lwork = BlasInt(real(work[1]))
                  resize!(work, lwork)
              end
          end
          A, tau
      end

      ## geqrf! - positive elements on diagonal of R - not defined yet
      # SUBROUTINE DGEQRFP( M, N, A, LDA, TAU, WORK, LWORK, INFO )
      # *     .. Scalar Arguments ..
      #       INTEGER            INFO, LDA, LWORK, M, N
      # *     .. Array Arguments ..
      #       DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
      function geqrf!(A::Union{AbstractMatrix{$elty},Vector{$elty}}, m::Integer, n::Integer, tau::AbstractVector{$elty})
          work  = Vector{$elty}(undef, 1)
          lwork = BlasInt(-1)
          info  = Ref{BlasInt}()
          for i = 1:2                # first call returns lwork as work[1]
              ccall((@blasfunc($geqrf), libblastrampoline), Cvoid,
                    (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                     Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                    m, n, A, max(1,m), tau, work, lwork, info)
              #chklapackerror(info[])
              if i == 1
                  lwork = BlasInt(real(work[1]))
                  resize!(work, lwork)
              end
          end
          A, tau
      end

  end
end

libQRmake! = orgqr!
libLQmake! = orglq!
libRQmake! = orgrq!
libQLmake! = orgql!

libLowerHessenberg! = gelqf!
libUpperHessenberg! = geqrf!

function libqr!(A::Array{W,P},a::Integer,b::Integer;m::Integer=min(a,b)) where {W <: Number, P}
  X,Y = LinearAlgebra.qr(A)

  tau = Array{W,1}(undef,m)
  G,tau = libUpperHessenberg!(A,a,b,tau) #geqrf!

  R = P == 1 ? Array{W,1}(undef,m*b) : Array{W,2}(undef,m,b)

  for y = 1:b
    thisind = m*(y-1)
    thisotherind = a*(y-1)
    @inbounds @simd for x = 1:min(y,m)
      R[x + thisind] = A[x + thisotherind]
    end
    @inbounds @simd for x = y+1:m
      R[x + thisind] = 0
    end
  end

  sizetau = length(tau)
  @inbounds while sizetau > 0 && !isapprox(tau[sizetau],0)
    sizetau -= 1
  end

  Q = libQRmake!(G,a,b,tau,sizetau) #orgqr
  return Q,R
end

function libqr(A::Array{W,2},a::Integer,b::Integer;m::Integer=min(a,b)) where W <: Number
  return libqr!(copy(A),a,b,m=m)
end

function libqr(A::tens{W},a::Integer,b::Integer;m::Integer=min(a,b)) where W <: Number
  Q,R = libqr!(copy(A.T),a,b,m=m)
  return tens{W}([a,m],Q),tens{W}([m,b],R)
end

function libqr!(A::tens{W},a::Integer,b::Integer;m::Integer=min(a,b)) where W <: Number
  Q,R = libqr!(A.T,a,b,m=m)
  return tens{W}([a,m],Q),tens{W}([m,b],R)
end

function liblq!(A::Array{W,P},a::Integer,b::Integer;m::Integer=min(a,b)) where {W <: Number, P}
  tau = Array{W,1}(undef,m)
  G,tau = libLowerHessenberg!(A,a,b,tau) #gelqf!
  
  L = P == 1 ? Array{W,1}(undef,a*m) : Array{W,2}(undef,a,m)

  for y = 1:m
    thisind = a*(y-1)
    @inbounds @simd for x = 1:y-1
      L[x+thisind] = 0
    end
    @inbounds @simd for x = y:a
      L[x+thisind] = G[x+thisind]
    end
  end

  sizetau = length(tau)
  @inbounds while sizetau > 0 && !isapprox(tau[sizetau],0)
    sizetau -= 1
  end

  Q = libLQmake!(G,a,b,tau,sizetau) #orglq

  return L,Q
end

function liblq(A::Array{W,2},a::Integer,b::Integer;m::Integer=min(a,b)) where W <: Number
  return liblq!(copy(A),a,b,m=m)
end

function liblq(A::tens{W},a::Integer,b::Integer;m::Integer=min(a,b)) where W <: Number
  L,Q = liblq!(copy(A.T),a=a,b=b,m=m)
  return tens{W}([a,m],L),tens{W}([m,b],Q)
end

function liblq!(A::tens{W},a::Integer,b::Integer;m::Integer=min(a,b)) where W <: Number
  L,Q = liblq!(A.T,a,b,m=m)
  return tens{W}([a,m],L),tens{W}([m,b],Q)
end

#         +---------------+
#>--------|     Schur     |---------<
#         +---------------+
