#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense> //per eigensolver serve il modulo dende
#include <unsupported/Eigen/SparseExtra>


using std::endl;
using std::cout;

// Some useful alias
using namespace Eigen;
using SpMat=Eigen::SparseMatrix<double,RowMajor>;
int main(){

    SpMat A;
    loadMarket(A,"Aexer2.mtx");
    /*
    Report on the sheet the matrix size, the number
    of non-zero entries, and the the Euclidean norm of A. Is the matrix A symmetric?
    */

   cout<<"The matrix A size is: "<<A.rows()<<" X "<<A.cols()<<endl;
   cout<<"The number of non Zeros of A is: "<<A.nonZeros()<<endl;
   cout<<"The norm of matrix A is: "<<A.norm()<<endl;

   double tol_sym = 1.0e-14;
   SpMat At = A.transpose();
   SpMat B = A-At;
   if(B.norm()<= tol_sym){
    cout<<"La matrice A è simmetrica"<<endl;
   }
   else{
     cout<<"La matrice A NON è simmetrica"<<endl;
   }

   //solve the eigenvalue problem

    // Compute Eigenvalues of the original matrix
  MatrixXd A_dense;
  A_dense = MatrixXd(A);
  Eigen:: EigenSolver<Eigen:: MatrixXd> eigensolver(A_dense);
  if (eigensolver.info() != Eigen::Success) abort();
  std::cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << std::endl;
  /*
  min (0.0685231,0)
  max (30.6515,0)
  */

  /*
   Solve the eigenvalue problem ATAx = λx using the proper solver provided by the Eigen
    library. Report on the sheet the smallest and largest computed eigenvalues of A. Which is
    he relation between the eigenvalues of A and the eigenvalues of ATA
  */

 SpMat ATA = At * A;
 //ATA is SPD i can use selfadjoint

 Eigen:: SelfAdjointEigenSolver<MatrixXd> eigensolversym(ATA);
 if (eigensolversym.info() != Eigen::Success) abort();
  std::cout << "The eigenvalues of ATA are:\n" << eigensolversym.eigenvalues() << std::endl;
    /*
      min: 0.0046876
      max: 939.518
    */


/*
Using the proper iterative solver
available in the LIS library compute the largest eigenvalue of A up to a tolerance of 10−10.
Report the computed eigenvalue and the number of iterations required to achieve the
prescribed tolerance
*/

/*

Power: eigenvalue           = 3.065148e+01
Power: number of iterations = 8388

Power: relative residual    = 9.986513e-11

*/


/*
Compute the three smallest eigenvalue of the Aexer2.mtx matrix up to a tolerance of
10−12. Explore different inner iterative methods and preconditioners (at least 3 alternative
strategies). Report on the sheet the iteration counts and the residual at the last iteration.
*/

// i know all the spectrum so i use ipm with shift


/*

# mpirun -n 4 ./eigen1 Aexer2.mtx eigvec.txt hist.txt -e pi -etol 1.0e-10

number of processes = 4
matrix size = 64 x 64 (189 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Power
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-10 * ||lx||_2
matrix storage format : CSR
shift                 : 0.000000e+00
eigensolver status    : LIS_MAXITER(code=4)

Power: mode number          = 0
Power: eigenvalue           = 3.065147e+01
Power: number of iterations = 1000
Power: elapsed time         = 1.185726e-02 sec.
Power:   preconditioner     = 0.000000e+00 sec.
Power:     matrix creation  = 0.000000e+00 sec.
Power:   linear solver      = 0.000000e+00 sec.
Power: relative residual    = 1.287065e-04

root@0bc56b79e068 test # mpirun -n 4 ./eigen1 Aexer2.mtx eigvec.txt hist.txt -e pi -etol 1.0e-10 -emaxiter 10000

number of processes = 4
matrix size = 64 x 64 (189 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Power
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-10 * ||lx||_2
matrix storage format : CSR
shift                 : 0.000000e+00
eigensolver status    : normal end

Power: mode number          = 0
Power: eigenvalue           = 3.065148e+01
Power: number of iterations = 8388
Power: elapsed time         = 5.160947e-02 sec.
Power:   preconditioner     = 0.000000e+00 sec.
Power:     matrix creation  = 0.000000e+00 sec.
Power:   linear solver      = 0.000000e+00 sec.
Power: relative residual    = 9.986513e-11

root@0bc56b79e068 test # mpirun -n 4 ./eigen1 Aexer2.mtx eigvec.txt hist.txt -e ii -etol 1.0e-12     -i bicg      

number of processes = 4
matrix size = 64 x 64 (189 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Inverse
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-12 * ||lx||_2
matrix storage format : CSR
shift                 : 0.000000e+00
linear solver         : BiCG
preconditioner        : none
eigensolver status    : normal end

Inverse: mode number          = 0
Inverse: eigenvalue           = 6.852311e-02
Inverse: number of iterations = 42
Inverse: elapsed time         = 1.322231e-02 sec.
Inverse:   preconditioner     = 1.196560e-04 sec.
Inverse:     matrix creation  = 2.117000e-06 sec.
Inverse:   linear solver      = 1.198739e-02 sec.
Inverse: relative residual    = 6.833831e-13

root@0bc56b79e068 test # mpirun -n 4 ./eigen1 Aexer2.mtx eigvec.txt hist.txt -e ii -etol 1.0e-12 -shift 0.11 -i bicg

number of processes = 4
matrix size = 64 x 64 (189 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Inverse
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-12 * ||lx||_2
matrix storage format : CSR
shift                 : 1.100000e-01
linear solver         : BiCG
preconditioner        : none
eigensolver status    : normal end

Inverse: mode number          = 0
Inverse: eigenvalue           = 1.238031e-01
Inverse: number of iterations = 30
Inverse: elapsed time         = 2.789131e-02 sec.
Inverse:   preconditioner     = 1.732100e-04 sec.
Inverse:     matrix creation  = 1.985000e-06 sec.
Inverse:   linear solver      = 2.493667e-02 sec.
Inverse: relative residual    = 3.449960e-13

root@0bc56b79e068 test # mpirun -n 4 ./eigen1 Aexer2.mtx eigvec.txt hist.txt -e ii -etol 1.0e-12 -shift 0.14 -i bicg

number of processes = 4
matrix size = 64 x 64 (189 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Inverse
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-12 * ||lx||_2
matrix storage format : CSR
shift                 : 1.400000e-01
linear solver         : BiCG
preconditioner        : none
eigensolver status    : normal end

Inverse: mode number          = 0
Inverse: eigenvalue           = 1.500000e-01
Inverse: number of iterations = 58
Inverse: elapsed time         = 2.479617e-02 sec.
Inverse:   preconditioner     = 2.388150e-04 sec.
Inverse:     matrix creation  = 2.525000e-06 sec.
Inverse:   linear solver      = 2.361130e-02 sec.
Inverse: relative residual    = 7.191999e-13


*/
    return 0;
}
