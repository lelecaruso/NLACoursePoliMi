#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/IterativeLinearSolvers>


using std::endl;
using std::cout;

// Some useful alias
using namespace Eigen;
using SpMat=Eigen::SparseMatrix<double,RowMajor>;

int main(){
    int n = 100;
    SpMat A(n,n);

    for(int i=0; i<n; i++){
        A.coeffRef(i,i) = 8.0;
        if(i>0)A.coeffRef(i,i-1) = -2.0;
        if(i<n-1)A.coeffRef(i,i+1) = -4.0;
        if(i<n-2)A.coeffRef(i,i+2) = -1.0;
    }

    cout<<"The norm of matrix A is: "<< A.norm()<<endl;

    //Solve the EigenValue Problem Ax = lmbda x reporting smallest and biggest eig
    MatrixXd A_dense = MatrixXd(A);
    EigenSolver<MatrixXd> eigensolver(A_dense);
    if (eigensolver.info() != Eigen::Success) abort();
    cout << "The eigenvalues of A are:\n" << eigensolver.eigenvalues() << endl;
    cout<< "The biggest eigenvalue of A is: "<<eigensolver.eigenvalues().real().maxCoeff()<<endl;
    cout<< "The smallest eigenvalue of A is: "<<eigensolver.eigenvalues().real().minCoeff()<<endl;

    saveMarket(A,"Aex2.mtx");

    /*
    Using the proper iterative solver available in the LIS library compute the largest eigenvalue λmax of
    A up to a tolerance of 10−7. Report on the .txt file the computed eigenvalue and the
    number of iterations required to achieve the prescribed tolerance.
    */

   /*
    mpirun -n 4 ./eigen1 Aex2.mtx  eigvec.txt hist.txt -e pi -etol 1.0e-7 -emaxiter 30000
    eigensolver           : Power

    Power: mode number          = 0
    Power: eigenvalue           = 1.299901e+01
    Power: number of iterations = 30000

    Power: relative residual    = 9.486912e-07
    
   */

  /*
  Find a shift μ ∈ R yielding an acceleration of the previous eigensolver. Report μ and the
    number of iterations required to achieve a tolerance of 10−7.*/
/*
    # mpirun -n 4 ./eigen1 Aex2.mtx  eigvec.txt hist.txt -e pi -etol 1.0e-7 -emaxiter 30000 -shift 7.0

eigensolver           : Power
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-07 * ||lx||_2

shift                 : 7.000000e+00

Power: eigenvalue           = 1.299901e+01
Power: number of iterations = 19920

Power: relative residual    = 9.997483e-08
*/

//Find the smallest eigenvalues using Lis upto tol 1.0 e-7
/*
mpirun -n 4 ./eigen1 Aex2.mtx  eigvec.txt hist.txt -e ii -etol 1.0e-7 -i bicg -emaxiter 1500

number of processes = 4
matrix size = 100 x 100 (396 nonzero entries)

initial vector x      : all components set to 1
precision             : double
eigensolver           : Inverse
convergence condition : ||lx-(B^-1)Ax||_2 <= 1.0e-07 * ||lx||_2
matrix storage format : CSR
shift                 : 0.000000e+00
linear solver         : BiCG
preconditioner        : none
eigensolver status    : normal end

Inverse: mode number          = 0
Inverse: eigenvalue           = 1.913297e+00
Inverse: number of iterations = 1348
Inverse: elapsed time         = 1.015674e-01 sec.
Inverse:   preconditioner     = 8.238490e-04 sec.
Inverse:     matrix creation  = 5.195000e-05 sec.
Inverse:   linear solver      = 8.276006e-02 sec.
Inverse: relative residual    = 9.961482e-08

*/

}