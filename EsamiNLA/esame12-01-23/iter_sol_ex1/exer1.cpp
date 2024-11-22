#include <cstdlib>                      // System includes
#include <iostream>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

using std::endl;
using std::cout;

#include"gmres.hpp"

using namespace Eigen;

int main(int argc, char** argv)
{
  using namespace LinearAlgebra;
  // Some useful alias
  using SpMat=Eigen::SparseMatrix<double,RowMajor>;


  SpMat A;
  loadMarket(A,"A_stokes.mtx");
  cout<<"Le dimensioni di A sono: "<<A.rows()<<" X "<< A.cols()<<endl;

  //check symm and report the norm
  SpMat AT = A.transpose();
  SpMat B = A-AT;

  double tol_symm = 1.0e-14;
  if(B.norm()<=tol_symm){
    cout<<"La matrice A è simmetrica!"<<endl;
  }
  else{
    cout<<"La matrice A non è simmetrica!"<<endl;
    }
  //A isnt symm
  cout<<"La norma della mat A e: "<<A.norm()<<endl;
  VectorXd b = VectorXd:: Ones(A.cols());
  Eigen::DiagonalPreconditioner<double> D(A);

  int result, restart, max_iter;
  max_iter = A.rows();
  restart = max_iter; //no use of restart since the max size of krylov Space is equivalent to dim(A)
  
  double tol = 1.0e-9;

  VectorXd x(A.cols());
  result = GMRES(A,x,b,D,restart,max_iter,tol);
  cout << "GMRES flag (0 convergence) = " << result << endl;
  cout << "tollerance achieved: " << tol << endl;
  cout << "iterations performed: " << max_iter << endl;

  cout<<"La prima componente della soluzione: "<<x(0)<<endl;

  int n = A.rows();
  SpMat Lap(n,n);
  for(int i = 0; i<n ; i++){
    Lap.coeffRef(i,i) = 2.0;
    if(i>0){ Lap.coeffRef(i,i-1) = 1.0; }
    if(i<n-1){ Lap.coeffRef(i,i+1) = -1.0; }
  }

  Lap = A.norm() * Lap;

  SpMat A_tilde = A + Lap;

  //need the norm of sym part of A_tilde 2*AS =  ̃A +  ̃AT 
  SpMat A_tildeT = A_tilde.transpose();
  SpMat A_s_tilde = (A_tildeT + A_tilde) * 0.5;
  cout<<"The norm of the sym part (A_tilde_S) of A_tilde  is: "<<A_s_tilde.norm()<<endl;
  saveMarket(A_tilde,"A_tilde.mtx");

//LIS GMRES
  /*
    Using the GMRES iterative solver available in the LIS
    library compute the approximate solution of the linear system  ̃Ax = b up to a tolerance of
    10−9. Explore at least two different preconditioning strategies that yield a decrease in the
    number of required iterations with respect to the GMRES method without preconditioning.
    Report on the sheet the iteration counts and the relative residual at the last iteration
  */

 /*
  mpicc -DUSE_MPI -I${mkLisInc} -L${mkLisLib} -llis test1.c -o test1
mpirun -n 4 ./test1 A_tilde.mtx 1 sol.txt hist.txt -i gmres -tol 1.0e-9

number of processes = 4
matrix size = 420 x 420 (7984 nonzero entries)

linear solver         : GMRES
preconditioner        : none


GMRES: number of iterations = 21
GMRES: relative residual    = 8.909813e-10
 */

// another is with -p ssor 
// GMRES: number of iterations = 12
//GMRES: relative residual    = 2.210331e-10

// preconditioner        : Jacobi(Inner sys of subdomains) + Additive Schwarz
// GMRES: number of iterations = 11
//GMRES: relative residual    = 3.938411e-10

  return 0;
  }
