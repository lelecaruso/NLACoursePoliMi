#include <Eigen/Sparse>     
#include <iostream>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>
#include <Eigen/SparseLU>
#include "bcgstab.hpp"

using namespace Eigen; //shorcut for eigen declaration
using SpMat=SparseMatrix<double,RowMajor>;
using namespace std;

int main(int argc, char ** agrv){
using namespace LinearAlgebra;
//Loading the matrix using loadmarket functions
SpMat A1;
loadMarket(A1,"A_1.mtx");

SpMat A2;
loadMarket(A2,"A_2.mtx");

SpMat A = A1*A2;

cout<<"The norm of A is: "<< A.norm()<<endl;

/*
Solve the linear system Ax = b, with b = (1,1,...,1)T, using both the SparseLU method
available in Eigen and the BICGSTAB method implemented in the bcgstab.hpp template.
For the BICGSTAB method fix a maximum number of iterations sufficient to reduce the
residual below than 10−10 and use the diagonal preconditioner provided by Eigen. Report
on the sheet ‖xLU ‖, ‖xBCG‖, and ‖xLU −xBCG‖, where xLU and xBCG are the approximate
solution obtained with the LU and BICGSTAB methods, respectively.
*/

VectorXd b = VectorXd:: Ones(A.cols());
VectorXd xlu(A.cols());

//Using LU factorization
SparseLU<Eigen::SparseMatrix<double> > solvelu;    // define solver
  solvelu.compute(A);
  if(solvelu.info()!=Eigen::Success) {                     // sanity check
      cout << "cannot factorize the matrix" << endl; 
      return 0;
  }
  xlu = solvelu.solve(b);    

  cout<<"The norm of xlu is: "<<xlu.norm()<<endl;

    VectorXd x(A.rows());
    double tol;
    int result,maxit;
  // Solve with BiCGSTAB method
  x=0*x; maxit = int(A.cols()); tol = 1.e-10;
  Eigen::DiagonalPreconditioner<double> D(A);
  result = LinearAlgebra::BiCGSTAB(A, x, b, D, maxit, tol);
  cout << "BiCGSTAB   flag = " << result << endl;
  cout << "iterations performed: " << maxit << endl;
  cout << "tolerance achieved  : " << tol << endl;


  cout<<"The norm of xbicg is: "<<x.norm()<<endl;

  cout<<"The difference of (xlu-xbicg).norm  is: "<<(xlu-x).norm()<<endl;
  
return 0;
}