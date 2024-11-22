#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>


using namespace std;

using namespace Eigen; //shorcut for eigen declaration
using SpMat=SparseMatrix<double,RowMajor>;

#include "gmres.hpp"

int main(int argc, char* argv[]) {

    using namespace LinearAlgebra;

    int n = 1000;    
    SpMat B(n,n);
    for(int i=0;i<n;i++){
        B.coeffRef(i,i) = 2*(i+1);
        if(i>0){B.coeffRef(i,i-1) = i;}
        if(i<n-1){B.coeffRef(i,i+1) = i+1;}
    }

    SpMat C(n,n);
       for(int i=0; i<n ;i++){
        C.coeffRef(i,n-i-1) = -(i + 1); 
       }  

    SpMat A(n,n);
    A = B+C;

    VectorXd xe = VectorXd:: Ones(n);
    VectorXd b = A*xe;
    VectorXd x = VectorXd:: Constant(A.cols(),0);

    int max_iter = n;
    double tol = 1.0e-12;
    Eigen::DiagonalPreconditioner<double> D(A);
    int no_restart = max_iter;
    int result;
    result = GMRES(A,x,b,D,no_restart,max_iter,tol);
    std::cout <<" GMRES iter_sol no restart"<< endl;
    std::cout << "#iterations:     " << max_iter<< endl;
    std::cout << "toll achived: " <<tol << endl;
    std::cout << "effective error: "<<(x-xe).norm()<< endl;

    //Using restart

    int restart = 50; //Fixing dimension of Krylov space.
    x = x*0;
    max_iter = n;
    tol = 1.0e-12;
    result = GMRES(A,x,b,D,restart,max_iter,tol);
    std::cout <<" GMRES iter_sol with RESTART "<< endl;
    std::cout << "#iterations:     " << max_iter<< endl;
    std::cout << "toll achived: " <<tol << endl;
    std::cout << "effective error: "<<(x-xe).norm()<< endl;

    return result;
}
