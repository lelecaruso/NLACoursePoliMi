#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <string>
#include <Eigen/IterativeLinearSolvers>


using namespace std;
using namespace Eigen;
using SpMat=Eigen::SparseMatrix<double,RowMajor>;
//When using VectorXd function like VectorXd b = VectorXd:: Ones(A.cols());

int main(int argc, char** argv){

    SpMat A;
    loadMarket(A,"A_exam1.mtx");
    SpMat AT = A.transpose();

    SpMat ATA = AT * A;

    cout<<"The norm of the matrix ATA is: "<<ATA.norm()<<endl;
    cout<<"The size of ATA is"<<ATA.rows()<<" x "<<ATA.cols()<<endl;

    VectorXd xe = VectorXd:: Ones(A.cols()); //important to define the solution of size cols
    VectorXd b = A*xe;

    cout<<"The size of b is: "<<b.size()<<endl;
    cout<<"The norm of the vector is: "<<b.norm()<<endl;

    VectorXd xqr(A.cols());
    Eigen::SparseQR<Eigen::SparseMatrix<double,RowMajor>, COLAMDOrdering<int>> solver;   
    solver.compute(A);
    if(solver.info()!=Eigen::Success) {                     // sanity check
      std::cout << "cannot factorize the matrix" << std::endl; 
      return 0;
    }
    xqr = solver.solve(b);// solve
    cout<<"The error norm is: "<<(xe-xqr).norm()<<endl;

    VectorXd xend(A.cols()); //null initial guess (by default is zero)
    double tol = 1.0e-8;
    int maxiter = 1000;

    Eigen::ConjugateGradient<SpMat, Eigen::Lower|Eigen::Upper> cg;
    cg.setTolerance(tol);
    cg.setMaxIterations(maxiter);
    cg.compute(ATA);
    VectorXd rhs = AT * b;
    cout<<"The size of Atb is: "<<rhs.size()<<endl;

    xend = cg.solve(rhs);
    std::cout << " Eigen native CG" << endl;
    std::cout << "#iterations:     " << cg.iterations() << endl;
    std::cout << "tolerance achived: " << cg.error()<< endl;
    std::cout << "effective error: " << (xend-xe).norm() << endl;

    cout<< "the error between the two computed sol is: "<<(xend-xqr).norm()<<endl;
    

    MatrixXd A1_dense = A.topLeftCorner(A.cols(),A.cols());
    MatrixXd A2_dense = A.bottomRightCorner(A.cols(),A.cols());

    SpMat A1 = A1_dense.sparseView();
    SpMat A2 = A2_dense.sparseView();

    cout<<"The norm of the matrix A1 is: "<<A1.norm()<<endl;

    VectorXd b1 = b.head(A.cols());
    VectorXd b2 = b.tail(A.cols());

    cout<<"The norm of the vector b2 is: "<<b2.norm()<<endl;

    
    VectorXd x1(A1.cols());

    double tol1 = 1.0e-8;
    int maxiter1 = 3000;

    Eigen::DiagonalPreconditioner<double> D(A1);// Create diagonal preconditioner
    Eigen::BiCGSTAB<SpMat,DiagonalPreconditioner<double>> bicg;
    bicg.setTolerance(tol1);
    bicg.setMaxIterations(maxiter1);
    bicg.compute(A1);

    x1 = bicg.solve(b1);
    std::cout << " Eigen native BICG" << endl;
    std::cout << "#iterations:     " << bicg.iterations() << endl;
    std::cout << "tolerance achived: " << bicg.error()<< endl;
   
    
    
    return 0;
}