#include "../third_party/Dense"
#include <iostream>

#define PR(x) (std::cout << x << std::endl);
#define BR() (std::cout << "---------" << std::endl);

int main() {
  Eigen::Vector3f t(1.0, 0.0, 3.0);
  Eigen::Matrix<float, 3, 3> A {
    {1, 0, 3}, 
    {4, 5, 6}, 
    {7, 8, 9}
  };
  Eigen::Matrix3f I = Eigen::Matrix3f::Identity();

  std::cout << t << std::endl;
  std::cout << A << std::endl;
  std::cout << I << std::endl;

  Eigen::Matrix<float, 4, 4> T;
  T << A, t,
       0, 0, 0, 1;
  std::cout << T << std::endl;

  Eigen::MatrixXf B = A.transpose();
  std::cout << B << std::endl;
  
  std::cout << "---------" << std::endl;

  t(1) = 2;
  A(0, 1) = 2;
  PR(t);
  BR();
  PR(A)
  BR();
  PR(T);
  BR();

  T << A, t,
       0, 0, 0, 1;
  PR(T);
  BR();

  Eigen::Matrix<float, 1, 3> r2 = A.block<1, 3>(1, 0);
  PR(r2);
  BR();

  Eigen::Matrix<float, 3, 1> c2 = A.block<3, 1>(0, 1);
  PR(c2);
  BR();

  Eigen::Matrix<float, 3, 4> T3x4 = T.block<3, 4>(0,0);
  PR(T3x4);
  BR();

  A.block<1, 3>(1, 0) = Eigen::Matrix<float, 1, 3>::Zero();
  A.block<3, 1>(0, 1) = Eigen::Matrix<float, 3, 1>::Zero();
  PR(A);
  BR();

  T.block<3, 4>(0, 0) = Eigen::Matrix<float, 3, 4>::Zero();
  PR(T);
  BR();

  Eigen::Vector<int, 2> a {0, 1};
  Eigen::Vector<int, 2> b {2, 3};
  PR(a+b);
  BR();

  PR(a.dot(b));

  Eigen::Matrix3f X {
    {0, 1, 0},
    {1, 0, 1},
    {0, 1, 0}
  };

  Eigen::Matrix3f Y {
    {3, 3, 3},
    {4, 0, 6},
    {7, 8, 9}
  };

  PR(X.array()*Y.array()); // coefficient-wise multiplication
  BR();

  PR(X.sum());
  BR();

  PR(Y.minCoeff());
  Eigen::Index minRow, minCol;
  float minY = Y.minCoeff(&minRow, &minCol); // stores location in minRow and minCol
  PR(minRow);
  PR(minCol);
  BR();
  Eigen::Vector3f minCoeffs = Y.colwise().minCoeff();
  PR(minCoeffs);
  PR(minCoeffs.lpNorm<2>());
  BR();
  std::cout << (Y.array() > 3.0f).count() << std::endl;
  return 0;
}