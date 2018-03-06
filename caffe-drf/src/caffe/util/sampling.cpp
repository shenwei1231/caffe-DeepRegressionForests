#include <float.h>
#include "caffe/util/sampling.hpp"
#include "caffe/util/math_functions.hpp"
#include "Eigen/Eigen"

using namespace Eigen;

#ifndef PI 
#define PI 3.1415926
#endif

#define swapInt(a, b) ((a ^= b), (b ^= a), (a ^= b))

namespace caffe {


template <>
void RandSample<float>(int num_samples, int num_sub_samples, float* sample_index_vec)
{
	int* nind = new int[num_samples];
	for (int i = 0; i < num_samples; ++i)
	{
		nind[i] = i;
	}
	int last = num_samples - 1;
	for (int i = 0; i < num_sub_samples; ++i)
	{
		int ktmp = caffe_rng_rand() % (last + 1);
		int k = nind[ktmp];
		swapInt(nind[ktmp], nind[last]);
        last--;
        sample_index_vec[i] = k;
	}
	delete [] nind;
}

/*
template <>
void RandSample<int>(int num_samples, int num_sub_samples, int* sample_index_vec)
{
	int* nind = new int[num_samples];
	for (int i = 0; i < num_samples; ++i)
	{
		nind[i] = i;
	}
	int last = num_samples - 1;
	for (int i = 0; i < num_sub_samples; ++i)
	{
		int ktmp = caffe_rng_rand() % (last + 1);
		int k = nind[ktmp];
		swapInt(nind[ktmp], nind[last]);
        last--;
        sample_index_vec[i] = k;
	}
	delete [] nind;
}
*/

template <>
void RandSample<double>(int num_samples, int num_sub_samples, double* sample_index_vec)
{
	int* nind = new int[num_samples];
	for (int i = 0; i < num_samples; ++i)
	{
		nind[i] = i;
	}
	int last = num_samples - 1;
	for (int i = 0; i < num_sub_samples; ++i)
	{
		int ktmp = caffe_rng_rand() % (last + 1);
		int k = nind[ktmp];
		swapInt(nind[ktmp], nind[last]);
        last--;
        sample_index_vec[i] = k;
	}
	delete [] nind;
}

template <>
float multivariate_gaussian<float>(const float* x, const float* mu, const float* sigma_square, int dim)
{
	
	if (dim == 1)
	{
    	return (float)1.0 / sqrt(2 * PI * (*sigma_square + FLT_MIN)) * (exp(-(*x - *mu) * (*x - *mu) / (2 * (*sigma_square + FLT_MIN)))+ FLT_MIN);
    	//return (float)1.0 / sqrt(2 * PI * (*sigma_square + FLT_MIN)) * exp(-(*x - *mu) * (*x - *mu) / (2 * (*sigma_square + FLT_MIN)));
	}
	else
	{
		VectorXf X(dim);

		VectorXf Mu(dim);

		MatrixXf Sigma_square(dim, dim);
		for (int i = 0; i < dim; ++i)
		{
			X(i) = x[i];
			Mu(i) = mu[i];
			for (int j = i; j < dim; ++j)
			{
				Sigma_square(i, j) = sigma_square[i * dim + j];
				Sigma_square(j, i) = Sigma_square(i, j);
			}
			
		}
		X = X - Mu;
		RowVectorXf Z(dim);
		Z = X.transpose() * Sigma_square.inverse();
		Z = Z * X;
    //    LOG(INFO)<<X<<":"<<X(0)<<","<<X(1);
     //   LOG(INFO)<<Sigma_square;
        //LOG(INFO)<<Z(0);
        float temp = sqrt(pow(2 * PI, dim) * (float) Sigma_square.determinant());
	    //LOG(INFO)<<(float) 1.0 / (temp> FLT_MIN? temp: FLT_MIN) <<","<<exp(-0.5 * Z(0));
        //LOG(INFO)<<(exp(-0.5 * Z(0))<FLT_MAX?exp(-0.5 * Z(0)):FLT_MAX);
	    return (float) (1.0 / (temp> FLT_MIN? temp: FLT_MIN) * exp(-0.5 * Z(0)))<FLT_MAX?(1.0 / (temp> FLT_MIN? temp: FLT_MIN) * exp(-0.5 * Z(0))):FLT_MAX;
		//return (float) 1.0 / sqrt(pow(2 * PI, dim) * ((float) Sigma_square.determinant() + FLT_MIN)) * exp(-0.5 * Z(0));
	}
	
}


template <>
double multivariate_gaussian<double>(const double* x, const double* mu, const double* sigma_square, int dim)
{
	if (dim == 1)
	{
		
    	return (double)1.0 / sqrt(2 * PI * (*sigma_square + DBL_MIN)) * (exp(-(*x - *mu) * (*x - *mu) / (2 * (*sigma_square + DBL_MIN)))+ DBL_MIN);
    	//return (double)1.0 / sqrt(2 * PI * (*sigma_square + DBL_MIN)) * exp(-(*x - *mu) * (*x - *mu) / (2 * (*sigma_square + DBL_MIN)));
	}
	else
	{
		VectorXd X(dim);

		VectorXd Mu(dim);

		MatrixXd Sigma_square(dim, dim);
		for (int i = 0; i < dim; ++i)
		{
			X(i) = x[i];
			Mu(i) = mu[i];
			for (int j = i; j < dim; ++j)
			{
				Sigma_square(i, j) = sigma_square[i * dim + j];
				Sigma_square(j, i) = Sigma_square(i, j);
			}
			
		}
		X = X - Mu;
		RowVectorXd Z(dim);
		Z = X.transpose() * Sigma_square.inverse();
		Z = Z * X;
		return (double) 1.0 / sqrt(pow(2 * PI, dim) * ((double) Sigma_square.determinant() + DBL_MIN)) * exp(-0.5 * Z(0));
	}
	
}

}
