#include <mkl_blas.h>
#include <mkl_spblas.h>
#include "mkl.h"
// obtaining the maximum value in a double type vector
double max_value_vector(int num_vector, double * value_vector, int absolute_ornot)
{
	int i;
	double result;
	if (absolute_ornot == 0) {
		result = value_vector[0];
		for (i = 1; i<num_vector; i++) {
			if (result < value_vector[i]) {
				result = value_vector[i];
			}
		}
	}
	else if (absolute_ornot == 1) {
		result = fabs(value_vector[0]);
		for (i = 1; i<num_vector; i++) {
			if (result < fabs(value_vector[i])) {
				result = fabs(value_vector[i]);
			}
		}
	}
	else {
		cout << "Function max_value_vector : the third parameter is neither 0 nor 1, has been set as 0" << endl;
		result = value_vector[0];
		for (i = 1; i<num_vector; i++) {
			if (result < value_vector[i])
				result = value_vector[i];
		}
	}
	return result;
}


// obtaining the minimum value in a double type vector
double min_value_vector(int num_vector, double * value_vector, int absolute_ornot)
{
	double result;
	int i;
	if (absolute_ornot == 0) {
		result = value_vector[0];
		for (i = 1; i<num_vector; i++) {
			if (result > value_vector[i])
				result = value_vector[i];
		}
	}
	else if (absolute_ornot == 1) {
		result = fabs(value_vector[0]);
		for (i = 1; i<num_vector; i++) {
			if (result > fabs(value_vector[i]))
				result = fabs(value_vector[i]);
		}
	}
	else {
		cout << "Function min_value_vector : the third parameter is neither 0 nor 1, has been set as 0" << endl;
		result = value_vector[0];
		for (i = 1; i<num_vector; i++) {
			if (result > value_vector[i])
				result = value_vector[i];
		}
	}
	return result;
}


// some functions for Wavelet transform and compression (e.g., Li and Oldenburg, 2003)
// some functions for Wavelet transform and compression (e.g., Li and Oldenburg, 2003)
// some functions for Wavelet transform and compression (e.g., Li and Oldenburg, 2003)
// some functions for Wavelet transform and compression (e.g., Li and Oldenburg, 2003)
double cal_r_wavelet_19(double * G_temp_waveletTrans, int num_vector,double relative_thrhd)
{
	double max_G_t_w = max_value_vector(num_vector,G_temp_waveletTrans,1);
	double absolute_thrhd = max_G_t_w*relative_thrhd;
	int i,j;

	double ffzi = 0,ffmu = 0; 
	for(i=0;i<num_vector;i++){
		if(fabs(G_temp_waveletTrans[i]) < absolute_thrhd){
			ffzi = ffzi + G_temp_waveletTrans[i]*G_temp_waveletTrans[i];
		}
		ffmu = ffmu + G_temp_waveletTrans[i]*G_temp_waveletTrans[i];
	}

	double result;
	result = sqrt(ffzi/ffmu);

	return result;
}

double cal_relative_threshold_WaveletTransCompression(double * G_temp_waveletTrans, int num_vector, double abo_r)
{
	int i,j;
	double max_G_t_w = max_value_vector(num_vector,G_temp_waveletTrans,1);
	double k_min,k_max;
	double k_err_stop;
	k_min = 0.000001;
	k_max = 0.1;
	k_err_stop = 0.00001;
	
	double result;

	double r_min,r_max;
	r_min = cal_r_wavelet_19(G_temp_waveletTrans, num_vector, k_min);
	r_max = cal_r_wavelet_19(G_temp_waveletTrans, num_vector, k_max);

	if(fabs(r_min-abo_r) <= 0.001){
		result = k_min;
		return result;
	}

	if(fabs(r_max-abo_r) <= 0.001){
		result = k_max;
		return result;
	}

	while(r_min >= abo_r){
		k_max = k_min;
		k_min = k_min/2;
		r_min = cal_r_wavelet_19(G_temp_waveletTrans, num_vector, k_min);
		if(fabs(r_min-abo_r) <= 0.001){
			result = k_min;
			return result;
		}
	}

	while(r_max <= abo_r){
		k_min = k_max;
		k_max = k_max*2;
		r_max = cal_r_wavelet_19(G_temp_waveletTrans, num_vector, k_max);
		if(fabs(r_max-abo_r) <= 0.001){
			result = k_max;
			return result;
		}
	}

	double k_min_max_2;
	double r_min_max_2;
	while(k_max - k_min > k_err_stop){
		k_min_max_2 = (k_max+k_min)/2;
		r_min_max_2 = cal_r_wavelet_19(G_temp_waveletTrans, num_vector, k_min_max_2);

		if(fabs(r_min_max_2-abo_r) <= 0.001){
			result = k_min_max_2;
			return result;
		}
		else{
			if(r_min_max_2 > abo_r){
				k_max = k_min_max_2;
			}
			else{
				k_min = k_min_max_2;
			}
		}
	}

	k_min_max_2 = (k_max+k_min)/2;

	result = k_min_max_2;

	return result;

}


// Recovering a vector in one row of a sparse matrix
int convertSparseMtx2Vector(double * G_T_wavelet_value,MKL_INT * columns_deltaT,MKL_INT * rowIndex_deltaT, int Nmod_waveletFull, int row_num, double * vector_extract)
{
	int i,j;
	if(row_num > Nmod_waveletFull){
		cout<<endl<<endl<<"-------------CAUTION!!!!!!"<<endl;
		cout<<"convertSparseMtx2Vector function : the number of the recovered row is larger than the maximum row of this matrix"<<endl;
		return 1;
	}
	for(i=0; i<Nmod_waveletFull; i++)
		vector_extract[i] = 0;

	int colunm_index_temp;
	for(i=rowIndex_deltaT[row_num]-1; i<rowIndex_deltaT[row_num+1]-1; i++) {
		colunm_index_temp = columns_deltaT[i] - 1;
		vector_extract[colunm_index_temp] = G_T_wavelet_value[i];
	}
	return 0;
}

void vector2Dyadic(double * Data, int length_a, int length_b, double * D_2full)
{
	int i;
	#pragma omp parallel for shared(length_a) private(i)
	for(i=0;i<length_a;i++)
		D_2full[i] = Data[i];
	
	#pragma omp parallel for shared(length_a) private(i)
	for(i=length_a;i<length_b;i++)
		D_2full[i] = 0;

	return;
}

void vector2Dyadic_inv(double * Data, int length_a, int length_b, double * D_2full)
{
	int i;
	#pragma omp parallel for shared(length_a) private(i)
	for(i=0;i<length_a;i++)
		Data[i] = D_2full[i];

	return;
}