#ifndef NEURALNETWORK_H__
#define NEURALNETWORK_H__

#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#define INPUT_SIZE 8
#define LYR1_SIZE 128
#define LYR2_SIZE 64
#define OUTPUT_SIZE 8

q7_t PredictFFNN(const q15_t* inputVector,const q15_t* weight1,const q15_t* weight2,const q15_t* weight3, const q15_t* bias1,const q15_t* bias2,const q15_t* bias3, float* outVector);
q7_t PredictFFNN_q7(const q7_t* inputVector,const q7_t* weight1,const q7_t* weight2,const q7_t* weight3,const q7_t* bias1,const q7_t* bias2,const q7_t* bias3, float* outVector);

q7_t max_index(q7_t *a, int n);
q7_t max_index_q15(q15_t *a, int n);

#endif