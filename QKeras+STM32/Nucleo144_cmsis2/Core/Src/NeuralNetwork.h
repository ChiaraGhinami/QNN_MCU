#ifndef NEURALNETWORK_H__
#define NEURALNETWORK_H__

#include "arm_math.h"
#include <stdio.h>
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

#define INPUT_SIZE 8
#define LYR1_SIZE 128
#define LYR2_SIZE 64
#define OUTPUT_SIZE 8

void PredictFFNN(const q15_t* inputVector,const q15_t* weight1,const q15_t* weight2,const q15_t* weight3, const q15_t* bias1,const q15_t* bias2,const q15_t* bias3, float* outVector);
void PredictFFNN_q7(const q7_t* inputVector,const q7_t* weight1,const q7_t* weight2,const q7_t* weight3,const q7_t* bias1,const q7_t* bias2,const q7_t* bias3, float* outVector);

void PredictFFNN_dsp(const float32_t * inputVector,const float32_t * weight1,const float32_t * weight2,const float32_t * weight3,const float32_t * bias1,const float32_t * bias2,const float32_t * bias3, float32_t * outVector);
void FC_layer(arm_matrix_instance_f32* input,arm_matrix_instance_f32* weight, arm_matrix_instance_f32* bias, arm_matrix_instance_f32* output, q15_t* interm_buffer);
void ReLU_layer(arm_matrix_instance_f32* output, int size);
void softmax_layer(arm_matrix_instance_f32* output);

#endif