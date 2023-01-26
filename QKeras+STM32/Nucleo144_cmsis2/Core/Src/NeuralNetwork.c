#include "NeuralNetwork.h"
#define Max(a,b) (a>b?a:b)


// Minimum amount of stack: 400 byte (0x190)
void PredictFFNN(const q15_t* inputVector,const q15_t* weight1, const q15_t* weight2,const q15_t* weight3,const q15_t* bias1,const q15_t* bias2,const q15_t* bias3, float* outVector){
	
	q15_t interVector[LYR1_SIZE];
	q15_t interVector2[LYR2_SIZE];
	q15_t out[OUTPUT_SIZE];
	int offset = 32768;
	int i;
	
	//Fully connected layer, the last parameter isn't used by the function 
	arm_fully_connected_q15(inputVector,weight1,INPUT_SIZE,LYR1_SIZE,5,8,bias1,interVector,NULL);
	arm_relu_q15(interVector,LYR1_SIZE);
	
	arm_fully_connected_q15(interVector,weight2,LYR1_SIZE,LYR2_SIZE,3,7,bias2,interVector2,NULL);
	arm_relu_q15(interVector2,LYR2_SIZE);
	
	arm_fully_connected_q15(interVector2,weight3,LYR2_SIZE,OUTPUT_SIZE,2,5,bias3,out,NULL);
	arm_softmax_q15(out,OUTPUT_SIZE,out);
	
	// Surprisingly arm_softmax_q15 associate the value 32768 with 100% probability extimation, so the final
	// output must be divided by 32768
	for (i=0;i<OUTPUT_SIZE;i++){
		outVector[i] = out[i]; //outVector[i] = out[i]/offset doesn't work, the q15_t result from / (=0) is cast to float (=0.00000)
		outVector[i] = outVector[i]/offset;
		//printf("%f\n",outVector[i]);
	}

}


//Minimum amount of stack: 328 bytes
void PredictFFNN_q7(const q7_t* inputVector,const q7_t* weight1,const q7_t* weight2,const q7_t* weight3,const q7_t* bias1,const q7_t* bias2,const q7_t* bias3, float* outVector){
	
	
	q7_t interVector[LYR1_SIZE];
	q7_t interVector2[LYR2_SIZE];
	q7_t out[OUTPUT_SIZE];
	q15_t vec_buffer[LYR1_SIZE];
	
	int offset = 256;
	int i;
	
	// The last parameter is a buffer where input data is copied, size = INPUT_SIZE. In case of two
	//layers: size = max(input_layer1 ; input_layer2, input_layer3) = Max(INPUT_SIZE, LYR1_SIZE, LYR2_SIZE)
	arm_fully_connected_q7(inputVector,weight1,INPUT_SIZE,LYR1_SIZE,5,7,bias1,interVector,vec_buffer);
	arm_relu_q7(interVector,LYR1_SIZE);
	
	arm_fully_connected_q7(interVector,weight2,LYR1_SIZE,LYR2_SIZE,3,5,bias2,interVector2,vec_buffer);
	arm_relu_q7(interVector2,LYR2_SIZE);

	arm_fully_connected_q7(interVector2,weight3,LYR2_SIZE,OUTPUT_SIZE,3,5,bias3,out,vec_buffer);
	arm_softmax_q7(out,OUTPUT_SIZE,out);
	// Surprisingly arm_softmax_q15 associate the value 32768 with 100% probability extimation, so the final
	// output must be divided by 256
	for (i=0;i<OUTPUT_SIZE;i++){
		outVector[i] = out[i]; //outVector[i] = out[i]/offset doesn't work, the q15_t result from / (=0) is cast to float (=0.00000)
		outVector[i] = outVector[i]/offset;
		//printf("%f\n",outVector[i]);
	}
	
}

/*
//If this function gets stuck, consider incrementing the stack min amount of stack: (LYR1_SIZE + LYR2_SIZE)*4 bytes
void PredictFFNN_dsp(const float32_t * inputVector,const float32_t * weight1,const float32_t * weight2,const float32_t * weight3,const float32_t * bias1,const float32_t * bias2,const float32_t * bias3, float32_t * outVector){
	
	arm_matrix_instance_f32 armWeights1,armWeights2,armWeights3;
	arm_matrix_instance_f32 armBias1, armBias2, armBias3, armInputVector, armOutputVector;
	arm_matrix_instance_f32 armIntermediateVector, armIntermediateVector2;
  float interVector[LYR1_SIZE];
	float interVector2[LYR2_SIZE];
	
	arm_mat_init_f32(&armInputVector, INPUT_SIZE, 1, inputVector);    	 //8x1
	arm_mat_init_f32(&armWeights1, LYR1_SIZE, INPUT_SIZE, weight1);  	   //128x8
	arm_mat_init_f32(&armWeights2, LYR2_SIZE, LYR1_SIZE, weight2);    	 //64x128
	arm_mat_init_f32(&armWeights3, OUTPUT_SIZE, LYR2_SIZE, weight3); 	   //8x64
	arm_mat_init_f32(&armBias1, LYR1_SIZE, 1, bias1);			  						 //128x1
	arm_mat_init_f32(&armBias2, LYR2_SIZE, 1, bias2);								 	   //64x1
	arm_mat_init_f32(&armBias3, OUTPUT_SIZE, 1, bias3); 								 //8x1
	arm_mat_init_f32(&armIntermediateVector, LYR1_SIZE, 1, interVector); //128x1
	arm_mat_init_f32(&armIntermediateVector2, LYR2_SIZE, 1, interVector2); //64x1
	arm_mat_init_f32(&armOutputVector,OUTPUT_SIZE, 1, outVector); 				//8x1
	
	//int size_buffer = Max(LYR1_SIZE*INPUT_SIZE, OUTPUT_SIZE*LYR1_SIZE);
	//q15_t interm_buffer[size_buffer];
	
	FC_layer(&armInputVector, &armWeights1, &armBias1, &armIntermediateVector, NULL);
	ReLU_layer(&armIntermediateVector, LYR1_SIZE);
	FC_layer(&armIntermediateVector, &armWeights2, &armBias2, &armIntermediateVector2, NULL);
	ReLU_layer(&armIntermediateVector2, LYR2_SIZE);
	FC_layer(&armIntermediateVector2, &armWeights3, &armBias3, &armOutputVector, NULL);
	softmax_layer(&armOutputVector);
}


void FC_layer(arm_matrix_instance_f32* input,arm_matrix_instance_f32* weight, arm_matrix_instance_f32* bias, arm_matrix_instance_f32* output, q15_t* interm_buffer){
	  (void)interm_buffer;
		// Multiply input with weights 
    arm_mat_mult_f32(weight, input, output);
    // Add bias 
    arm_mat_add_f32(output, bias, output);
	
}

void ReLU_layer(arm_matrix_instance_f32* output, int size){
	int i;
	for(i=0; i<size; i++)
		*((*output).pData+i) = fmaxf(*((*output).pData+i),0);
}

void softmax_layer(arm_matrix_instance_f32* output){
	int i;
	uint32_t maxActivationIdx;
  float offset, sumActivationOutputs = 0, maxActivation = 0;
	
	arm_max_f32((*output).pData, OUTPUT_SIZE, &maxActivation, &maxActivationIdx);

	// Sum of exponentials
	for(i=0; i<OUTPUT_SIZE; i++)
			sumActivationOutputs += expf(*((*output).pData+i) - maxActivation);

	// Calculate softmax
	offset = maxActivation + logf(sumActivationOutputs);
	for(i=0; i<OUTPUT_SIZE; i++)
			*((*output).pData+i) = expf(*((*output).pData+i) - offset);
}
*/
