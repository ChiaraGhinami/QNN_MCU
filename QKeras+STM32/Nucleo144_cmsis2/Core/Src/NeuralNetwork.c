#include "NeuralNetwork.h"
#define Max(a,b) (a>b?a:b)


q7_t PredictFFNN(const q15_t* inputVector,const q15_t* weight1, const q15_t* weight2,const q15_t* weight3,const q15_t* bias1,const q15_t* bias2,const q15_t* bias3, float* outVector){
	//volatile uint32_t clk_cnt = 0;
	q15_t interVector[LYR1_SIZE];
	q15_t interVector2[LYR2_SIZE];
	q15_t out[OUTPUT_SIZE];
	int offset = 32768;
	q7_t i, idx;
	
	//Fully connected layer, the last parameter isn't used by the function 
	arm_fully_connected_q15(inputVector,weight1,INPUT_SIZE,LYR1_SIZE,9,8,bias1,interVector,NULL);
	arm_relu_q15(interVector,LYR1_SIZE);
	
	arm_fully_connected_q15(interVector,weight2,LYR1_SIZE,LYR2_SIZE,7,7,bias2,interVector2,NULL);
	arm_relu_q15(interVector2,LYR2_SIZE);
	
	arm_fully_connected_q15(interVector2,weight3,LYR2_SIZE,OUTPUT_SIZE,7,7,bias3,out,NULL);
	arm_softmax_q15(out,OUTPUT_SIZE,out);
	idx =  max_index_q15(out, OUTPUT_SIZE);
	/*
	// Surprisingly arm_softmax_q15 associates the value 32768 with 100% probability extimation, so the final
	// output must be divided by 32768
	for (i=0;i<OUTPUT_SIZE;i++){
		outVector[i] = out[i]; //outVector[i] = out[i]/offset doesn't work, the q15_t result from / (=0) is cast to float (=0.00000)
		outVector[i] = outVector[i]/offset;
		//printf("%f\n",outVector[i]);
	}
	*/
	return idx;
}


//Minimum amount of stack: 328 bytes
q7_t PredictFFNN_q7(const q7_t* inputVector,const q7_t* weight1,const q7_t* weight2,const q7_t* weight3,const q7_t* bias1,const q7_t* bias2,const q7_t* bias3, float* outVector){
	
	//volatile uint32_t clk_cnt = 0;
	q7_t interVector[LYR1_SIZE];
	q7_t interVector2[LYR2_SIZE];
	q7_t out[OUTPUT_SIZE];
	q15_t vec_buffer[LYR1_SIZE];
	
	int offset = 256;
	q7_t i, idx;
	
	// The last parameter is a buffer where input data is copied, size = INPUT_SIZE. In case of two
	//layers: size = max(input_layer1 ; input_layer2, input_layer3) = Max(INPUT_SIZE, LYR1_SIZE, LYR2_SIZE)
	arm_fully_connected_q7(inputVector,weight1,INPUT_SIZE,LYR1_SIZE,7,7,bias1,interVector,vec_buffer);
	arm_relu_q7(interVector,LYR1_SIZE);
	
	arm_fully_connected_q7(interVector,weight2,LYR1_SIZE,LYR2_SIZE,6,6,bias2,interVector2,vec_buffer);
	arm_relu_q7(interVector2,LYR2_SIZE);
	
	arm_fully_connected_q7(interVector2,weight3,LYR2_SIZE,OUTPUT_SIZE,6,8,bias3,out,vec_buffer);
	arm_softmax_q7(out,OUTPUT_SIZE,out);
	idx = max_index(out, OUTPUT_SIZE);
	
	/*
	// Surprisingly arm_softmax_q15 associates the value 32768 with 100% probability extimation, so the final
	// output must be divided by 256
	for (i=0;i<OUTPUT_SIZE;i++){
		outVector[i] = out[i]; //outVector[i] = out[i]/offset doesn't work, the q15_t result from / (=0) is cast to float (=0.00000)
		outVector[i] = outVector[i]/offset;
		//printf("%f\n",outVector[i]);
	}*/
	return idx;
}

q7_t max_index(q7_t *a, int n){
	
		if(n <= 0) return -1;
		q7_t i, max_i = 0;
		q7_t max = a[0];
	
		for(i = 1; i < n; ++i){
			if(a[i] > max){
				max = a[i];
				max_i = i;
			}
		}
		return max_i;
}

q7_t max_index_q15(q15_t *a, int n){
	
		if(n <= 0) return -1;
		q7_t i, max_i = 0;
		q15_t max = a[0];
	
		for(i = 1; i < n; ++i){
			if(a[i] > max){
				max = a[i];
				max_i = i;
			}
		}
		return max_i;
}

