#include "RTE_Components.h"
#include  CMSIS_device_header
#include "main.h"

namespace
{
    tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* model_input = nullptr;
    TfLiteTensor* model_output = nullptr;

    // Create an area of memory to use for input, output, and intermediate arrays.
    // Finding the minimum value for your model may require some trial and error.
    constexpr uint32_t kTensorArenaSize = 2*1024; 
    uint8_t tensor_arena[kTensorArenaSize];
} // namespace


float_t in_data[TEST_SIZE] = INPUT_V; // Input data for inference
int8_t labels[TEST_SIZE] = Y_TEST;


int main(void)
{
  /* USER CODE BEGIN 1 */
	int8_t y_test[N_FEATURES];
	volatile float accuracy = 0; //Those variables are set as volatile to see their values from the debug
  

  /* Configure the system clock */
  //SystemClock_Config();
	

  /* USER CODE BEGIN 3 */
	static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
	
  model = tflite::GetModel(FFNN_model);
	
	// This pulls in all the operation implementations we need.
  static tflite::AllOpsResolver resolver; //After including this, the memory footprint explodes
	
	// Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;
	
	// Allocate memory from the tensor_arena for the model's tensors.
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk)
	{
			TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
			return 0;
	}
	// Obtain pointers to the model's input tensors.
  model_input = interpreter->input(0);
	
	
		// Run 10 inferences 
		for(int i=0; i<TEST_SIZE; i=i+8){
			
			std::copy(in_data+i, in_data+i+N_FEATURES, model_input->data.f);
			//std::copy(in_data, in_data+N_FEATURES, model_input->data.f);
			
			// Run inference, and report any error
			TfLiteStatus invoke_status = interpreter->Invoke();
			
			model_output = interpreter->output(0);
			__NOP();
			std::copy(labels+i, labels+i+N_FEATURES, y_test);
			//std::copy(labels, labels+N_FEATURES, y_test);
			
			// If the prediction is correct 1 is summed, 0 otherwise
			//accuracy = accuracy + compare_output(y_test, model_output->data.f);
		}
	
	/*
		Tested with 100 samples, 82% accurate, equal to the python implementation
		Warning: accuracy must be tested with much more samples, if with 100 samples
		it's == python, the assumption is that's equal with 10000 samples (70%)
	*/
	//__NOP(); // nop operation for debugging
	accuracy = accuracy/N_INFERENCE;
	
	return 0;
  /* USER CODE END 3 */
}
