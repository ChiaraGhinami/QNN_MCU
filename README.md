# NNs quantization and deployment
 This project quantizes a NN using TFLite and QKeras, and deploys the NN on two MCUs. Tutorial.pdf is a document with a tutorial that covers the setup of uVision MDK for latency and energy measurements.   
 In the folder *QKeras+STM32* there are two folders:  
 1. *qkeras_quant*, with the python code for the QKeras training, quantization and parameters manipulation  
 2. *Nucleo144_cmsis*, with the uVision project, targeting the NUCLEO-H7A3ZI board  
 
 The folder *TFLite+VHT* contains:  
 1. *tflite_quant*, with the python code for the Keras training, quantization and TensorFlow Lite model conversion  
 2. *VHT-tflite*, with the cpp code to run the inference, targeting the SSE-300 subsystem (virtual hardware target), which leverages the Cortex-M55 ARM processor
 
