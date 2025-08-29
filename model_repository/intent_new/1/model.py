import json
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import pipeline
import torch

class TritonPythonModel:
    def initialize(self, args):
        """Initialize the model."""
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get output configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "label")
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "score")
        
        # Check if configurations are found
        if output0_config is None:
            raise ValueError("Output configuration for 'label' not found in model config")
        if output1_config is None:
            raise ValueError("Output configuration for 'score' not found in model config")
        
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config['data_type'])
        
        # Initialize the Hugging Face pipeline
        try:
            self.classifier = pipeline(
                "text-classification", 
                model="dinesh-001/intent_classification_v2",
                return_all_scores=True  # This will return all class probabilities
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def execute(self, requests):
        """Execute inference on the batch of requests."""
        responses = []
        
        for request in requests:
            # Get input text
            input_tensor = pb_utils.get_input_tensor_by_name(request, "text_input")
            input_text = input_tensor.as_numpy()
            
            # Convert bytes to string if necessary
            if input_text.dtype == np.object_:
                texts = [text.decode('utf-8') if isinstance(text, bytes) else text for text in input_text.flatten()]
            else:
                texts = input_text.flatten().tolist()
            
            # Run inference
            try:
                predictions = []
                scores = []
                
                for text in texts:
                    result = self.classifier(text)
                    
                    # Handle different result formats
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], list):
                            # If return_all_scores=True, result is [[{label, score}, ...]]
                            all_scores = result[0]
                        else:
                            # If return_all_scores=False, result is [{label, score}]
                            all_scores = result
                        
                        # Get the prediction with highest score
                        best_pred = max(all_scores, key=lambda x: x['score'])
                        predictions.append(best_pred['label'])
                        scores.append(best_pred['score'])
                    else:
                        raise ValueError(f"Unexpected classifier result format: {result}")
                
                # Create output tensors
                label_tensor = pb_utils.Tensor("label", 
                    np.array(predictions, dtype=self.output0_dtype).reshape(-1, 1))
                score_tensor = pb_utils.Tensor("score", 
                    np.array(scores, dtype=self.output1_dtype).reshape(-1, 1))
                
                # Create inference response
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[label_tensor, score_tensor])
                
            except Exception as e:
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"Inference failed: {str(e)}"))
            
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        """Clean up resources."""