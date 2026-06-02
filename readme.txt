## Installation

	Install all required dependencies:
	pip install -r requirements.txt


## Dataset Setup

	Download the dataset from:
	https://drive.google.com/file/d/1GIuiI13HF_CltKTRyfWljQvD88NcgB2M/view?usp=sharing

	Extract it inside the "code" folder with the name:
	rescuenet/


## Usage

	• presentation.ipynb  
	  View overall workflow, inference results, and comparisons:
	  - Base model vs our model  
	  - Our model vs ground truth (GT)

	• inference.ipynb  
	  Run inference on your own images (upload supported)

	• inferenceTime.ipynb  
  	  Compare inference time of both models:
	  - Measures latency for base model vs proposed model  
	  - Helps analyze speed vs performance trade-offs 

	• training_v9.ipynb, training_segmenter.ipynb  
	  Used for training the models  

	Note: Run all cells in the .ipynb files sequentially to see complete results.


## Core Files

	• dataset.py  
	  Handles data loading, preprocessing, and augmentation  

	• model.py  
	  Defines and integrates the models  


## Models

	• segmenter/  
	  Contains the base model and related implementation  


## Training Artifacts

	• training_log_V13.txt  
	  Contains training logs and metrics  

	• LossGraph.png  
	  Visualizes training loss over epochs  

	• training_weights384.npy  
	  Used for dataset sampling, especially for handling rare classes  


## Outputs

	• outputs/  
	  Stores predictions and results  

	• uploads/  
	  Used for input images during inference