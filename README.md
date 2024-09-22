# Neural Image Correction Using OCR-Based Loss Function

This is a research project that utilizes neural networks to provide images tailored for easier viewing by individuals with refractive errors.


![image](https://github.com/TorlenBuffet/image_correction/assets/138644618/19210d3a-996c-44fe-a54d-1a82f5e69a70)



The networks used for the image correction model include SRCNN, VDSR, and UNet, etc. 

You can use different networks depending on the specifications of your PC, but among the three, VDSR has the deepest layers and the highest accuracy.

The newly designed loss function employs the levenshtein distance, which measures the degree of agreement between the output text and the correct text when an image-corrected visual reproduction image is input to easyOCR, and is combined with MSE.

For any text image, the text readability is improved over conventional loss functions such as PSNR, MSE, SSIM, and L1.

## Installation

### Environment
To easily install the required dependencies, you can use the commands below:
```bash
pip install -r requirements.txt
```

## Usage
### Train/use your own datasets
Here is an example of a train.py script that uses Python's argparse module to parse the given command line arguments. This script shows how to set up training for a model based on the given arguments.

```bash
python train.py  --datasets ./path/yourimage/　--annotation_path path/your_image_list.txt --max_epochs 500 --gpus 1 --batch_size 10 --loss l1 --model swinir --sphere 0.8 --cylinder 0 --axis 0 --radius 1.5 --lr 0.00002 --img_shape 3 256 256
```
Note1: Image data must be prepared as datasets. To use the prepared dataset, **datasets_path** and **annotation_path** must be set in **datasets.py**. 
**annotation_path** refers to a text file containing the relative paths of the images that make up the dataset.

Note2: To display help information for command line arguments, use the -h or --help option in your Python script; for the train.py script, the correct command is：
```bash
python train.py -h
```
or
```bash
python train.py -help
```

### Evaluate with text image data

your trained model is evaluated in **val.py**.
Need to specify checkpoint_path and create test data path and annotation text.The evaluation is based on the average value of MSE, PSNR, and Levenshtein distance. 

MSE (Mean Square Error) represents the error between images, with smaller values indicating better performance. 

PSNR (Peak Signal to Noise Ratio) is a measure of image quality and is mainly used to evaluate the quality of compressed images.The higher this value is, the better the image quality is. 

The Levenshtein distance represents the error in the text that can be read from the image; the smaller this value, the better the reading accuracy.

| | MSE | PSNR | Levenshtein Distance |
|-----------|-----------|-----------|-----------|
| MSE | 0.0243 | 16.15 | 57.23 |
| PSNR | 0.0235 | 18.21 | 55.24 |
| MSE+OCR | 0.0431 | 12.46 | 24.54 |

The horizontal axis represents the evaluation metric and the vertical axis represents the Loss function used.

OCR-based MSE improves legibility of text images.
On the other hand, lower MSE and PSNR values than when using other loss functions suggest increased pixel-level errors and noise in the overall image; OCR-based MSE may be more concerned with character recognition than overall image quality.

The following images were visually reproduced after being input into an image correction model trained with different loss functions.



![image](https://github.com/Kawano-Lab/Image-Correction/assets/138644618/6a299c78-1b8b-495e-bd5f-4a142173312f)


Models trained with OCR-based loss functions have the advantage that character recognition is easier, even for character images with formats, alignments, and fonts not used in training.

![image](https://github.com/Kawano-Lab/Image-Correction/assets/138644618/5fa8bc96-38ab-4807-8d84-b8e6d0b7b70e)



## Demonstration

To input an actual image and display the corrected and corrected visual reproduction images, you can run **actual_eval.py**.

## Postscript

**DRL.py** is code for training an image correction model by applying deep reinforcement learning. However, this differs from the traditional reinforcement learning approach in that it treats the weights of the model as actions. This method is not recommended for regular use as it is very time consuming to train.

**create_datasets.py** is code to generate an image with a list of characters. By reducing the character size step by step, model learning using the loss function with OCR can proceed smoothly.

**data_list.py** is the code to create annotation text.

**test_blur.py** is the code to simulate the visual appearance of an image that reproduces the vision of a person with refractive error. **actual_eval.py** can be used for the same purpose.

## Considerations

### Loss Function Optimization
The OCR-based loss function used in this study is represented by the following formula, where $α=1.0$ and $β=1.0$ are set as default values:

![Loss](https://github.com/Kawano-Lab/Image-Correction/assets/138644618/4299fb00-7bcf-4634-886c-32c40c7671c2)


Here, OCR_loss is defined as:

![OCR_Loss_math](https://github.com/Kawano-Lab/Image-Correction/assets/138644618/b271cf05-79cd-4692-8c48-ae45d92096ff)

​
 
**original_text** is the text outputted by processing the original image through OCR. 
**crr_blur_text** is the text obtained by processing the corrected visual representation image through OCR.

Using fine-tuning tools like Optuna to determine the optimal values of $α$ and $β$ is expected to further improve accuracy.

**optim_loss.py** is used for optimizing $α$ and $β$

### Network Architecture
While VDSR was used in this study, applying state-of-the-art super-resolution models could potentially improve accuracy.

### Visual Reproduction Model
The visual information reproduction using mathematical models has limitations. 

To enhance the accuracy of the visual reproduction model, training deep learning models using pairs of original images and their visually reproduced counterparts obtained through actual lenses is anticipated to enhance the reproduction fidelity.
