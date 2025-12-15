# fashion-compatibility-prediction
Siamese networks for fashion compatibility - Course Project
# Fashion Compatibility Prediction - Course Project

## Team Members
- Jeannine Romero (jromer37@ucsc.edu)

## Project Description
Siamese neural network architecture for fashion compatibility prediction using the Maryland Polyvore dataset.

## Model Weights
**Google Drive Link:**

## Requirements
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pillow>=9.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

## Installation
```bash
pip install torch torchvision numpy pillow scikit-learn matplotlib tqdm
```

## Dataset
Maryland Polyvore dataset (33,375 outfits)
- Download from Kaggle: https://www.kaggle.com/datasets/dnepozitek/maryland-polyvore-images
- Place in: `/path/to/dataset/images/`



## Running the Code

### 1. Pre-compute Color Features (optional, for Siamese+Color model)
```bash
python color_features.py --data_dir /path/to/dataset/
```

### 2. Train Models
```bash
# Baseline
python train.py --model baseline --epochs 10 --batch_size 32

# Siamese
python train.py --model siamese --epochs 10 --batch_size 32

# Siamese + Color
python train.py --model siamese_color --epochs 10 --batch_size 32
```

### 3. Evaluate
```bash
python evaluate.py --model_path /path/to/model.pth
```

## Results
- Baseline: 50.48% accuracy
- Siamese: 54.07% accuracy (7.1% relative improvement)
- Siamese+Color: 53.40% accuracy

## Trained Model Weights
Download from: [[Google Drive Link =(https://drive.google.com/drive/folders/1cpMSupomf2r1tWMoYqYVCRRDtcxl0Hyk?usp=drive_link)

Place in: `./models/` directory

## Hardware Requirements
- GPU recommended (NVIDIA A100 used in experiments)
- Training time: ~12 minutes per model on A100
- RAM: 16GB minimum

## Citation
If you use this code, please cite:
```
@article{romero2024fashion,
  title={Siamese Neural Networks for Fashion Compatibility Prediction},
  author={Romero, Jeannine},
  year={2024}
}
```

## Contact
For questions: jromer37@ucsc.edu
