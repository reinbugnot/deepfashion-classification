# Using Synthetic Image Generation for Maximizing the Performance of a Multi-label, Multi-class Image Classification Model

## Description

This is the final group project for the AI6103: Deep Learning and Applications class in Nanyang Technological University, Singapore, 2024. Here, we performed a multi-label, multi-class classification task using the [DeepFashion Dataset](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html) by fine-tuning a pre-trained ResNeXt model. To optimize the model's performance, we employed Optuna, a hyper-parameter tuning framework, to identify the best combination of various hyper-parameters. In addition to traditional data augmentation techniques, we devised an innovative method to synthesize new images: utilizing a diffusion model trained on the given dataset to generate additional images for underrepresented classes, thereby balancing the dataset. Our experimental results demonstrated that this approach was effective in improving model performance.

## Getting Started

### Prerequisites

Make sure you have the following software installed:
- Python 3.x
- git
- Other dependencies listed in `requirements.txt`

### Cloning the Repository

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

Create the following folders in the project directory:
```
mkdir results
mkdir diagram
```

### Download the Dataset
Download the DeepFashion dataset for attribute prediction from the [official website](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html). 

Once downloaded, place the FashionDataset/ folder in the root directory of the project.

### Installing Dependencies

Install the required libraries using requirements.txt:
```
pip install -r requirements.txt
```

## Running the Project

You can run the training script using SLURM or directly with Python:

• Using SLURM:
Run the provided train_small.sh script or customize your own bash script.
```
sbatch train_small.sh
```

• Without SLURM:
Run the main Python script directly via CLI.
```python
python main.py \
--dataset_dir './FashionDataset/' \
--seed 0 \
--batch_size 256 \
--epochs 80 \
--lr_scheduler \
--smoothing 0.00931113078764147 \
--dropout_p 0.5 \
--lr 0.00945284750884842 --wd 0.000887541785266803 \
--fig_name sample.png \
--test \
--tuning_optuna --n_trials 15 --trial_epochs 30 \
```

### Acknowldegements

This project is a team-effort with my awesome colleagues from the AI6103 - Deep Learning and Applications class in the NTU MSAI program, 2024. Checkout their github profiles at:

- [Syed Anas Majid](https://github.com/modestscriptor)
- [Xiang Xinye](https://github.com/Sherlock-Watson)

If you wish to cite this project, please use the following format:

```plaintext
@project{YourProject2024,
  author = {Reinelle Jan Bugnot, Syed Anas Majid, and Xiang Xinye},
  title = {Using Synthetic Image Generation for Maximizing the Performance of a Multi-label, Multi-class Image Classification Model},
  year = {2024},
  url = {https://github.com/reinbugnot/deepfashion-classification}
}
```



