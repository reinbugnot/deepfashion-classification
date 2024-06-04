# Project Title

## Description

Brief description of your project and what it aims to achieve.

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
Download the DeepFashion dataset for attribute prediction from the ![official website](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/AttributePrediction.html). Once downloaded, place the FashionDataset/ folder in the root directory of the project.

### Installing Dependencies

Install the required libraries using requirements.txt:
```
pip install -r requirements.txt
```

### Running the Project

You can run the training script using SLURM or directly with Python:

• Using SLURM:
Run the provided train_large.sh script or write your own bash script.
```
sbatch train_large.sh
```

• Without SLURM:
Run the main Python script directly.
```
python main.py
```

### Contributing

We welcome contributions to this project. To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes.
4. Commit your changes (git commit -m 'Add some feature').
5. Push to the branch (git push origin feature-branch).
6. Create a new Pull Request.
7. Please make sure your code follows our coding standards and includes tests for any new features.

### Acknowldegements

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



