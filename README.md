# AI_Project_CPSC_4370
For AI Project CPSC 4370 tracking and documentation

## Description
CPSC 4370 Fraud Analysis AI Project

This project is for determining fraudulent transaction activity 
by using Naive Bayes and Random Forest classifier models for 
CPSC_4370_AI's end of the semester project

The referenced datasets being used for training and testing each model are train.csv and test.csv
These datasets may be found at https://www.kaggle.com/datasets/kartik2112/fraud-detection
All resources were discovered through research.
While the datasets are open source, all code has been written and committed by the students/collaborators 
for this project. For more information, refer to https://github.com/blockkaaron/AI_Project_CPSC_4370

## Project Objectives
**The Goal**: Detect the presence of fraudulent activities from a dataset.

Using Python and machine learning, develop a program that determines probabilities of fraudulent 
activities automatically using a predetermined dataset

## How To Run

>Ensure the datasets `train.csv` and `test.csv` are at the project root.
>Please give the program enough time to complete each step of both models (approximately 1 minute)

1. From the project root, create a virtual environment
    ```shell
    python3 -m venv venv
    ```

2. Activate the virtual environment
    ```shell
    source venv/bin/activate
    ```

3. Install the requirements
    ```shell
    pip install -r requirements.txt
    ```

4. Run the script
    ```shell
    python3 main.py
    ```
   
   There will be 4 pop-up results in this order:
   1. Pre-Data Prep
   2. Post-Data Prep
   3. Naive Bayes Confusion Matrix 
   4. Random Forest Confusion Matrix

5. Deactivate the environment when finished
   ```shell
   deactivate
   ```

>To progress to the next step/popup, close the current window.

The console will also print results based on progress of each model.

## Collaborators
- Linu Robin
- Stephanie Scherb
- Aaron Block

## More Info
- The datasets can be found at https://www.kaggle.com/datasets/kartik2112/fraud-detection
- Project source code: https://github.com/blockkaaron/AI_Project_CPSC_4370