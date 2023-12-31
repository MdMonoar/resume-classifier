# Resume Classification model
In this project:
1. a machine learning model was developed to categorize given resumes.
2. a python script was developed to: run on command line, use the model to categorize all resumes (in .pdf format) from a <given_path>, and move them in their respective folders (inside the <given_path> and named as the category)

## About the dataset:
The dataset was collected from a kaggle repository. The link is [here](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)

## Keypoints of Building the model:
1. The dataset contains a total of 2484 entries, in text and html format.
2. The dataset has 24 different categories.
3. Text was preprocessed with the help of NLTK library.
4. All the categories do not have same number of entries. So the data was stratified based on the 'Category' column while splitting and StratifiedKFold cross validation was used for validating the models.
5. TfidfVectorizer was used for feature extraction.
6. Several models were validated before choosing two best performing models.
7. Parameter tuning was done using the GridSearchCV
8. Best performing model was choosed as the final model.

### Models Performance: 
To see the models performance details please download the 'resume-classification.ipynb' file. Open it with jupyter notebook and check the last cell of 'Model Selection and Training' section.

## What the script does?
1. It uses the model to classify resumes (in .pdf format for now) in a <given_path>
2. Move the resumes in their respective folders (inside the <given_path> and named as the category) if the folder exist.
3. If the folder does not exist, it will make the folder (inside the <given_path> and named as the category) and move the resume there
4. Makes a 'categorized_resumes.csv' file in the <given_path> .

A ['categorized_resumes.csv'](https://github.com/MdMonoar/resume-classifier/tree/main/sample_output/categorized_resumes.csv) file is included in the repsitory inside the 'sample_output' folder which was created after running the script. You can check the difference: before running the script in ['sample_input'](https://github.com/MdMonoar/resume-classifier/tree/main/sample_input) and after running the script ['sample_output'](https://github.com/MdMonoar/resume-classifier/tree/main/sample_output) folder.

## How to run the script?
**Prerequisites:**
1. You should (not necessarily) create a virtual environment and activate that
2. You must have 'pip' installed

**Steps for checking:**
1. First clone the repository in a folder (eg. <download_folder>)
2. Open command line in the <download_folder> folder and Install the necessary packages by running the following command: <br> 'pip install -r requirements.txt'
3. Run the following command for testing the script: <br> 'python script.py <given_path>' ; Here given path is the path of the folder that contains the resumes
4. Check in the <given_path> to see the result.

! Important: There is a 'do_not_delete' folder in the repository. This is important for running the script. You know what to do : )
