import os
import sys
import shutil
import pandas as pd
import PyPDF2
import joblib
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')

stopWords = nltk.corpus.stopwords.words('english')
# Function to extract text from PDF resume
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def cleaner(text):
    # converting the whole string to lower case
    txt = text.lower()

    # removing non-english characters, punctucations and numbers
    txt = re.sub('[^a-zA-Z]', ' ', txt)

    # removing extra spaces from the text
    txt = re.sub('\s+', ' ', txt)

    # tokenizing the string
    txt = nltk.tokenize.word_tokenize(txt)

    # removing stop words
    txt = [w for w in txt if not w in stopWords]

    return ' '.join(txt)

# Function to categorize resumes and move them to respective folders
def categorize_resumes(input_dir, model):
    categorized_resumes = []

    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):  # Check if the file is a PDF
            pdf_path = os.path.join(input_dir, filename)
            resume_text = extract_text_from_pdf(pdf_path)
            cleaned_text = cleaner(resume_text)
            text = [cleaned_text]

            # predicting
            predicted_category = model.predict(text)

            # Create or move the resume to the respective category folder
            category_folder = os.path.join(input_dir, predicted_category[0])
            os.makedirs(category_folder, exist_ok=True)
            shutil.move(pdf_path, os.path.join(category_folder, filename))

            # Record the categorized resume
            categorized_resumes.append({'filename': filename, 'category': predicted_category})

    return categorized_resumes

# Main function
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py path/to/dir")
        sys.exit(1)

    input_dir = sys.argv[1]

    model_path = './do_not_delete/model.joblib'
    model = joblib.load(model_path)

    categorized_resumes = categorize_resumes(input_dir, model)

    # Save categorized resumes to a CSV file
    csv_filename = "categorized_resumes.csv"
    output_csv_path = os.path.join(input_dir, csv_filename)
    categorized_df = pd.DataFrame(categorized_resumes)
    categorized_df.to_csv(output_csv_path, index=False)
    print('Successfully moved resumes to their respective category folder.')
    print('To see the categorized csv file, check the path you provided.')
