# Description
---
# Dream Job Navigator

An interactive web application that helps users find similar companies based on their ideal job preferences.


The application will be available at `http://127.0.0.1:8050/` in your web browser.

## Interactive Features

### Input Section (Left Panel)
1. **Score Input**
   - Rate your ideal company in 6 dimensions (0-5):
     - Opportunities
     - Compensation
     - Management
     - Work-life Balance
     - Culture
     - Diversity

2. **Review Input**
   - Enter pros and cons (drawbacks that you can tolerate) about your ideal company

3. **Weight Slider**
   - Adjust the weight between quantitative scores and text reviews (0-100)
   - 0: Only consider scores
   - 100: Only consider reviews

4. **Distribution Resolution**
   - Choose the number of company clusters (2-10)
   - Higher number means more detailed categorization

5. **Search Button**
   - Generate visualization based on your inputs

### Visualization Section (Center Panel)
1. **Interactive Network Graph**
   - Nodes represent companies
   - Node size indicates average score
   - Node color represents company category
   - Edge thickness shows similarity between companies
   - Click any node to:
     - Highlight connected companies
     - View detailed company information

### Company Information Section (Right Panel)
- Displays detailed information about the selected company:
  - Company name
  - Scores in 6 dimensions
  - Pros and cons reviews

## Data Flow

1. **User Input Processing**
   - Collects user's ideal company preferences
   - Converts text reviews into embeddings using BERT model
   - Normalizes all scores

2. **Similarity Calculation**
   - Combines existing company data with user input
   - Performs dimensionality reduction on text embeddings (UMAP)
   - Calculates similarity scores between companies

3. **Clustering and Visualization**
   - Groups similar companies using K-means clustering
   - Creates network graph structure:
     - Nodes: Companies
     - Edges: Similarity connections
   - Identifies representative companies for each cluster

4. **Interactive Updates**
   - Updates graph highlighting on node click
   - Synchronizes company information display
   - Maintains smooth user interaction with state management

## Note
The application requires the company database file `firmInfo_cleaned_longProsCons_embeddings_utf8.csv` in the same directory. <br>

---
# Installation
1. Create a Python 3.9 environment
   - It is recommended to use an environment manager of your choice e.g. venv or Anaconda
2. In a terminal with your python environment active, <br>
from the root of this code folder, run
> pip install -r requirements.txt
    
    pip may have to be substituted with pip3 depending on your environment setup
Ensure all package installations are successful before moving on to execution.

---
# Execution
1. In a terminal with your python environment active, <br>
from the root of this code folder, run
> python JobApp.py
    
    python may have to be substituted with python3 depending on your environment setup
2. Open a browser to http://127.0.0.1:8050 or http://localhost:8050 and the dashboard should be running
    - the link may be different if configurations or code is changed
    - the exact browser link will be reflected in the terminal outputs