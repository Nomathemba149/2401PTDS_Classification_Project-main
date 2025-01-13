# 2401PTDS_Classification_Project

# Analysing News Articles Dataset


![](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](URL_TO_YOUR_APP)

<div id="main image" align="center">
  <img src="https://github.com/ereshia/2401FTDS_Classification_Project/blob/main/announcement-article-articles-copy-coverage.jpg" width="550" height="300" alt=""/>
</div>

## Table of content
1. Objectives 
2. The dataset 
3. Creating the coding environment 
4. Notebook structure 
5. MLFlow 
6. Streamlit App
7. Team members 

## 1. Objectives  <a class="anchor" id="project-description"></a>
Objectives: The objective is to deliver a practical demonstration of the implementation of machine learning methods in natural language processing applications. This comprehensive project covers the full workflow, which includes data acquisition, preprocessing, model training, assessment, and ultimate deployment. The key stakeholders involved in the news classification initiative for the news organization consist of the editorial team, IT and technical support, management, and readers. These parties are focused on achieving better content categorization, increased operational efficiency, and an improved user experience.




## 2. Dataset <a class="anchor" id="dataset"></a>
The dataset is comprised of news articles that need to be classified into categories based on their content, including `Business`, `Technology`, `Sports`, `Education`, and `Entertainment`. You can find both the `train.csv` and `test.csv` datasets [here](https://github.com/Jana-Liebenberg/2401PTDS_Classification_Project/tree/main/Data/processed). 

**Dataset Features:**
| **Column**                                                                                  | **Description**              
|---------------------------------------------------------------------------------------------|--------------------   
| Headlines   | 	The headline or title of the news article.
| Description | A brief summary or description of the news article.
| Content | The full text content of the news article.
| URL | The URL link to the original source of the news article.
| Category | The category or topic of the news article (e.g., business, education, entertainment, sports, technology).

## 3. Packages <a class="anchor" id="packages"></a>

To carry out all the objectives for this repo, the following necessary dependencies were loaded:
+ `Pandas 2.2.2` and `Numpy 1.26`
+ `Matplotlib 3.8.4`
 

## 4. Creating the coding environment  <a class="anchor" id="environment"></a>



### Create the new evironment - you only need to do this once

```bash
# create the conda environment
conda create --name <env>
```

### This is how you activate the virtual environment in a terminal and install the project dependencies

```bash
# activate the virtual environment
conda activate <env>
# install the pip package
conda install pip
# install the requirements for this project
pip install -r requirements.txt
```
## 5. MLFlow<a class="anchor" id="mlflow"></a>

MLOps, which stands for Machine Learning Operations, is a practice focused on managing and streamlining the lifecycle of machine learning models. The modern MLOps tool, MLflow is designed to facilitate collaboration on data projects, enabling teams to track experiments, manage models, and streamline deployment processes. For experimentation, testing, and reproducibility of the machine learning models in this project, you will use MLflow. MLflow will help track hyperparameter tuning by logging and comparing different model configurations. This allows you to easily identify and select the best-performing model based on the logged metrics.

- Please have a look here and follow the instructions: https://www.mlflow.org/docs/2.7.1/quickstart.html#quickstart

## 6. Streamlit App

## 7. Team Members<a class="anchor" id="team-members"></a>

| Name                                                                                        |  Email              
|---------------------------------------------------------------------------------------------|--------------------             
| [Maphuthi Matsape]                                                                          | maphuthimatsape@gmail.com
| [Nomathemba A Maphike]                                                                      | namaphike@gmail.com
| [Mikateko Chauke]                                                                           | mikaprudence@gmail.com




