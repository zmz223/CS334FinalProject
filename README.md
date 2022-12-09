# ML Final Project
**By Zach Zaiman, Mert Ozbay, Niki Vasan**

### Introduction 
This repository contains the code and files from our CS334 Machine Learning course final project. The goal of the project was to work collaboratively in teams to apply machine learning models to a real-world tasks. Our group chose to analyze the effect of confounding variables on a Convolutional Neural Network (CNN) designed to classify comorbidities in chest x-rays. We had four main deliverables: 
1. Spotlight Presentation: 90 second elevator pitch to the class that motivates the premise behind our study
2. Code Repository: Train, test, evaluate model and push code + relevant files to github repository 
3. Final Presentation: 5 minute presentation explaining model design, evaluation and results
4. Report: 10+ paged report detailing problem statement, relevant past research, methods, results and key findings of our project

The abstract of our paper is stated below:
> This paper seeks to build upon previous applications of deep learning models in healthcare by evaluating potential biases in a convolutional neural network (CNN) trained to classify comorbidities in chest X-rays using the MIMIC-IV and MIMIC-CXR datasets. The model is evaluated in terms of overall raw performance and stratified performance along five different clinical and demographic confounders relevant to healthcare: race, age, sex, ICU status and insurance type. We find that the model performs better on average for male patients and older patients, which is reflective of skews in the training data, but find inconclusive results when stratifying by ICU status. This study demonstrates that diverse training datasets are important when building models for clinical deployment, as representation biases in the data are indeed reflected in performance. Further research is needed to confirm possible biases with regards to race, ICU status and insurance type. The code and other resources are available at our github repository.


### Repo Structure 
* **analysis**: contains for folders for each of the three models with evaluation plots for each strata and final script
* **gradcams**: contains config file, training code and example gradcams (under `gradcams_out`)
* **reports**: contains final presentation slide deck and final report 
* **runs**: contains output directory for each CNN version trained (message authors for weights)
* **trainlib**: contains training code, config file and data prep file
