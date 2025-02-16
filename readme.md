# Can we really rank features according to their importance?

This code is primarily didactic. 
Researchers for whom I work as data scientist often wish a ranking of feature importance. 
So here I provide some intuitively appealing visuals to explain all this. 
Three scenarios are briefly shown in this readme, but many more can be assessed by running the code.

## Summary
*  Datasets for supervised classification are created with one binary class (the target) and 3 continuous features (the predictors)
*  The first two features can be informative for classification, while the third is always totally non-informative
*  How the first two features inform classification can be actively chosen (see three scenarios below)
*  Random Forest classifiers are trained for 'all 3 features' and for smaller subsets of the features 
*  The predictive performance (ROC-AUC) and the **impurity-based feature importance** are computed
*  All this is summarized on plots that allow to explain the strengths and limitations of **impurity-based feature importance**.

## Illustration

*  **Figure 1:** Scenario that most people have in mind: Both features are almost independent and equally informative.
    * Removing f01 or f02 reduces predictive performance only a bit.
*  **Figure 2:** Both feature are equally informative but highly redundant.
    * Removing f01 or f02 hardly affects predictive performance.
*  **Figure 3:** Information for classification is jointly shared among both features (i.e they are complementary).
    * Removing f01 or f02 kills predictive performance (AUC drops to almost 0.50).
*  **Take home** 
    * **In all 3 scenarios the feature importance of full model is very similar, approx. (0.45, 0.45, 0.10).**
    * **However, the impact of removing f01 or f02 is very different!** 


**Figure 1**

![](./pics/figure01.png)

**Figure 2**

![](./pics/figure02.png)

**Figure 3**

![](./pics/figure03.png)


## Dependencies
* Developed under Python 3.12.8
* First make a venv, then:
```
pip install -r requirements.txt
```

## Usage
* There is one script **main.py** meant for interactive use.
* Several scenarios can be defined in the dict at the top of the script
* A few helper functions are imported from **utils.py**


