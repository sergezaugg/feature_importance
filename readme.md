# Assess procedures for "feature importance" against true feature importance

## Summary
This code is primarily didactic. 

Researchers for whom I work as data scientist often wish a ranking of feature importance.

I tell them that it is not so trivial ... but they don't fully believe me.

So here I made some intuitively appealing visuals to explain all this. 


## Rationale

*  Dataset for supervised classification are created with one binary class (the target) and 3 continuous features (the predictors)

*  The first two features can be informative for classification, while the third is always totally non-informative

*  How the first two features inform classification can be actively chosen (see three scenarios below)

*  Random Forest classifiers are trained for 'all 3 features' and for subsets of only two features 

*  The predictive performance (ROC-AUC) and the **impurity-based feature importance** are computed

*  All this is summarized on plots that allow to explain the strengths and limitations of **impurity-based feature importance**.

## Illustration

*  **Figure 1** shown the easy scenario that most people have in mind: Both features are independently informative.
    * Removing one feature reduces predictive performance only a bit.
*  In **Figure 2**, both feature are informative but highly redundant.
    * Removing one feature hardly affects predictive performance.
*  In **Figure 3**, the information for classification is jointly shared among both features.
    * Removing one feature kills predictive performance (AUC drops to approx 0.50).
* **Take home** In all 3 scenarios, the feature importance of full model is very similar (0.45, 0.45, 0.10). **However the impact of removing some variable is very different** 


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


