# Bayesian_project
447_project
Comparing Frequentist and Bayesian Regression for Mortality Prediction

This project will use Right Heart Catheterization Dataset which obtained from http://hbiostat.org/data courtesy of the Vanderbilt University Department of Biostatistics.

## Method
1. Variables: 
Outcome (Y): death (binary: Yes/No)
Predictor (X): swang1 (binary: RHC/No RHC)
2. Frequentist Logistic Regression: 
Use glm function to fit the logistic regression model
Report coefficients, p-values, odds ratios, and 95% confidence intervals
3. Bayesian Logistic Regression:
Select appropriate priors and evaluate posterior distributions.
Use Markov Chain Monte Carlo (MCMC) sampling
4. Model comparison
Compare the coefficients, uncertainty, and predictive performance
Evaluate model accuracy using AUC-ROC curves, posterior distribution, and convergence diagnostics
5. Discussion
Interpret the differences in results between the Frequentist and Bayesian regressions.
Discuss the advantages and limitations of each method for healthcare decision-making.
