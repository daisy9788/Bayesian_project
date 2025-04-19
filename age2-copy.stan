//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> y;
  vector[N] swang1;
  vector[N] age;

  // New inputs for predicting next observation
  real swang1_new;
  real age_new;
}

parameters {
  real alpha;                   // intercept
  real beta_swang1;             // coefficient for swang1
  real beta_age;                // coefficient for age
}
// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  // Priors (weakly informative)
  alpha ~ normal(0, 5);
  beta_swang1 ~ normal(0, 5);
  beta_age ~ normal(0, 5);

  // Likelihood
  y ~ bernoulli_logit(alpha + beta_swang1 * swang1 + beta_age * age);
}

generated quantities {
  real y_pred;
  y_pred = bernoulli_logit_rng(alpha + beta_swang1 * swang1_new
                                      + beta_age * age_new
                                    );
}

