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
  int<lower=1> N;               // number of observations
  array[N] int<lower=0, upper=1> y;    // binary outcome (death)
  vector[N] swang1;             // binary predictor
  vector[N] age;                // numeric predictor
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
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
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(alpha + beta_swang1 * swang1[n]
                                  + beta_age * age[n]
                                 );
  }
}

generated quantities {
  vector[N] y_pred_prob;
  for (n in 1:N) {
    y_pred_prob[n] = inv_logit(alpha + beta_swang1 * swang1[n]
                                      + beta_age * age[n]
                                     );
  }
}
