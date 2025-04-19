library(tidyverse)    # Data manipulation
library(rstan)        # Bayesian modeling with Stan
library(pROC)         # ROC-AUC calculations
library(bayesplot)    # Bayesian diagnostics
library(loo)          # Model comparison
library(dplyr)
library(caret)

# import dataset
rhc <- read.csv("https://hbiostat.org/data/repo/rhc.csv")

# clean data
rhc_data <- rhc %>%
  filter(!is.na(death), !is.na(swang1)) %>%
  filter(cat1 == "MOSFw w/Sepsis")  %>%
  mutate(
    death = as.integer(death == "Yes"),
    swang1 = as.integer(swang1 == "RHC")
  ) %>%
  select(death, swang1, cat1,age)
rhc_data

set.seed(123)
train_indices <- sample(1:nrow(rhc_data), size = 0.8 * nrow(rhc_data))
train_data <- rhc_data[train_indices, ]
test_data <- rhc_data[-train_indices, ]

# fit frequentist logistic regression
freq_model <- glm(death ~ swang1 + age , 
                  data = train_data, 
                  family = binomial())
summary(freq_model)

# Predict probabilities on test data
freq_pred <- predict(freq_model, newdata = test_data, type = "response")
freq_pred

# 5
class_pred <- ifelse(freq_pred > 0.5, 1, 0)
class_pred

# Actual outcomes
actual <- test_data$death

# Confusion matrix
conf_matrix <- confusionMatrix(factor(class_pred), factor(actual))
print(conf_matrix)

# ROC Curve and AUC
roc_obj <- roc(actual, freq_pred)
auc_value <- auc(roc_obj)
plot(roc_obj, main = paste("ROC Curve (AUC =", round(auc_value, 3), ")"))
auc_value

# Bayesian regression model

stan_data <- list(
  N = nrow(rhc_data),
  y = rhc_data$death,
  swang1 = rhc_data$swang1,
  age = rhc_data$age
)

stan_model <- stan_model("/Users/daisyl/desktop/age-copy.stan")
fit <- sampling(stan_model, data = stan_data, chains = 4, iter = 2000, warmup = 1000, seed = 123)
fit

posterior_samples <- extract(fit)
pred_probs <- posterior_samples$y_pred_prob  # matrix: iterations x N
mean_pred <- colMeans(pred_probs)
mean_pred


ci_level <- 0.9
ci_plims <- c((1 - ci_level)/2, (1 + ci_level)/2)
N_obs <- nrow(rhc_data)
ci_limits <- matrix(0, nrow = N_obs, ncol = 2)
pred_probs <- numeric(N_obs)

stan_model2 <- stan_model("/Users/daisyl/desktop/age2-copy.stan")
for (i_test in 1:N_obs) {
  train_data <- rhc_data[-i_test, ]
  test_point <- rhc_data[i_test, ]
  
  stan_data_cv <- list(
    N = nrow(train_data),
    y = train_data$death,
    swang1 = train_data$swang1,
    age = train_data$age,

    swang1_new = test_point$swang1,
    age_new = test_point$age
    
  )
  

  
  fit_cv <- sampling(
    stan_model2,
    data = stan_data_cv,
    chains = 1,
    iter = 1000,
    warmup = 500,
    refresh = 0
  )
  
  samples <- extract(fit_cv)
  prob_samples <- samples$y_pred  # draws of predicted binary outcome
  pred_probs[i_test] <- mean(prob_samples)  # posterior mean
  ci_limits[i_test, ] <- quantile(prob_samples, probs = ci_plims)
}
ci_limits[N_obs, ]

# calibration plot
library(ggplot2)
cal_df <- rhc_data %>%
  mutate(
    pred = pred_probs,              # posterior means of predicted probability
    ci_low = ci_limits[, 1],        # lower credible interval
    ci_high = ci_limits[, 2],       # upper credible interval
    actual = death,
    bin = ntile(pred, 10)           # split into deciles
  )

bin_summary <- cal_df %>%
  group_by(bin) %>%
  summarise(
    mean_pred = mean(pred),
    obs_rate = mean(actual),
    lower = mean(ci_low),           # mean of the lower CI per bin
    upper = mean(ci_high)           # mean of the upper CI per bin
  )

ggplot(bin_summary, aes(x = mean_pred, y = obs_rate)) +
  geom_point(size = 2, color = "black") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.02, color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Bayesian Calibration Plot with 90% Credible Intervals (Binned)",
    x = "Mean Predicted Probability",
    y = "Observed Proportion of Deaths"
  ) +
  theme_minimal()

pred_probs
hist(pred_probs)

