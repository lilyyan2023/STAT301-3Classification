# Load packages
library(tidyverse)
library(tidymodels)
library(naniar)
library(corrplot)
library(corrr)
library(glmnet)
library(ranger)
# set seed
set.seed(123)
# load data
loan_train <- read_csv("stat-301-3-classification-2021-loan-repayment/train.csv")
loan_test <- read_csv("stat-301-3-classification-2021-loan-repayment/test.csv")
dim(loan_train)
dim(loan_test)
loan_train <- loan_train %>% 
  mutate(hi_int_prncp_pd_f = factor(hi_int_prncp_pd))

# split data
loan_folds <- vfold_cv(data = loan_train, v = 5, repeats = 3)
# create another recipe
loan_recipe2 <- 
  recipe(hi_int_prncp_pd_f ~ int_rate + loan_amnt +
           out_prncp_inv + application_type + grade + initial_list_status + term,
         data = loan_train) %>%  
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% 
  step_normalize(all_predictors())


# random forest model
rf_model <- rand_forest(
  mode = "classification",
  mtry = tune(),
  min_n = tune()
) %>% 
  set_engine("ranger", importance = "impurity")

# set-up tuning grid ----
rf_params <- parameters(rf_model) %>% 
  update(mtry = mtry(range = c(2,6)))

# define tuning grid
rf_grid <- grid_regular(rf_params, levels = c(3,4))

# workflow ----
rf_workflow <- workflow() %>% 
  add_model(rf_model) %>% 
  add_recipe(loan_recipe2)

# Tuning/fitting
rf_tuned <- rf_workflow %>% 
  tune_grid(loan_folds, rf_grid)

write_rds(rf_tuned, "rf_results.rds")



