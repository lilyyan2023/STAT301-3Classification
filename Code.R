# Load packages
library(tidyverse)
library(tidymodels)
library(naniar)
# set seed
set.seed(123)

# load data
loan_train <- read_csv("stat-301-3-classification-2021-loan-repayment/train.csv")
loan_test <- read_csv("stat-301-3-classification-2021-loan-repayment/test.csv")
dim(loan_train)
dim(loan_test)
miss_var_summary(loan_train)
skimr::skim_without_charts(loan_train)
# split data
loan_folds <- vfold_cv(data = loan_train, v = 5, repeats = 3)

# define recipe
# get rid of id. addr_state ? 
#earliest_cr_line -> date conversion, no emp_title, no last_credit_pull_d -> date conversion
# no sub_grade
# corrplot
loan_recipe <- recipe()





