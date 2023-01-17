##tidy models
### predictive modeling case study https://www.tidymodels.org/start/case-study/

# Load Libraries ----------------------------------------------------------

library(tidyverse)
library(readxl)
library(tidymodels)
library(lubridate)
library(ggthemes)
library(viridis)
library(vip)
library(missRanger)


# READ IN DATA & REFACTOR -------------------------------------------------

modeldata <- read_excel("data/modeldata.xlsx", 
                        col_types = c("text", "numeric", "text", 
                                      "date", "date", "numeric", "numeric", 
                                      "text", "text", "text", "text", "text", 
                                      "text", "text", "text", "text", "text", 
                                      "text", "numeric", "numeric", "numeric", 
                                      "numeric", "text", "text", "numeric", 
                                      "numeric", "numeric", "numeric", "numeric", 
                                      "numeric", "numeric", "numeric", "numeric", "numeric", 
                                      "numeric", "numeric", "numeric", "numeric", "numeric", 
                                      "numeric", "numeric", "numeric", "numeric", "numeric", 
                                      "numeric", "numeric", "numeric", "numeric", 
                                      "numeric", "numeric", "numeric", "text"))

summary(modeldata)

View(modeldata)

# There are a lot of variables in modeldata that probably don't help out with the model, and others are coded weird, so I'm gonna do some transmute action and fix them (supercedes a lot of cleanup work commented out below)

data_tidy <- modeldata %>%
  transmute(
    id = id,
    completed_interview = (completedinterviews == 1) %>% as.factor(),
    interview_initiate_date = ymd(interview_initated_date),
    days_since_covid_start = time_since_covid_start,
    age = age,
    gender = relevel(as.factor(gender), ref = "female") %>% as.numeric(),
    race = relevel(as.factor(race), ref = "black") %>% as.numeric(),
    ethnicity = (ethnicity == "hispanic") %>% as.numeric(), #What if nonwhite, non male was the default? Let's try it once maybe!
    interpreter_required = (interpreter_required == "yes") %>% as.numeric(),
    county = county,
    tract_poptotal = tract_poptotal,
    tract_popdensitysqmi = tract_popdensitysqmi,
    tract_nhl_wa = tract_nhl_wa,
    tract_white = tract_white,
    tract_minorityrace = tract_minorityrace,
    tract_englishltvw = tract_englishltvw,
    tract_laborforce = tract_laborforce,
    tract_unemployed = tract_unemployed,
    tract_healthsocialfactor = tract_healthsocialfactor,
    tract_pop_under5 = tract_pop_under5,
    tract_pop_over64 = tract_pop_over64,
    tract_lifeexpectancy = tract_lifeexpectancy,
    tract_poc = tract_poc,
    tract_lessthanhs = tract_lessthanhs,
    tract_lowincome = tract_lowincome,
    tract_disability = tract_disability,
    tract_housingcost = tract_housingcost,
    county_positivityrate = county_positivityrate,
    county_rateper100000 = county_rateper100000,
    county_deathsper100000 = 100000 * county_deaths / county_population, # Maybe it's better as a rate?
    completionrate = completionrate # Is this county call completion rate?
  ) %>%
  
  #There are some missing values so we're going to impute them.
  missRanger(num.trees = 100)

# ##Change Reference Levels
# levels(modeldata$completedint)
# modeldata$completedint <- relevel(modeldata$completedint, ref = "Y")
# 
# levels(modeldata$race)
# modeldata$race <- relevel(modeldata$race, ref = "white") ##white as reference category
# 
# levels(modeldata$ethnicity)
# modeldata$ethnicity <- relevel(modeldata$ethnicity, ref = "not_hispanic")



####

dim(data_tidy)

mod <- data_tidy %>% 
  count(completed_interview) %>% 
  mutate(prop = n/sum(n))

##92% yes and 8% no
## will need data splitting strategy and resample to balance
### NOTE: anytime 'children' is referenced  in the source doc it refers to 'completedint' (completed interviews) in our data


set.seed(123)
splits      <- initial_split(data_tidy, strata = completed_interview)

# Renaming these so they aren't confusing later
data_non_test <- training(splits)
data_test  <- testing(splits)

# training set proportions by completed_interview
data_non_test %>% 
  count(completed_interview) %>% 
  mutate(prop = n/sum(n))

# test set proportions by completedint
data_test  %>% 
  count(completed_interview) %>% 
  mutate(prop = n/sum(n))

# We'll use the validation_split() function to allocate 20% of the hotel_other stays to the validation set and 
# 30,000 stays to the training set. This means that our model performance metrics will be computed on a single set 
# of 7,500 hotel stays. This is fairly large, so the amount of data should provide enough precision to be a 
# reliable indicator for how well each model predicts the outcome with a single iteration of resampling.

set.seed(234)
val_set <- validation_split(data_non_test, 
                            strata = completed_interview, 
                            prop = 0.80)
val_set


# This function, like initial_split(), has the same strata argument, which uses stratified sampling to create 
# the resample. This means that we'll have roughly the same proportions of hotel stays with and without children 
# in our new validation and training sets, as compared to the original hotel_other proportions.


##a First Model: Penalized Logistic Regression
# ### Since our outcome variable children is categorical, logistic regression would be a good first model to start.
# Let's use a model that can perform feature selection during training. 
# The glmnet R package fits a generalized linear model via penalized maximum likelihood. 
# This method of estimating the logistic regression slope parameters uses a penalty on the process so that
# less relevant predictors are driven towards a value of zero. One of the glmnet penalization methods, 
# called the lasso method, can actually set the predictor slopes to zero if a large enough penalty is used.


lr_mod <- 
  logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")



# We'll set the penalty argument to tune() as a placeholder for now. 
# This is a model hyperparameter that we will tune to find the best value for making predictions with our data. 
# Setting mixture to a value of one means that the glmnet model will potentially remove irrelevant predictors 
# and choose a simpler model.


holidays <- c("AllSouls", "AshWednesday", "ChristmasEve", "Easter", 
              "ChristmasDay", "GoodFriday", "NewYearsDay", "PalmSunday")

lr_recipe <- 
  recipe(completed_interview ~ ., data = data_non_test) %>% 
  step_date(interview_initiate_date) %>% 
  step_holiday(interview_initiate_date, holidays = holidays) %>% 
  step_rm(interview_initiate_date) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())


lr_workflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(lr_recipe)

##create grid for tuning
lr_reg_grid <- tibble(penalty = 10^seq(-4, -1, length.out = 30))
lr_reg_grid %>% top_n(-5) # lowest penalty values
lr_reg_grid %>% top_n(5)  # highest penalty values


#train and tune model

###ERROR
library(glmnet)

lr_res <- 
  lr_workflow %>% 
  tune_grid(val_set,
            grid = lr_reg_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

lr_plot <- 
  lr_res %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + 
  geom_point() + 
  geom_line() + 
  ylab("Area under the ROC Curve") +
  scale_x_log10(labels = scales::label_number())

lr_plot





# A SECOND MODEL: TREE-BASED ENSEMBLE 

cores <- parallel::detectCores()
cores


rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger", num.threads = cores) %>% 
  set_mode("classification")


rf_recipe <- 
  recipe(completedint ~ ., data = hotel_other) %>% 
  step_date(interview_complete_date) %>% 
  step_holiday(interview_complete_date) %>% 
  step_rm(interview_complete_date) 


rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(rf_recipe)

rf_mod

extract_parameter_set_dials(rf_mod)


set.seed(345) ####ERROR for missing values
rf_res <- 
  rf_workflow %>% 
  tune_grid(val_set,
            grid = 25,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

rf_res %>% 
  show_best(metric = "roc_auc")
