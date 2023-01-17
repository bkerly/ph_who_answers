##tidy models
### predictive modeling case study https://www.tidymodels.org/start/case-study/

####READ IN DATA & REFACTOR:
library(readxl)
modeldata <- read_excel("C:/Users/maramire/Desktop/IPG3.0/modeldata.xlsx", 
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
View(modeldata)

##Change Reference Levels
levels(modeldata$completedint)
modeldata$completedint <- relevel(modeldata$completedint, ref = "Y")

levels(modeldata$race)
modeldata$race <- relevel(modeldata$race, ref = "white") ##white as reference category

levels(modeldata$ethnicity)
modeldata$ethnicity <- relevel(modeldata$ethnicity, ref = "not_hispanic")



####
library(tidymodels)
library(viridis)
library(vip)

dim(modeldata)

mod <- modeldata %>% 
  count(completedint) %>% 
  mutate(prop = n/sum(n))

##92% yes and 8% no
## will need data splitting strategy and resample to balance
### NOTE: anytime 'children' is referenced it refers to 'completedint' (completed interviews) in our data


set.seed(123)
splits      <- initial_split(modeldata, strata = completedint)

hotel_other <- training(splits)
hotel_test  <- testing(splits)

# training set proportions by completedint
hotel_other %>% 
  count(completedint) %>% 
  mutate(prop = n/sum(n))

# test set proportions by completedint
hotel_test  %>% 
  count(completedint) %>% 
  mutate(prop = n/sum(n))

# We'll use the validation_split() function to allocate 20% of the hotel_other stays to the validation set and 
# 30,000 stays to the training set. This means that our model performance metrics will be computed on a single set 
# of 7,500 hotel stays. This is fairly large, so the amount of data should provide enough precision to be a 
# reliable indicator for how well each model predicts the outcome with a single iteration of resampling.

set.seed(234)
val_set <- validation_split(hotel_other, 
                            strata = completedint, 
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
  recipe(completedint ~ ., data = hotel_other) %>% 
  step_date(interview_complete_date) %>% 
  step_holiday(interview_complete_date, holidays = holidays) %>% 
  step_rm(interview_complete_date) %>% 
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
