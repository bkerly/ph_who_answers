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
library(glmnet)


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

#View(modeldata)

# There are a lot of variables in modeldata that probably don't help out with the model, and others are coded weird, so I'm gonna do some transmute action and fix them (supercedes a lot of cleanup work commented out below)

data_tidy <- modeldata %>%
  transmute(
    #id = id,
    completed_interview = (completedinterviews == 1) %>% as.factor(),
    interview_initiate_date = ymd(interview_initated_date),
    days_since_covid_start = time_since_covid_start,
    age = age,
    gender = relevel(as.factor(gender), ref = "female") %>% as.numeric(),
    race = relevel(as.factor(race), ref = "black"),
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


# Create data splits ------------------------------------------------------



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



# ## First Model: Penalized Logistic Regression ---------------------------


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



# Train and tune LR model -------------------------------------------------




lr_res <- 
  lr_workflow %>% 
  tune_grid(val_set,
            grid = lr_reg_grid,
            control = control_grid(save_pred = TRUE,extract = extract_fit_engine),
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


# Check out our top performing models

top_models <-
  lr_res %>% 
  show_best("roc_auc", n = 15) %>% 
  arrange(penalty) 
top_models

# Look at our best model
lr_best <- 
  lr_res %>% 
  collect_metrics() %>% 
  arrange(penalty) %>% 
  slice(13)
lr_best

# Plot AUC for our best model
lr_auc <- 
  lr_res %>% 
  collect_predictions(parameters = lr_best) %>% 
  roc_curve(completed_interview, .pred_FALSE) %>% 
  mutate(model = "Logistic Regression")

autoplot(lr_auc)

# Get variable values for our best model
# cf: https://stackoverflow.com/questions/63087592/how-to-extract-glmnet-coefficients-produced-by-tidymodels
extracted_linear_model <- lr_workflow %>%
  finalize_workflow(lr_best) %>%
  fit(data_non_test) %>%
  extract_fit_parsnip() %>%
  tidy() %>%
  arrange(desc(abs(estimate)))


# Random forest model -----------------------------------------------------



cores <- parallel::detectCores()
cores


rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger", num.threads = cores, importance = "impurity") %>% 
  set_mode("classification")


rf_recipe <- 
  recipe(completed_interview ~ ., data = data_non_test) %>% 
  step_date(interview_initiate_date) %>% 
  step_holiday(interview_initiate_date) %>% 
  step_rm(interview_initiate_date) 


rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(rf_recipe)

rf_mod

extract_parameter_set_dials(rf_mod)


set.seed(345) ####ERROR for missing values.
#Now there are no more missing values! Instead we imputed them.
rf_res <- 
  rf_workflow %>% 
  tune_grid(val_set,
            grid = 25,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc))

rf_res %>% 
  show_best(metric = "roc_auc")

autoplot(rf_res)

rf_best <- 
  rf_res %>% 
  select_best(metric = "roc_auc")

rf_best

rf_res %>% 
  collect_predictions()

rf_auc <- 
  rf_res %>% 
  collect_predictions(parameters = rf_best) %>% 
  roc_curve(completed_interview, .pred_FALSE) %>% 
  mutate(model = "Random Forest")

autoplot(rf_auc)

bind_rows(rf_auc, lr_auc) %>% 
  select(-.threshold) %>%
  ggplot() + 
  geom_path(aes(x = 1 - specificity, y = sensitivity, color = model),
            linewidth = 1.5, alpha = 0.8) +
  geom_abline(linetype = 3,linewidth = 1) + 
#coord_equal() + 
  theme_fivethirtyeight() +
  labs(color = "Model")

# extract the best rf model
extracted_rf_model <- rf_workflow %>%
  finalize_workflow(rf_best) %>%
  fit(data_non_test) %>%
  extract_fit_engine() 

extracted_rf_model %>%
  ranger::importance_pvalues(method = "janitza")

# Plot variable importance
feat_imp_df <- importance(extracted_rf_model) %>% 
  data.frame() %>% 
  rownames_to_column() %>%
  `colnames<-`(c("feature","importance"))

# plot dataframe
ggplot(feat_imp_df, aes(x = reorder(feature, importance), 
                        y = importance)) +
  geom_bar(stat='identity') +
  coord_flip() +
  theme_classic() +
  labs(
    x     = "Feature",
    y     = "Importance",
    title = "Feature Importance: <Model>"
  )
### Nothing beyond here works!
# To do: Test the best parameters with the test data set to see if it works good.


# 
# # Let's test the model we like --------------------------------------------
# # the best model
# last_rf_mod <- 
#   rand_forest(mtry = 14, min_n = 26, trees = 1000) %>% 
#   set_engine("ranger", num.threads = cores, importance = "impurity") %>% 
#   set_mode("classification")
# 
# rf_recipe_test <- 
#   recipe(completed_interview ~ ., data = data_test) %>% 
#   step_date(interview_initiate_date) %>% 
#   step_holiday(interview_initiate_date) %>% 
#   step_rm(interview_initiate_date) 
# 
# # the last workflow
# last_rf_workflow <- 
#   rf_workflow %>% 
#   update_model(last_rf_mod) %>%
#   update_recipe(rf_recipe_test)
# 
# set.seed(345)
# last_rf_fit <- 
#   last_rf_workflow 
# 
# last_rf_fit %>% 
#   collect_metrics()
# 
# rf_auc <- 
#   rf_res %>% 
#   collect_predictions(parameters = rf_best) %>% 
#   roc_curve(completed_interview, .pred_FALSE) %>% 
#   mutate(model = "Random Forest")
# 
# # Deeper dive into the best RF model ---------------------------------
# 
# # the last model
# last_rf_mod <- 
#   rand_forest(mtry = 14, min_n = 26, trees = 1000) %>% 
#   set_engine("ranger", num.threads = cores, importance = "impurity") %>% 
#   set_mode("classification")
# 
# # the last workflow
# last_rf_workflow <- 
#   rf_workflow %>% 
#   update_model(last_rf_mod)
# 
# # the last fit
# set.seed(345)
# last_rf_fit <- 
#   last_rf_workflow %>% 
#   last_fit(splits)
# 
# last_rf_fit
# 
# last_rf_fit %>% 
#   collect_metrics()
# 
# last_rf_fit %>% 
#   extract_fit_parsnip() %>% 
#   vip(num_features = 20)