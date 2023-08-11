library(tidyverse)
library(mlr3verse)

task = as_task_regr(diamonds, target = "price", id = "diamonds")

set.seed(1234)

splits = partition(task,ratio=0.8, stratify = TRUE)

train = task$clone()$filter(splits$train)
test = task$clone()$filter(splits$test)

cv10 = rsmp("cv", folds = 10)
cv10$instantiate(train)

measure = msr("regr.rmse")

# Create a graph

learner = lrn("regr.kknn",k=to_tune(c(1,5,10,30)))
graph =  po("scale") %>>% po("encode") %>>% learner
pipeline = GraphLearner$new(graph)

instance = ti(
  task = train,
  learner = pipeline,
  resampling = cv10,
  measures = measure,
  terminator = trm("none")
)

tuner = tnr("grid_search")
tuner$optimize(instance)

as.data.table(instance$archive)[order(regr.rmse),c(1,2)]

# Extract optimal parameters

optimal_params = instance$result$learner_param_vals

# Train pipeline with optimal parameters

pipeline$param_set$values = optimal_params[[1]]
pipeline$train(train)

# Predict and get scores

resultado = pipeline$predict(test)
resultado$score(measure)

