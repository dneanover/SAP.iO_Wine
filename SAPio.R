#Importing tidyverse,setting wd, importing data----
library(tidyverse)
setwd('C:\\Users\\damon\\OneDrive - North Carolina State University\\Documents\\Job Hunt\\')
#Reading in the CSV
wine <- read_csv('SAPio_DataScience_Challenge.csv')
#Setting seed
set.seed(9688)

#Quick glance at the data to prove it is there----
glimpse(wine)
summary(wine)

#Setting the Target Variable----
target = wine$quality

#Cleaning the column names for R and pair plot----
names(wine)[names(wine) == 'fixed acidity'] <- 'fixed_acidity'
names(wine)[names(wine) == 'volatile acidity'] <- 'volatile_acidity'
names(wine)[names(wine) == 'citric acid'] <- 'citric_acid'
names(wine)[names(wine) == 'astringency rating'] <- 'astringency_rating'
names(wine)[names(wine) == 'residual sugar'] <- 'residual_sugar'
names(wine)[names(wine) == 'free sulfur dioxide'] <- 'free_sulfer_dioxide'
names(wine)[names(wine) == 'total sulfur dioxide'] <- 'total_sulfer_dioxide'


#Univariate analysis----

#Looking at the shape of the data for models-----
hist(wine$fixed_acidity) #slight skew to the right.
boxplot(wine$fixed_acidity)#above 10 appears to be an outlier
hist(wine$volatile_acidity) 
hist(wine$citric_acid) #not very normally distributed
hist(wine$astringency_rating) #slightly right skewed
hist(wine$residual_sugar) #very right skewed.  One very strong outlier
hist(wine$chlorides)#right skewed
hist(wine$free_sulfer_dioxide) #very right skewed
hist(wine$total_sulfer_dioxide) #less skewed
hist(wine$density)
hist(wine$pH) #approximately normal
hist(wine$sulphates)
hist(wine$alcohol)

#Frequency Counts-----
table(wine$type)
table(wine$vintage)
table(wine$quality)
hist(wine$quality) #Fairly normal distribution

#initial thought that reds would be better because of personal preference----
#So I made a graph.
ggplot(wine, aes(x=quality, fill=type))+
  geom_histogram(stat="count")+
  facet_grid(type~.)
#Reds generally are rated lower than whites

#removing the categorical variable for the cor plot----
wine_nums <- as.data.frame(wine[2:15])

#Correlation Matrix and Corplot----
library(corrplot)
library(Deducer)
cormatrix <- cor.matrix(wine_nums)
ggcorplot(cormatrix,data= wine[2:15], var_text_size = 5,
          cor_text_limits = c(5,10))
#Noting these for later analysis-----
#fixed_acidity very correlated with astringency_rating .9926
#total_sulfer_dioxide very correlated with free_sulfer_dioxide .7227
#alcohol with density -.69

#Missing Value check----
for (i in colnames(wine)){
  print(colnames(wine[i]))
  print(sum(is.na(wine[i]))/ nrow(wine[i]))
}
#residual_sugar is missing for 36.38% of observations
#astringency is next highest with 5.11%
#volatile_acidity 4.6%
#this information will be important when building a predictive model and deciding how to impute.

#Decision Tree----
#Starting with Decision Tree since it is able to handle missing values
library(rpart)
library(rpart.plot)
library(party)

tree <- rpart(as.factor(wine$quality) ~ ., data = wine, method = "class")
print(tree)
plotcp(tree)
printcp(tree)

rpart.plot(tree, type =1, fallen.leaves = TRUE)
#Decision tree was not helpful yet. ----

#Addressing Missing values----

#Drop Variable that has too high of a missing value threshold (33%)
wine_drop <- wine%>% dplyr::select(-residual_sugar)

#Trying MICE imputation for Variables with ~5 % or less missing
library(mice)
library(VIM)

#A great tool for identifying missing values
md.pattern(wine_drop)
mice_plot <- aggr(wine_drop, col=c('navyblue','yellow'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(wine_drop), cex.axis=.7,
                  gap=3, ylab=c("Missing data","Pattern"))
#Interesting pattern of missing variables.  Not uniformly missing

#Values are Imputed----
imputed_Data <- mice(wine_drop, m=5, maxit = 50, method = 'pmm', seed = 500)
wine_imputed <- complete(imputed_Data, 1)

#Looking good
mice_plot <- aggr(wine_imputed, col=c('navyblue','orange'),
                  numbers=TRUE, sortVars=TRUE,
                  labels=names(wine_imputed), cex.axis=.7,
                  gap=2, ylab=c("Missing data","Pattern"))

#Some Variable Adjustment for Modeling----
wine_imputed_factors <- wine_imputed #changing items to factor variables
wine_imputed_factors$type <- as.factor(wine_imputed_factors$type)
wine_imputed_factors$quality <- as.factor(wine_imputed_factors$quality)

#one hot encoding to work with scaling 
wine_imputed_factors$red <- if_else(wine_imputed_factors$type == "red", 1, 0)
wine_imputed_factors$white <- if_else(wine_imputed_factors$type == "white", 1, 0)
wine_imputed_factors <- wine_imputed_factors %>% dplyr::select(-type)
#initially thought this would be helpful, proved to be pointless

#Sampling -----
spec = c(train = .6, valid =.25, test = .15)

g = sample(cut(
  seq(nrow(wine_imputed_factors)),
  nrow(wine_imputed_factors)*cumsum(c(0,spec)),
  labels = names(spec)
))

res = split(wine_imputed_factors, g)

train = res$train
valid = res$valid
test = res$test

#Clustering ----
#Starting with Hieararchical for an idea of deciding # of clusters
#removing quality so I can make this unsupervised----
train_no_target <- train %>% dplyr::select(-quality)

#starting with hiearchical----
d <- dist(train_no_target, method = "euclidean")
fit <- hclust(d, method="ward.D")
plot(fit)
#something that I thought would show better groupings, but was mistaken
groups <-cutree(fit, k=5)
plot(groups)
rect.hclust(fit, k=10, border = "red")

#Kmeans----
#standardization function for Range standardization
range_standard = function(data) {
  x = unlist(data)
  out = (x-mean(x))/(max(x)-min(x))
}

wine_imputed_standardized <- wine_imputed_factors
wine_imputed_standardized <- wine_imputed_standardized %>% dplyr::select(-c(white,quality))
for( i in 1:ncol(wine_imputed_standardized)){
  wine_imputed_standardized[,i] = range_standard(wine_imputed_standardized[,i])
}
#sampling----
spec = c(train = .6, valid =.25, test = .15)

g = sample(cut(
  seq(nrow(wine_imputed_standardized)),
  nrow(wine_imputed_standardized)*cumsum(c(0,spec)),
  labels = names(spec)
))

res = split(wine_imputed_standardized, g)

train_scaled = res$train
valid_scaled = res$valid
test_scaled = res$test

#initializing vector for total within sum of squares----
tot_wss = vector(length = 15)

#Testing clustering solutions 1 - 15 -----
for (i in 1:15){
  kmeans.out = kmeans(wine_imputed_standardized, i, iter.max = 100, algorithm = "Hartigan-Wong")
  tot_wss[i] = kmeans.out$tot.withinss
}

plot (tot_wss, type = "b", xlab ="Number of Clusters",  main = "TWSS for K Clusters", ylab = "Total Within SS")

print(tot_wss)
#looks like 5 clusters based on the plot

#Kmeans Clustering -----
kmeans_clusters <- kmeans(wine_imputed_standardized, 5, iter.max = 100, nstart = 20, algorithm = "Hartigan-Wong")
summary(kmeans_clusters)
table(kmeans_clusters$cluster)

#Adding cluster to standardized set
wine_imputed_standardized$cluster <- kmeans_clusters$cluster
#add cluster to normal set.
wine_imputed_factors$cluster <- kmeans_clusters$cluster

#plutting quality against cluster----
plot(wine_imputed_factors$quality, col=wine_imputed_factors$cluster)

#Better more compelling plot----
#Confusing color choice, but green = Red Wine, red = White Wine
ggplot(wine_imputed_factors, aes(x=quality, fill=as.factor(red)))+
    geom_histogram(stat="count")+
    facet_grid(cluster~.)

wine_imputed_factors$quality_num <- as.numeric(wine_imputed_factors$quality)
wine_imputed_factors$quality_num2 <- wine_imputed_factors$quality_num + 2
#From the GGplot it looks like we have 3 red clusters, and 2 white clusters.  
#Frequency Plot----
#confirms breakdown of clusters
table(wine_imputed_factors$quality, wine_imputed_factors$red, wine_imputed_factors$cluster)

#Tree part two for cluster classification----
tree_data <- wine_imputed_factors %>% dplyr::select(-c(quality, white))
tree_cluster <- rpart(as.factor(cluster) ~. , data= tree_data)

print(tree_cluster)
rpart.plot(tree_cluster, type =3, fallen.leaves = TRUE, under = TRUE, cex=1)

#Pruned Decision tree
pruned_tree <- prune(tree_cluster, cp = 0.016593 )
print(pruned_tree)
rpart.plot(pruned_tree, type =3, fallen.leaves = TRUE, under = TRUE, cex=1)
#Much better

#Here I would retrain the tree with training data and test it, as well as, rescore clsuters
#but in the essence of time, I will use the tree from the whole dataset to profile. 
#I wanted to see what was going to be important for classifying wines between clusters
#Vintage is our first split, then red or white, then alcohol.

#PCA analysis----
#This is a way to visualize the components and see the splits ----
PCA_wine <- prcomp(wine_imputed_factors[,c(1:12,14:15)], center = TRUE, scale = TRUE)
summary(PCA_wine)
plot(PCA_wine, type = "l")

#Can see groupings of variables together and two clusers along the first 2 PCA components
biplot(PCA_wine)
library(pca3d)

#2D Plot of the PCs coloring denotes factors----
plot(PCA_wine$x[,1:2], col=as.factor(kmeans_clusters$cluster)) #This uses only 52% of Variance
#But shows that there are two groups, Red and White wines

#3D PCA plot----
#This 3D uses 64% of the Variance
#Color is the Cluster assignment, Shape is the wine Type: Triangle for Red, Sphere for White
pca3d(PCA_wine, col=as.factor(kmeans_clusters$cluster), group=wine_imputed_standardized$red)

#Cluster Profiling----
#Using our Decision Tree we can see the following from each cluster

#Cluster 1 : Younger Reds----
  #Terminal Node: Younger than 2006, Red ine
cluster1 <- wine_imputed_factors[wine_imputed_factors$cluster == 1,]
summary(cluster1)
table(cluster1$quality)
mean(cluster1$quality_num2)
median(cluster1$quality_num2)

#Cluster 2: ----- Vintage, Light, White Wines
  #Terminal Node: Older than 2006, Is White, Less than 10.75 < ABV
cluster2 <- wine_imputed_factors[wine_imputed_factors$cluster == 2,]
summary(cluster2)
table(cluster2$quality)
mean(cluster2$quality_num2)
median(cluster2$quality_num2)
#Lower quality, with few high quality wines.

#Cluster 3: Young, White Wines-----
  #Terminal Node1: Younger than 2006, is White
cluster3 <- wine_imputed_factors[wine_imputed_factors$cluster == 3,]
summary(cluster3)
table(cluster3$quality)
mean(cluster3$quality_num2)
median(cluster3$quality_num2)
#Better quality than 2 with 3 of the 5 9s!

#Cluster 4: Older Reds-----
#Terminal Node: Older than 2006, is Red
cluster4 <- wine_imputed_factors[wine_imputed_factors$cluster == 4,]
summary(cluster4)
table(cluster4$quality)
mean(cluster4$quality_num2)
median(cluster4$quality_num2)

#Cluster 5: Strong, Vintage, Whites-----
#Terminal Node: Older than 2006, Is White, Alcohol >= 10.75 
cluster5 <- wine_imputed_factors[wine_imputed_factors$cluster == 5,]
summary(cluster5)
table(cluster5$quality)
mean(cluster5$quality_num2)
median(cluster5$quality_num2)


#The only 9's were found in Clusters 5 and 3
# Cluster 5 is on average higher, 


#To compare groups, useful to chain together commands
#Quality
quality <-wine_imputed_factors %>% group_by(cluster) %>% summarize(mean = mean(quality_num2), 
                                                                   max = max(quality_num2),
                                                                   min = min(quality_num2))
quality
#Alcohol
alcohol <- wine_imputed_factors %>% group_by(cluster) %>% summarize(mean = mean(alcohol), 
                                                                    max = max(alcohol),
                                                                    min = min(alcohol))
alcohol
#Cluster 5 has the highest level of alcohol, and cluster 2 has the lowest.
#This is the same relationship as quality

#Fixed Acidity
fixed_acidity <- wine_imputed_factors %>% group_by(cluster) %>% summarize(mean = mean(fixed_acidity), 
                                                                    max = max(fixed_acidity),
                                                                    min = min(fixed_acidity))
fixed_acidity
#Red's have higher acid, while Cluster 3 and 5 have lower average fixed acidity than the other white

#Volatile Acidity
volatile_acidity <- wine_imputed_factors %>% group_by(cluster) %>% summarize(mean = mean(volatile_acidity), 
                                                                          max = max(volatile_acidity),
                                                                          min = min(volatile_acidity))
volatile_acidity
#similar relationship, but Cluster 5 is a little higher than 2 and 3

#Citric Acid
citric_acid <- wine_imputed_factors %>% group_by(cluster) %>% summarize(mean = mean(citric_acid), 
                                                                             max = max(citric_acid),
                                                                             min = min(citric_acid))
citric_acid
#Clusters 1 and 4 have lower citric acid, Cluster 3 has the highest

#Astringency Rating
astringency_rating <- wine_imputed_factors %>% group_by(cluster) %>% summarize(mean = mean(astringency_rating), 
                                                                        max = max(astringency_rating),
                                                                        min = min(astringency_rating))
astringency_rating
#Cluster 1 and 4 are the highest, with 3 and 5 having lower ratings of Astringency

#Chlorides
chlorides <- wine_imputed_factors %>% group_by(cluster) %>% summarize(mean = mean(chlorides), 
                                                                               max = max(chlorides),
                                                                               min = min(chlorides))
chlorides
#Reds (1 and 4) have highest chlorides, but 3 and 5 have lower chlorides than Cluster 2

#Free Sulfer Dioxide
free_sulfer_dioxide <- wine_imputed_factors %>% group_by(cluster) %>% summarize(mean = mean(free_sulfer_dioxide), 
                                                                      max = max(free_sulfer_dioxide),
                                                                      min = min(free_sulfer_dioxide))
free_sulfer_dioxide
#Reds have lower FSD, but Cluster 3 and 5 have lower FSD

#Total Sulfer Dioxide
total_sulfer_dioxide <- wine_imputed_factors %>% group_by(cluster) %>% summarize(mean = mean(total_sulfer_dioxide), 
                                                                                max = max(total_sulfer_dioxide),
                                                                                min = min(total_sulfer_dioxide))
total_sulfer_dioxide
#Cluster 3 and 5 have lower TSD

#Density
density <- wine_imputed_factors %>% group_by(cluster) %>% summarize(mean = mean(density), 
                                                                                max = max(density),
                                                                                min = min(density))
density
#Cluster 3 and 5 are less dense

#pH
pH <- wine_imputed_factors %>% group_by(cluster) %>% summarize(mean = mean(pH), 
                                                                                max = max(pH),
                                                                                min = min(pH))
pH

#3 and 5 have higher PH

#Sulphates
sulphates <- wine_imputed_factors %>% group_by(cluster) %>% summarize(mean = mean(sulphates), 
                                                                                max = max(sulphates),
                                                                                min = min(sulphates))
sulphates
#3 and 5 have lower sulphates

#Current Verdict----
#If you are looking for a higher quality wine, choose a White Wine.
  #If it was made before 2006, pick one with >10.75% ABV
  #Otherwise the alcohol is less important.

  #Also consider buying a wine with these distinguishing features from Cluster 3 and 5:
    #Lower fixed acidity
    #Lower astringency rating
    #Lower chlorides count
    #Lower Free and Total Sulfer Dioxide count
    #Lower density
    #Higher PH
    #Lower sulphates

#Room for Expansion-------
  #1.Further Cluster Breakdowns: 
    #Given that there are now clusters I would have divided the clusters further, 
    #subsetting by quality level, looking for differences between the group.
    #This would have allowed me to really pull out key differences between the different whites and reds.
  #2. Examination of Reds:
    #Reds and Whites are very different wintes as shown by the summarize/groupby statistics. 
    #I'd like to expand further to see what makes a good red, separate from what makes a good wine.


#Linear Regression Done early on but not helpful----
null_model <- lm(quality ~1, data=train)
summary(null_model)

full_regular_model <- lm(quality ~ fixed_acidity + volatile_acidity + citric_acid + astringency_rating + chlorides +
                    free_sulfer_dioxide + total_sulfer_dioxide + density + pH + sulphates+
                      alcohol + vintage + red, data=train)
summary(full_regular_model)

full_twoway_model <- lm(quality ~ (fixed_acidity + volatile_acidity + citric_acid + astringency_rating + chlorides +
     free_sulfer_dioxide + total_sulfer_dioxide + density + pH + sulphates+
     alcohol + vintage + red )^2, data=train)
summary(full_twoway_model)

step_model <- step(null_model, scope =list(upper=full_regular_model), data=train, direction = "both")
summary(step_model)

backward_model <- step(full_twoway_model, data=train, direction ="backward")
summary(backward_model)

step_pred <- predict(step_model, valid)
step_res <- valid$quality - step_pred
sqrt(sum(step_res)^2 / nrow(valid))
plot(valid$quality, step_pred) 

plot(step_model)

step_residuals <- resid(step_model)
plot(train$quality, step_residuals)
#Very fruitless effort given that at best we can account for only 40% of variance due to its not linear nature