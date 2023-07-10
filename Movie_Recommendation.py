from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import split, explode


spark = SparkSession.builder \
        .master("local[2]") \
        .appName("Lab 5 Exercise") \
        .config("spark.local.dir","/fastdata/acq21ps") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")





# load in ratings data
ratings = spark.read.load('../Data/ml-20m/ratings.csv', format = 'csv', inferSchema = "true", header = "true").cache()
#ratings.show(20,False)
#ratings.printSchema()
myseed = 1765677
print("-------------------------------- TASK A QUESTION 1 ------------------------------------------------------------------")
# Sorting the ratings dataframe by the timestamp column in ascending order so that earlier times occur before later times
sorted_ratings = ratings.orderBy("timestamp", ascending=True).cache()
#sorted_ratings.show(20,False)

# Forming the respective training sets by changing the training ratio to 0.4,0.6 and 0.6.
num_rows = sorted_ratings.count()
training_ratio = 0.4
num_training_rows = int(num_rows * training_ratio)

#The rows must be taken from top to ensure that earlier times should be on the training sets
training_data1 = sorted_ratings.limit(num_training_rows)
test_data1 = sorted_ratings.subtract(training_data1) # Including all the rows in the test data other than than the ones in the training set
training_data1 = training_data1.cache()
test_data1 = test_data1.cache()
training_data1.show(20,False)
print(training_data1.count()/num_rows)
test_data1.show(20,False)
print(test_data1.count()/num_rows)

training_ratio = 0.6
num_training_rows = int(num_rows * training_ratio)
training_data2 = sorted_ratings.limit(num_training_rows)
test_data2 = sorted_ratings.subtract(training_data2)
training_data2 = training_data2.cache()
test_data2 = test_data2.cache()
training_data2.show(20,False)
print(training_data2.count()/num_rows)
test_data2.show(20,False)
print(test_data2.count()/num_rows)

training_ratio = 0.8
num_training_rows = int(num_rows * training_ratio)
training_data3 = sorted_ratings.limit(num_training_rows)
test_data3 = sorted_ratings.subtract(training_data3)
training_data3 = training_data3.cache()
test_data3 = test_data3.cache()
training_data3.show(20,False)
print(training_data3.count()/num_rows)
test_data3.show(20,False)
print(test_data3.count()/num_rows)

print("------------------------------------ TASK A QUESTION 2 --------------------------------------------------------")
# Lab-5 model
als1 = ALS(userCol = "userId", itemCol = "movieId", seed = myseed, coldStartStrategy = "drop")
# My model with rank 25
als2 = ALS(rank = 25, userCol = "userId", itemCol = "movieId", seed = myseed, coldStartStrategy = "drop")
# define evaluators
evaluator1 = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction")
evaluator2 = RegressionEvaluator(metricName = "mse", labelCol = "rating", predictionCol = "prediction")
evaluator3 = RegressionEvaluator(metricName = "mae", labelCol = "rating", predictionCol = "prediction")

print("-------------------------------------- TASK A QUESTION 3 -------------------------------------------------------")
Rmse1 = []
Mse1 = []
Mae1 = []

Rmse2 = []
Mse2 = []
Mae2 = []
# -------------------- ALS SETTING 1 FOR 40% TRAINING DATA -------------------------------
model1 = als1.fit(training_data1)
predictions = model1.transform(test_data1)
rmse = evaluator1.evaluate(predictions)
mse = evaluator2.evaluate(predictions)
mae = evaluator3.evaluate(predictions)
Rmse1.append(rmse)
Mse1.append(mse)
Mae1.append(mae)

# ------------------- ALS SETTING 1 FOR 60% TRAINING DATA --------------------------------
model1 = als1.fit(training_data2)
predictions = model1.transform(test_data2)
rmse = evaluator1.evaluate(predictions)
mse = evaluator2.evaluate(predictions)
mae = evaluator3.evaluate(predictions)
Rmse1.append(rmse)
Mse1.append(mse)
Mae1.append(mae)
# ------------------- ALS SETTING 1 FOR 80% TRAINING DATA --------------------------------
model1 = als1.fit(training_data3)
predictions = model1.transform(test_data3)
rmse = evaluator1.evaluate(predictions)
mse = evaluator2.evaluate(predictions)
mae = evaluator3.evaluate(predictions)
Rmse1.append(rmse)
Mse1.append(mse)
Mae1.append(mae)





# -------------------- ALS SETTING 2 FOR 40% TRAINING DATA -------------------------------
model21 = als2.fit(training_data1)
predictions = model21.transform(test_data1)
rmse = evaluator1.evaluate(predictions)
mse = evaluator2.evaluate(predictions)
mae = evaluator3.evaluate(predictions)
Rmse2.append(rmse)
Mse2.append(mse)
Mae2.append(mae)

# -------------------- ALS SETTING 2 FOR 60% TRAINING DATA -------------------------------
model22 = als2.fit(training_data2)
predictions = model22.transform(test_data2)
rmse = evaluator1.evaluate(predictions)
mse = evaluator2.evaluate(predictions)
mae = evaluator3.evaluate(predictions)
Rmse2.append(rmse)
Mse2.append(mse)
Mae2.append(mae)

# -------------------- ALS SETTING 2 FOR 80% TRAINING DATA -------------------------------
model23 = als2.fit(training_data3)
predictions = model23.transform(test_data3)
rmse = evaluator1.evaluate(predictions)
mse = evaluator2.evaluate(predictions)
mae = evaluator3.evaluate(predictions)
Rmse2.append(rmse)
Mse2.append(mse)
Mae2.append(mae)

print(" RMSE For ALS Setting 1: ", Rmse1)
print("\n")
print(" MSE For ALS Setting 1: ", Mse1)
print("\n")
print(" MAE For ALS Setting 1: ", Mae1)
print("\n")

print(" RMSE For ALS Setting 2: ", Rmse2)
print("\n")
print(" MSE For ALS Setting 2: ", Mse2)
print("\n")
print(" MAE For ALS Setting 2: ", Mae2)
print("\n")



training_groups = ['40%','60%','80%']
plt.clf()
fig, ax = plt.subplots(figsize=(10,8))
bar_width = 0.08

# The below function is to add labels on the bar graph because it is difficult to spot the difference
def writeLabels(pps,pos):
    for p in pps:
        height = round(p.get_height(),3)
        ax.annotate('{}'.format(height),xy=(p.get_x() + p.get_width() / 2, height),xytext=(0,3),textcoords="offset points",ha=pos,va='bottom')

pps=ax.bar([i-bar_width*2.5 for i in range(len(training_groups))], Rmse1, width=bar_width, align='edge', label='Rmse For ALS Setting 1', color = 'red')
writeLabels(pps,'left')

pps=ax.bar([i-bar_width*1.5 for i in range(len(training_groups))], Mse1, width=bar_width, align='edge', label='MSE for ALS Setting 1', color = 'orange')
writeLabels(pps,'left')

pps=ax.bar([i-bar_width/2 for i in range(len(training_groups))], Mae1, width=bar_width, align='edge', label='Mae for ALS Setting 1', color = 'cyan')
writeLabels(pps,'left')

pps=ax.bar([i+bar_width/2 for i in range(len(training_groups))], Rmse2, width=bar_width, align='edge', label='RMSE for ALS Setting 2', color = 'green')
writeLabels(pps,'left')

pps=ax.bar([i+bar_width*1.5 for i in range(len(training_groups))], Mse2, width=bar_width, align='edge', label='MSE for ALS Setting 2', color = 'purple')
writeLabels(pps,'left')

pps=ax.bar([i+bar_width*2.5 for i in range(len(training_groups))], Mae2, width=bar_width, align='edge', label='Mae for ALS Setting 2', color = 'blue')
writeLabels(pps,'left')

ax.set_xticks(range(len(training_groups)))
ax.set_xticklabels(training_groups)
ax.set_xlim(-0.6, len(training_groups)-0.4)

y_min = min(min(Rmse1), min(Mse1), min(Mae1), min(Rmse2), min(Mse2), min(Mae2)) - 0.02
y_max = max(max(Rmse1), max(Mse1), max(Mae1), max(Rmse2), max(Mse2), max(Mae2)) + 0.02
ax.set_ylim(y_min,y_max)

ax.set_xlabel('Training Groups')
ax.set_ylabel('Errors')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1))



# Save the figure
plt.savefig('../Output/Recommendation_Performance.png', bbox_inches='tight')

print("------------------------------------ TASK B QUESTION 1 -----------------------------------------------")
cluster40 = []
cluster60 = []
cluster80 = []
# --------------------------------------------------- K-MEANS FOR 40% TRAINING DATA ---------------------------
# Getting the user factors from the second setting
userFactors_df1 = model21.userFactors
#userFactors_df.show(20,False)
# Defining K-means with K=25
kmeans = KMeans(k=25, seed=myseed, featuresCol="features")
clusterModel = kmeans.fit(userFactors_df1)
predictions1 = clusterModel.transform(userFactors_df1)
predictions1.show(10,True)

cluster_sizes = predictions1.groupBy('prediction').count() # Getting the cluster sizes
sorted_clusters = cluster_sizes.orderBy('count', ascending=False) # Sorting the clusters according to their size
top_5_clusters = sorted_clusters.take(5) # taking only the top-5 clusters
#print(top_5_clusters)
print(" For 40% training data, the size of each of top-5 clusters are as follows \n")
for cluster in top_5_clusters:
    print(f"Cluster {cluster['prediction']} has {cluster['count']} users.")
    cluster40.append(cluster['count'])
print("\n")
largest_cluster1 = top_5_clusters[0]['prediction'] # Recording the largest cluster
# --------------------------------------------------- K-MEANS FOR 60% TRAINING DATA ---------------------------
userFactors_df2 = model22.userFactors
#userFactors_df.show(20,False)
kmeans = KMeans(k=25, seed=myseed, featuresCol="features")
clusterModel = kmeans.fit(userFactors_df2)
predictions2 = clusterModel.transform(userFactors_df2)
predictions2.show(10,True)

cluster_sizes = predictions2.groupBy('prediction').count()
sorted_clusters = cluster_sizes.orderBy('count', ascending=False)
top_5_clusters = sorted_clusters.take(5)
print(top_5_clusters)
print(" For 60% training data, the size of each of top-5 clusters are as follows \n")
for cluster in top_5_clusters:
    print(f"Cluster {cluster['prediction']} has {cluster['count']} users.")
    cluster60.append(cluster['count'])
print("\n")
largest_cluster2 = top_5_clusters[0]['prediction']
# --------------------------------------------------- K-MEANS FOR 80% TRAINING DATA ---------------------------
userFactors_df3 = model23.userFactors
#userFactors_df.show(20,False)
kmeans = KMeans(k=25, seed=myseed, featuresCol="features")
clusterModel = kmeans.fit(userFactors_df3)
predictions3 = clusterModel.transform(userFactors_df3)
predictions3.show(10,True)

cluster_sizes = predictions3.groupBy('prediction').count()
sorted_clusters = cluster_sizes.orderBy('count', ascending=False)
top_5_clusters = sorted_clusters.take(5)
print(top_5_clusters)
print(" For 80% training data, the size of each of top-5 clusters are as follows \n")
for cluster in top_5_clusters:
    print(f"Cluster {cluster['prediction']} has {cluster['count']} users.")
    cluster80.append(cluster['count'])
print("\n")
largest_cluster3 = top_5_clusters[0]['prediction']


print("------------------------------------ TASK B QUESTION 2 -----------------------------------------------")

movie_data = spark.read.load('../Data/ml-20m/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()
# ----------------------------------------- LARGEST MOVIE CLUSTER AND TOP MOVIES FOR 40% TRAINING DATA ---------------------------

# Getting the user IDs in the largest cluster from the respective training set

largest_cluster_df = predictions1.filter(predictions1.prediction==largest_cluster1).select('id').withColumnRenamed('id', 'userId')
# Getting the movies rated by the users in the largest cluster

movies_largest_cluster = largest_cluster_df.join(training_data1,["userId"]).select('userId','movieId','rating')
# Computing the average rating for each movies rated by the users in the largest cluster 
movies_largest_cluster = movies_largest_cluster.groupBy("movieId").agg({"rating": "avg"}).withColumnRenamed("avg(rating)", "avg_rating")
print("Movies rated by users of largest cluster(for 40% training data) and their respective average ratings are: \n ")
movies_largest_cluster.show(20,True)
# Getting the top movies that have any average rating greater than or equal to 4
top_movies = movies_largest_cluster.filter(movies_largest_cluster.avg_rating>=4)
print("The top movies rated by users of largest cluster (for 40% training data) are \n")
top_movies.show(20,False)

#Getting the Genres for the top movies
genre_df = movie_data.join(top_movies,["movieId"]).select('movieId','title','genres')
#genre_df.show(10,False)
genre_df = genre_df.select(split(genre_df.genres, '\|').alias('genre')) # forming the list of genres from the tab-separated values
genre_df = genre_df.select(explode(genre_df.genre).alias('genre')) #Getting each genre as a new row
genre_df = genre_df.groupBy('genre').count().withColumnRenamed('count','genre_count').orderBy('genre_count',ascending=False) #Grouping the genres and counting
genre_df.show(20,False)
genre = genre_df.select('genre').rdd.flatMap(lambda x: x).collect()
print("Top 10 most genres for top_movies using the 40% training data are: \n")
print(genre[:10])

# ----------------------------------------- LARGEST MOVIE CLUSTER AND TOP MOVIES FOR 60% TRAINING DATA ---------------------------
largest_cluster_df = predictions2.filter(predictions2.prediction==largest_cluster2).select('id').withColumnRenamed('id', 'userId')
movies_largest_cluster = largest_cluster_df.join(training_data2,["userId"]).select('userId','movieId','rating')
movies_largest_cluster = movies_largest_cluster.groupBy("movieId").agg({"rating": "avg"}).withColumnRenamed("avg(rating)", "avg_rating")
print("Movies rated by users of largest cluster(for 60% training data) and their respective average ratings are: \n ")
movies_largest_cluster.show(20,True)
top_movies = movies_largest_cluster.filter(movies_largest_cluster.avg_rating>=4)
print("The top movies rated by users of largest cluster (for 60% training data) are \n")
top_movies.show(20,False)

genre_df = movie_data.join(top_movies,["movieId"]).select('movieId','title','genres')
#genre_df.show(10,False)
genre_df = genre_df.select(split(genre_df.genres, '\|').alias('genre'))
genre_df = genre_df.select(explode(genre_df.genre).alias('genre'))
genre_df = genre_df.groupBy('genre').count().withColumnRenamed('count','genre_count').orderBy('genre_count',ascending=False)
genre_df.show(20,False)
genre = genre_df.select('genre').rdd.flatMap(lambda x: x).collect()
print("Top 10 most genres for top_movies using the 60% training data are: \n")
print(genre[:10])
# ----------------------------------------- LARGEST MOVIE CLUSTER AND TOP MOVIES FOR 80% TRAINING DATA ---------------------------
largest_cluster_df = predictions3.filter(predictions3.prediction==largest_cluster3).select('id').withColumnRenamed('id', 'userId')
movies_largest_cluster = largest_cluster_df.join(training_data3,["userId"]).select('userId','movieId','rating')
movies_largest_cluster = movies_largest_cluster.groupBy("movieId").agg({"rating": "avg"}).withColumnRenamed("avg(rating)", "avg_rating")
print("Movies rated by users of largest cluster(for 80% training data) and their respective average ratings are: \n ")
movies_largest_cluster.show(20,True)
top_movies = movies_largest_cluster.filter(movies_largest_cluster.avg_rating>=4)
print("The top movies rated by users of largest cluster (for 80% training data) are \n")
top_movies.show(20,False)

genre_df = movie_data.join(top_movies,["movieId"]).select('movieId','title','genres')
#genre_df.show(10,False)
genre_df = genre_df.select(split(genre_df.genres, '\|').alias('genre'))
genre_df = genre_df.select(explode(genre_df.genre).alias('genre'))
genre_df = genre_df.groupBy('genre').count().withColumnRenamed('count','genre_count').orderBy('genre_count',ascending=False)
genre_df.show(20,False)
genre = genre_df.select('genre').rdd.flatMap(lambda x: x).collect()
print("Top 10 most genres for top_movies using the 80% training data are: \n")
print(genre[:10])

plt.clf()
fig, ax = plt.subplots(figsize=(10,8))
bar_width = 0.08

#
lst1 = [cluster40[0],cluster60[0],cluster80[0]]
lst2 = [cluster40[1],cluster60[1],cluster80[1]]
lst3 = [cluster40[2],cluster60[2],cluster80[2]]
lst4 = [cluster40[3],cluster60[3],cluster80[3]]
lst5 = [cluster40[4],cluster60[4],cluster80[4]]

pps=ax.bar([i-bar_width*2.5 for i in range(len(training_groups))], lst1, width=bar_width, align='edge', color = 'red')
writeLabels(pps,'center')

pps=ax.bar([i-bar_width*1.5 for i in range(len(training_groups))], lst2, width=bar_width, align='edge', color = 'orange')
writeLabels(pps,'center')

pps=ax.bar([i-bar_width/2 for i in range(len(training_groups))], lst3, width=bar_width, align='edge', color = 'cyan')
writeLabels(pps,'center')

pps=ax.bar([i+bar_width/2 for i in range(len(training_groups))], lst4, width=bar_width, align='edge', color = 'green')
writeLabels(pps,'center')

pps=ax.bar([i+bar_width*1.5 for i in range(len(training_groups))], lst5, width=bar_width, align='edge', color = 'purple')
writeLabels(pps,'left')


ax.set_xticks(range(len(training_groups)))
ax.set_xticklabels(training_groups)
ax.set_xlim(-0.6, len(training_groups)-0.4)

y_min = min(min(cluster40),min(cluster60),min(cluster80)) - 500
y_max = max(max(cluster40),max(cluster60),max(cluster80)) + 500
ax.set_ylim(y_min,y_max)

ax.set_xlabel('Training Groups')
ax.set_ylabel('No. of Users')

# Saving the figure 
plt.savefig('../Output/Cluster_Analysis.png')
