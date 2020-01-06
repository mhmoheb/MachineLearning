import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.metrics as metrics
import sklearn.linear_model as linear_model


ChicagoDiabetes = pandas.read_csv('ChicagoDiabetes.csv', delimiter=',')

# Feature variables
X = ChicagoDiabetes[['Crude Rate 2000','Crude Rate 2001','Crude Rate 2002','Crude Rate 2003',
                  'Crude Rate 2004', 'Crude Rate 2005', 'Crude Rate 2006', 'Crude Rate 2007',
                  'Crude Rate 2008','Crude Rate 2009', 'Crude Rate 2010', 'Crude Rate 2011']]

nObs = X.shape[0]
nVar = X.shape[1]


  
# Calculate the Correlations among the variables
XCorrelation = X.corr(method = 'pearson', min_periods = 1)

print('Empirical Correlation: \n', XCorrelation)

# Extract the Principal Components
_thisPCA = decomposition.PCA(n_components = nVar)
_thisPCA.fit(X)

#a
min(len(X), len(X.columns))



#b
plt.plot(_thisPCA.explained_variance_ratio_, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Explained Variance')
plt.xticks(numpy.arange(0,nVar))
plt.axhline((1/nVar), color = 'r', linestyle = '--')
plt.grid(True)
plt.show()
#c
cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)
plt.plot(cumsum_variance_ratio, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(numpy.arange(0,nVar))
plt.grid(True)
plt.show()

#d
cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)

print('Explained Variance: \n', _thisPCA.explained_variance_)
print('Explained Variance Ratio: \n', _thisPCA.explained_variance_ratio_)
print('Cumulative Explained Variance Ratio: \n', cumsum_variance_ratio)
print('Principal Components: \n', _thisPCA.components_)

#e
first2PC = _thisPCA.components_[:, [0,1]]
print('Principal COmponent: \n', first2PC)

#  Transform the data using the first two principal components
_thisPCA = decomposition.PCA(n_components = 2)
X_transformed = pandas.DataFrame(_thisPCA.fit_transform(X))
X_transformed
# Find clusters from the transformed data
maxNClusters = 10

nClusters = numpy.zeros(maxNClusters-1)
Elbow = numpy.zeros(maxNClusters-1)
Silhouette = numpy.zeros(maxNClusters-1)
TotalWCSS = numpy.zeros(maxNClusters-1)
Inertia = numpy.zeros(maxNClusters-1)

for c in range(maxNClusters-1):
   KClusters = c + 2
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=20190405 ).fit(X_transformed)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (KClusters > 1):
       Silhouette[c] = metrics.silhouette_score(X_transformed, kmeans.labels_)
   else:
       Silhouette[c] = float('nan')

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nObs):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = X_transformed.iloc[i,] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += (WCSS[k] / nC[k])
      TotalWCSS[c] += WCSS[k]

   print("The", KClusters, "Cluster Solution Done")

print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(maxNClusters-1):
   print('\t{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))
#F
# Draw the Elbow and the Silhouette charts  
plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(2, maxNClusters, 1))
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(2, maxNClusters, 1))
plt.show()

# Fit the 4 cluster solution'
kmeans = cluster.KMeans(n_clusters=4, random_state=20190405).fit(X_transformed)
X_transformed['Cluster ID'] = kmeans.labels_


# Draw the first two PC using cluster label as the marker color 
carray = ['red', 'orange', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(4):
    subData = X_transformed[X_transformed['Cluster ID'] == i]
    plt.scatter(x = subData[0],
                y = subData[1], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.axis(aspect = 'equal')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.axis(aspect = 'equal')
plt.legend(title = 'Cluster ID', fontsize = 12, markerscale = 2)
plt.show()
###########################################################
# g List the names of the communities in each cluster.

#ChicagoDiabetes.label.value_counts()

new_X = X.copy()
new_X['Cluster ID'] = kmeans.labels_
print(new_X)
new_X['Community'] = ChicagoDiabetes['Community']


print(new_X.loc[new_X['Cluster ID'] == 0])
print(new_X.loc[new_X['Cluster ID'] == 1])
print(new_X.loc[new_X['Cluster ID'] == 2])
print(new_X.loc[new_X['Cluster ID'] == 3])
#############################################################
#h
#####	Calculate the annual total population and crude rate

ChicagoDiabetes['label']=kmeans.labels_
ChicagoDiabetes['pop_2000']=(ChicagoDiabetes['Hospitalizations 2000']*10000)/ChicagoDiabetes['Crude Rate 2000']
ChicagoDiabetes['pop_2001']=(ChicagoDiabetes['Hospitalizations 2001']*10000)/ChicagoDiabetes['Crude Rate 2001']
ChicagoDiabetes['pop_2002']=(ChicagoDiabetes['Hospitalizations 2002']*10000)/ChicagoDiabetes['Crude Rate 2002']
ChicagoDiabetes['pop_2003']=(ChicagoDiabetes['Hospitalizations 2003']*10000)/ChicagoDiabetes['Crude Rate 2003']
ChicagoDiabetes['pop_2004']=(ChicagoDiabetes['Hospitalizations 2004']*10000)/ChicagoDiabetes['Crude Rate 2004']
ChicagoDiabetes['pop_2005']=(ChicagoDiabetes['Hospitalizations 2005']*10000)/ChicagoDiabetes['Crude Rate 2005']
ChicagoDiabetes['pop_2006']=(ChicagoDiabetes['Hospitalizations 2006']*10000)/ChicagoDiabetes['Crude Rate 2006']
ChicagoDiabetes['pop_2007']=(ChicagoDiabetes['Hospitalizations 2007']*10000)/ChicagoDiabetes['Crude Rate 2007']
ChicagoDiabetes['pop_2008']=(ChicagoDiabetes['Hospitalizations 2008']*10000)/ChicagoDiabetes['Crude Rate 2008']
ChicagoDiabetes['pop_2009']=(ChicagoDiabetes['Hospitalizations 2009']*10000)/ChicagoDiabetes['Crude Rate 2009']
ChicagoDiabetes['pop_2010']=(ChicagoDiabetes['Hospitalizations 2010']*10000)/ChicagoDiabetes['Crude Rate 2010']
ChicagoDiabetes['pop_2011']=(ChicagoDiabetes['Hospitalizations 2011']*10000)/ChicagoDiabetes['Crude Rate 2011']
ChicagoDiabetes

ChicagoDiabetes_new_col=['Hospitalizations 2000','Hospitalizations 2001','Hospitalizations 2002','Hospitalizations 2003',
                                  'Hospitalizations 2004','Hospitalizations 2005','Hospitalizations 2006','Hospitalizations 2007', 
                                  'Hospitalizations 2008','Hospitalizations 2009', 'Hospitalizations 2010','Hospitalizations 2011', 'pop_2000',
                                  'pop_2001', 'pop_2002', 'pop_2003', 'pop_2004', 'pop_2005', 'pop_2006',
                                  'pop_2007', 'pop_2008', 'pop_2009', 'pop_2010', 'pop_2011']

c0=ChicagoDiabetes.groupby('label').get_group(0).sum()[ChicagoDiabetes_new_col]
c1=ChicagoDiabetes.groupby('label').get_group(1).sum()[ChicagoDiabetes_new_col]
c2=ChicagoDiabetes.groupby('label').get_group(2).sum()[ChicagoDiabetes_new_col]
c3=ChicagoDiabetes.groupby('label').get_group(3).sum()[ChicagoDiabetes_new_col]

x=pandas.DataFrame(columns=['cluster_0','cluster_1','cluster_2','cluster_3'])
x['cluster_0']=c0
x['cluster_1']=c1
x['cluster_2']=c2
x['cluster_3']=c3
x_Table=x.T
x_Table

x_Table['crudHos_2000']=(x_Table['Hospitalizations 2000']*10000)/x_Table['pop_2000']
x_Table['crudHos_2001']=(x_Table['Hospitalizations 2001']*10000)/x_Table['pop_2001']
x_Table['crudHos_2002']=(x_Table['Hospitalizations 2002']*10000)/x_Table['pop_2002']
x_Table['crudHos_2003']=(x_Table['Hospitalizations 2003']*10000)/x_Table['pop_2003']
x_Table['crudHos_2004']=(x_Table['Hospitalizations 2004']*10000)/x_Table['pop_2004']
x_Table['crudHos_2005']=(x_Table['Hospitalizations 2005']*10000)/x_Table['pop_2005']
x_Table['crudHos_2006']=(x_Table['Hospitalizations 2006']*10000)/x_Table['pop_2006']
x_Table['crudHos_2007']=(x_Table['Hospitalizations 2007']*10000)/x_Table['pop_2007']
x_Table['crudHos_2008']=(x_Table['Hospitalizations 2008']*10000)/x_Table['pop_2008']
x_Table['crudHos_2009']=(x_Table['Hospitalizations 2009']*10000)/x_Table['pop_2009']
x_Table['crudHos_2010']=(x_Table['Hospitalizations 2010']*10000)/x_Table['pop_2010']
x_Table['crudHos_2011']=(x_Table['Hospitalizations 2011']*10000)/x_Table['pop_2011']
R_x_Table = x_Table.T
R_x_Table



Chicago_crude_2000=ChicagoDiabetes['Hospitalizations 2000'].sum()*10000/ChicagoDiabetes['pop_2000'].sum()
Chicago_crude_2001=ChicagoDiabetes['Hospitalizations 2001'].sum()*10000/ChicagoDiabetes['pop_2001'].sum()
Chicago_crude_2002=ChicagoDiabetes['Hospitalizations 2002'].sum()*10000/ChicagoDiabetes['pop_2002'].sum()
Chicago_crude_2003=ChicagoDiabetes['Hospitalizations 2003'].sum()*10000/ChicagoDiabetes['pop_2003'].sum()
Chicago_crude_2004=ChicagoDiabetes['Hospitalizations 2004'].sum()*10000/ChicagoDiabetes['pop_2004'].sum()
Chicago_crude_2005=ChicagoDiabetes['Hospitalizations 2005'].sum()*10000/ChicagoDiabetes['pop_2005'].sum()
Chicago_crude_2006=ChicagoDiabetes['Hospitalizations 2006'].sum()*10000/ChicagoDiabetes['pop_2006'].sum()
Chicago_crude_2007=ChicagoDiabetes['Hospitalizations 2007'].sum()*10000/ChicagoDiabetes['pop_2007'].sum()
Chicago_crude_2008=ChicagoDiabetes['Hospitalizations 2008'].sum()*10000/ChicagoDiabetes['pop_2008'].sum()
Chicago_crude_2009=ChicagoDiabetes['Hospitalizations 2009'].sum()*10000/ChicagoDiabetes['pop_2009'].sum()
Chicago_crude_2010=ChicagoDiabetes['Hospitalizations 2010'].sum()*10000/ChicagoDiabetes['pop_2010'].sum()
Chicago_crude_2011=ChicagoDiabetes['Hospitalizations 2011'].sum()*10000/ChicagoDiabetes['pop_2011'].sum()

pandas.DataFrame([Chicago_crude_2000,Chicago_crude_2001,Chicago_crude_2002,Chicago_crude_2003,Chicago_crude_2004,Chicago_crude_2005,
                 Chicago_crude_2006,Chicago_crude_2007,Chicago_crude_2008,Chicago_crude_2009,Chicago_crude_2010,Chicago_crude_2011],
               index=[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011],columns=['Chicago'])
carray = ['blue','green', 'yello','orange']
year = ['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011']
rate = [25.4, 25.8, 27.2, 25.4, 26.2, 26.6, 27.4, 28.7, 27.9, 27.5, 26.8, 25.6]

df_chicago=pandas.DataFrame(rate,index=year,columns=['Chicago'])
df_chicago.mean()
print(df_chicago)

### i Plot the crude hospitalization rates in each cluster against the years
plt.figure(figsize=(10,6))
plt.plot([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011],R_x_Table['cluster_0'].filter(like='crudHos'),color='blue',label='Cl0')
plt.plot([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011],R_x_Table['cluster_1'].filter(like='crudHos'),color='green',label='Cl1')
plt.plot([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011],R_x_Table['cluster_2'].filter(like='crudHos'),color='yellow',label='Cl2')
plt.plot([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011],R_x_Table['cluster_3'].filter(like='crudHos'),color='orange',label='Cl3')
plt.plot([2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011],[Chicago_crude_2000,Chicago_crude_2001,Chicago_crude_2002,Chicago_crude_2003,Chicago_crude_2004,
          Chicago_crude_2005,Chicago_crude_2006,Chicago_crude_2007,Chicago_crude_2008,Chicago_crude_2009,Chicago_crude_2010,Chicago_crude_2011],color='red',label='Chicago',marker='*')
plt.legend(loc='upper left')
plt.xlabel('Year')
plt.ylabel('Crude Rate')
plt.grid(True)












