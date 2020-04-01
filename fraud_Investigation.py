
x=np.asmatrix(fraud_df[['TOTAL_SPEND','DOCTOR_VISITS','NUM_CLAIMS','MEMBER_DURATION','OPTOM_PRESC','NUM_MEMBERS']])

xtx = x.transpose() * x
print("t(x) * x = \n", xtx)

evals, evecs = LA.eigh(xtx)
print("Eigenvalues of x = \n", evals)
print("Eigenvectors of x = \n",evecs)

# Here is the transformation matrix
transf = evecs * LA.inv(np.sqrt(np.diagflat(evals)));
print("Transformation Matrix = \n", transf)

# Here is the transformed X
transf_x = x * transf;
print("The Transformed x = \n", transf_x)

# Check columns of transformed X
xtx = transf_x.transpose() * transf_x;
print("Expect an Identity Matrix = \n", xtx)

from sklearn.neighbors import NearestNeighbors as kNN

kNNSpec = kNN(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')

trainData=transf_x

nbrs = kNNSpec.fit(trainData)
distances, indices = nbrs.kneighbors(trainData)


from sklearn.neighbors import KNeighborsClassifier
target=fraud_df['FRAUD']

neigh = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs = neigh.fit(trainData, target)

accuracy = nbrs.score(trainData, target)
print("The answer to Question 3-D:\n ")
print(accuracy)
print("\n")
########################

obs=[[7500,15,3,127,2,2]]
focal = obs * transf
print(focal)
print("\n")
myNeighbors = nbrs.kneighbors(focal, return_distance = False)
print("My Neighbors = ", myNeighbors)
print("\n")
########################


res=nbrs.predict(focal)
res_prob=nbrs.predict_proba(focal)
print(res_prob)
