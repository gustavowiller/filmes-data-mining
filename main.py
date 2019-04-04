# -*- coding: utf-8 -*-

#http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import operator
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

def main(k=4,i=500,m='braycurtis',nc=20):

	#Prepara os dados
	dt_rating = pd.read_csv('ml-100k/u.data',sep='\t',header=None).groupby(1,as_index=False).mean()[[2]]
	dt_item = pd.read_csv('ml-100k/u.item',sep='|',header=None)	 
	dt_movie = pd.concat([dt_item, dt_rating],axis=1) 

	#Remove o filme de teste (i) da instancia de filmes dt_movie
	dt_movie_test = dt_movie.iloc[i,:] 
	dt_movie.drop(i,inplace=True)
	
	#Pega as características de genero dos filmes
	list_genre_test = dt_movie_test[5:24].values.reshape(1,-1)
	list_genre = dt_movie.iloc[:,5:24]

	#Kmeans
	kmeans = KMeans(n_clusters=nc, random_state=0).fit(list_genre)
	#print kmeans.cluster_centers_

	#Knn
	nbrs = NearestNeighbors(n_neighbors=k,algorithm='auto',metric=m).fit(list_genre)
	#With Kmeans
	#nbrs = NearestNeighbors(n_neighbors=k,algorithm='auto',metric=m).fit(kmeans.cluster_centers_)
		
	#Knn
	distances, indices = nbrs.kneighbors(list_genre_test)
	#With Kmeans
	#distances, indices = nbrs.kneighbors(kmeans.predict(list_genre_test))

	
	#Avalia o Erro
	predict = dt_movie.iloc[indices[0],24].mean()
	real = dt_movie_test.iloc[24]
	erro = abs(predict - real)
	return erro

