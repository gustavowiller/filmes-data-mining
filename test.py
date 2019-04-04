# -*- coding: utf-8 -*-


import main
import numpy as np

def test():

	X = range(1,1602,100) #Conjunto de amostra de teste da base de dados de filmes. 
	K = range(3,4)	#Valores para K Nearest Neighbors
	M =  ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']


	#Choice the best
	#3 - canberra - 0.444 - 0.522
	K = range(3,4)
	#M = ['canberra']

	for k in K:
		for m in M:
			error = []
			for i in X:
				error.append( main.main(k,i,m,20) )
			print str(k) + ' - ' + str(m) + ' - '+str(np.mean(error).round(3)) + " - " + str(np.std(error).round(3))
		#plt.plot(X,error)
		#plt.show()

test()
