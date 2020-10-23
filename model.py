import pymongo
from pymongo import MongoClient
from datetime import timedelta, datetime
import time
import itertools as it
from hmmlearn import hmm
import numpy as np
from threading import Thread

#TODO - validar observacoes - verificar ordem das obs com latencia
#gravar observacoes completas com data num csv para facilitar validacao
#validar deixando fixo o tamanho do periodo - iniciar teste pelos maiores
#por logs
#threads
#14368s processo total
#13227 total com 4 threads
#12383s total com 10 thread
#procurar saber a base em relacao do AIC e BIC
#AIC generica na teoria da selecao de modelo

#Divisors for determinate periods sizes
SMALL = 10
MEDIUM = 5
BIG = 1
threads = []	
thread_group_size = 4




class Forecast:
	def __init__(self, paper, day_for_forecasting):
		self.paper = paper
		self.day_for_forecasting = day_for_forecasting
	
	def load_observations_full(self,paper):
		client = MongoClient()
		db = client.stock

		collection = db.papers

		cursor = collection.find(
		 	{"codneg": paper, 
		 	"tpmerc" : "010", 
		 		"date": {
		         "$gte": datetime(2009, 1, 1),
		         "$lte": datetime(2019, 12, 31)
		        }
		    }
		    ).sort([("date", pymongo.ASCENDING)])
		observations_list = list(cursor)
		
		return observations_list
		
	def train_model(self,params_group, observations):
		for params in params_group:
			model = Model(params, observations)
			model.train()

	def execute(self):
		start = time.time()
		self.params = {}
		self.params["period_sizes"] = [SMALL, MEDIUM, BIG]
		self.params["number_hidden_states"] = list(range(3,15))
		self.params["latency"] = list(range(1,100))
		#TODO falta incluir os parametros:
		#formato observacoes
		#fdp

		observations = self.load_observations_full(self.paper)
		observations_formatted = np.array(list(map(lambda x: [x["preult"], x["premin"], x["premax"], x["preab"]], observations)))
		observations_formatted = np.flipud(observations_formatted)

		#print("VALIDACAO observacoes completa %s" % observations_formatted)
		
		params_combinations = it.product(*(self.params[Name] for Name in self.params))

		#Falta variar:
		#Formato das observacoes (inteiro, fracoes de mudanca)
		#Funcao distribuicao de probabilidade (Gaussiana,Gamma)
		params_combinations_grouped = np.array_split(np.array(list(params_combinations)),thread_group_size)

		for params_group in params_combinations_grouped:
		    t = Thread(target=self.train_model, args=(params_group,observations_formatted, ))
		    t.start()
		    threads.append(t)
		
		for t in threads:
			t.join()
		
		end = time.time()
		print(end-start)

	

	


class Model:
	
	def __init__(self, params, observations):
		self.period_size = params[0]
		self.number_hidden_states = params[1]
		self.latency = params[2]
		self.observations = observations[:(observations.shape[0] // self.period_size)]


	def train(self):
		#print("treinando com os parametros periodo %s estados ocultos %s latencia %s" % (self.period_size, self.number_hidden_states, self.latency))
		current_observation = np.array(self.observations[0:self.latency])
		
		#print("tamanho observacoes %s" % self.observations.shape[0])
		#print("VALIDACAO observacoes parciais %s" % self.observations)
		#print("VALIDACAO inicio periodo %s" % self.observations[-1])
		#print("VALIDACAO fim periodo %s" % self.observations[0])
		#print("VALIDACAO obsercacao atual %s" % current_observation)
		NUM_ITERS=10000
		model = hmm.GaussianHMM(n_components=self.number_hidden_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS, init_params='tmc')
		model.fit(np.flipud(self.observations))
		model.score(np.flipud(current_observation))
			#print(n)

	def valid(self):
		print("method VALid")


f = Forecast('ITUB4', '02/10/2020')
f.execute()