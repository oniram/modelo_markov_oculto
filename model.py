import pymongo
from pymongo import MongoClient
from datetime import timedelta, datetime
import time
import itertools as it
from hmmlearn import hmm
import numpy as np
from threading import Thread
import ipdb #debug
import csv

#TODO:

SMALL = 245 #Numero de dias em 2018
MEDIUM = 986 #Numero de dias entre 2014 e 2018
BIG = 1977 #Numero de dias entre 2010 e 2018
MEDIA_OBS_BY_YEAR = 262
VALID_DATA_SIZE=248
TRAIN_DATA_SIZE=1977
threads = []	
thread_group_size = 1




class Forecast:
	def __init__(self, paper, day_for_forecasting):
		self.paper = paper
		self.day_for_forecasting = day_for_forecasting
	
	#Carrega do mongo dados de 10 anos e mantem na memoria para evitar custo de novas buscas
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
		    ).sort([("date", pymongo.DESCENDING)])
		observations_list = list(cursor)
		return observations_list
		
	#Treina um grupo de modelos - util para quando for rodar em paralelo
	def train_group_model(self,params_group, observations):
		for params in params_group:
			model = Model(params, observations)
			model.train()
			model.valid()

	def execute(self):
		
		self.params = {}
		self.params["period_sizes"] = [BIG]
		self.params["number_hidden_states"] = [6]#[2,3,4,8,10,12]
		self.params["latency"] = [1]
		self.params["paper"] = [self.paper]
		

		observations = self.load_observations_full(self.paper)
		observations_formatted = np.array(list(map(lambda x: [x["preult"], x["premin"], x["premax"], x["preab"]], observations)))
		
		#carregar dados de observacoes de um csv
		#observations_formatted = np.genfromtxt("LUV_FULL.csv", delimiter=',')

		
		#Faz uma combinacao de todos parametros do modelo
		params_combinations = it.product(*(self.params[Name] for Name in self.params))
		
		#separa combinações de parametros em mais de um grupo, caso for rodar em paralelo
		params_combinations_grouped = np.array_split(np.array(list(params_combinations)),thread_group_size)

		#Ira treinar o modelo em paralelo caso seja definido thread_group_size > 1
		for params_group in params_combinations_grouped:
		    t = Thread(target=self.train_group_model, args=(params_group,observations_formatted, ))
		    t.start()
		    threads.append(t)
		
		for t in threads:
			t.join()
	


class Model:
	
	def __init__(self, params, observations):
		self.period_size = int(params[0])
		TRAIN_DATA_SIZE = self.period_size
		self.number_hidden_states = int(params[1])
		self.latency = int(params[2])
		self.paper = params[3]
		self.observations = observations
		
		#define as janelas de treinamento e validação de acordo os tamanhos definidos
		self.valid_data = observations[0:VALID_DATA_SIZE]
		self.train_data = observations[VALID_DATA_SIZE:VALID_DATA_SIZE+TRAIN_DATA_SIZE]
		NUM_ITERS=10000

		#Inicia o modelo - dados aleatorios definem as matrizes
		self.model = hmm.GaussianHMM(n_components=self.number_hidden_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS, init_params='tmc')
		self.start_time = time.time()

	def train(self):
		print("treinando com os parametros periodo %s estados ocultos %s latencia %s" % (self.period_size, self.number_hidden_states, self.latency))
		
		#print("tamanho observacoes %s" % self.observations.shape[0])
		#print("VALIDACAO observacoes parciais %s" % self.observations)
		#print("VALIDACAO inicio periodo %s" % self.observations[-1])
		#print("VALIDACAO fim periodo %s" % self.observations[0])
		#print("VALIDACAO obsercacao atual %s" % current_observation)
		
		

	def forecast(self):
		
		NUM_ITERS=10000
		
		self.model = hmm.GaussianHMM(n_components=self.number_hidden_states, covariance_type='full', tol=0.0001, n_iter=NUM_ITERS, init_params='tmc')
		#otimiza o modelo
		self.model.fit(np.flipud(self.train_data)) 
		likelihoods = []
		current_observation = np.array(self.train_data[0:self.latency][0])
		
		#Calcula likelihood de cada observação do periodo de treinamento
		for i in range(self.train_data.shape[0]):
			likelihoods.append(self.model.score(np.array(self.train_data[i:i+self.latency])))
		current_likelihood = likelihoods.pop()
		
		diff_likelihods_most_near = min(likelihoods, key=lambda x:abs(x-current_likelihood))
		diff_likelihods_most_near_index = likelihoods.index((diff_likelihods_most_near))
		
		#Faz a previsão conforme Hassan
		close_price_match_past = self.train_data[diff_likelihods_most_near_index, :][0]
		next_close_price_match_past = self.train_data[diff_likelihods_most_near_index+1, :][0]
		diff_to_add_in_forecast = abs(close_price_match_past - next_close_price_match_past)
		close_price_forecasted = current_observation + diff_to_add_in_forecast
		# print("current_observation %s" % current_observation)
		# print("close_price_match_past index %s" % diff_likelihods_most_near_index)
		# print("close_price_match_past %s" % close_price_match_past)
		# print("next_close_price_match_past %s" % next_close_price_match_past)
		# print("close_price_forecasted %s" % close_price_forecasted)
		#exit()
		return np.array([close_price_forecasted])
		

	#Faz a validação do modelo executando a previsão para cada dia do periodo de validação
	def valid(self):
		forecasts = np.zeros(shape=(0,4))
		for i in range(self.valid_data.shape[0]):
			current_forecast = self.forecast()
			forecasts = np.concatenate((forecasts, current_forecast))
			#Move a janela de treinamento para a proxima previsão
			self.train_data = self.observations[VALID_DATA_SIZE-1-i:(VALID_DATA_SIZE+TRAIN_DATA_SIZE)-i-1]
		
		forecasts = np.flipud(forecasts)
		self.end_time = time.time()
		total_time = self.end_time - self.start_time
		
		#np.savetxt('validacoes.csv',self.valid_data,delimiter=',',fmt='%.2f')
		#print(self.calc_mape(forecasts,self.valid_data))
		
		#Calcula o MAPE das previsões do periodo de validação
		sum_error_forecast = 0
		for i in range(self.valid_data.shape[0]):
			error_forecast = (abs(self.valid_data[i][0] - forecasts[i][0]))/self.valid_data[i][0]
			sum_error_forecast += error_forecast

		mape = (sum_error_forecast * (1/self.valid_data.shape[0])) *100

		#Grava os resultados da validação de uma ação num arquivo
		with open('resultados.csv', mode='a+') as result_file:
			result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			result_writer.writerow([self.paper, mape, total_time,VALID_DATA_SIZE, TRAIN_DATA_SIZE, self.number_hidden_states])


#Inicia arquivo onde gravaremos resultados
with open('resultados.csv', mode='a+') as result_file:
	result_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
	result_writer.writerow(["ACAO", "MAPE(PORCENTAGEM)", "TEMPO EXECUCAO(SEGUNDOS)", "TAMANHO VALIDACAO", "TAMANHO TREINAMENTO", "ESTADOS"])


#Definie ações que iremos rodar no modelo
papers =  ["AALR3","ABEV3","ALPA4","AMAR3","ANIM3","ARZZ3","BEEF3","BKBR3","BRFS3","BTOW3","CAML3","CEAB3","CNTO3","COGN3","CRFB3","CVCB3","CYRE3","DIRR3","EVEN3","EZTC3","FLRY3","GFSA3","GNDI3","GRND3","GUAR3","HAPV3","HBOR3","HGTX3","HYPE3","JBSS3","JHSF3","LAME3","LAME4","LCAM3","LEVE3","LREN3","MDIA3","MEAL3","MGLU3","MOVI3","MRFG3","MRVE3","MYPK3","NTCO3","ODPV3","PARD3","PCAR3","QUAL3","RADL3","RENT3","SEER3","SLCE3","SMLS3","SMTO3","TCSA3","TEND3","TRIS3","VIVA3","VULC3","VVAR3","YDUQ3"]
papers += ["ALSO3","BRML3","BRPR3","CYRE3","DIRR3","EVEN3","EZTC3","GFSA3","HBOR3","IGTA3","JHSF3","LOGG3","LPSB3","MRVE3","MULT3","TCSA3","TEND3","TRIS3"]
papers += ["ABEV3","ALPA4","BEEF3","BRFS3","BRKM5","CAML3","CSNA3","CYRE3","DIRR3","DTEX3","EMBR3","EVEN3","EZTC3","GFSA3","GGBR4","GOAU4","GRND3","HBOR3","HGTX3","JBSS3","JHSF3","KLBN11","MDIA3","MRFG3","MRVE3","MYPK3","NTCO3","POMO4","POSI3","RAPT4","SMTO3","SUZB3","TCSA3","TEND3","TRIS3","TUPY3","USIM5","VIVA3","WEGE3"]
papers += ["ALUP11","CESP6","CMIG4","COCE5","CPFE3","CPLE6","EGIE3","ELET3","ENBR3","ENEV3","ENGI11","EQTL3","LIGT3","NEOE3","OMGE3","TAEE11","TIET11","TRPL4"]

#Inicia a previsão para cada ação definida anteriormente
for paper_code in (papers):
	start = time.time()
	f = Forecast(paper_code, '02/10/2020')
	f.execute()
	end = time.time()
	total = end -start
	print("time for %s: %s" % (paper_code, total))
