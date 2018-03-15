import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def runsim(
	mean_infectious_duration = 10, 	         		# year
	transmissibility = .6,                         		# infection / contact
	contact_factor = .8,                          		# (contact / year) / person
	mean_incubation = 9,                               	# year
	mean_quarantine = 15,					# year
	mean_death = 30,                                 	# year
	plot = True):
	
	recovery_factor = 1 / mean_infectious_duration    	# 1 / year
	incubation_factor = 1 / mean_incubation                      	# 1 / year
	quarantine_factor = 1 / mean_quarantine                 # 1 / year
	death_factor = 1 / mean_death	                        # 1 / year

	delta_t = .1                                    	# year
	time_values = np.arange(2018, 2168, delta_t)

	S = np.empty(len(time_values))
	E = np.empty(len(time_values))
	Q = np.empty(len(time_values))
	I = np.empty(len(time_values))
	R = np.empty(len(time_values))
	D = np.empty(len(time_values))

	total_pop = 10000
	
	S[0] = total_pop - 100
	E[0] = 0
	Q[0] = 0
	I[0] = 100
	R[0] = 0
	D[0] = 0

	for i in range(1, len(time_values)):

		frac_susceptible =  S[i - 1] / (total_pop - Q[i - 1] - D[i - 1])		
		
		SI_contact_rate = I[i - 1] * frac_susceptible * contact_factor 		# contacts / year
		
		exposed_rate = SI_contact_rate * transmissibility			# people / year
		
		quarantine_rate = E[i - 1] * quarantine_factor				# people / year
		quarantine_recovery_rate = Q[i - 1] * recovery_factor  			# recoveries / year
		quarantine_death_rate = Q[i - 1] * death_factor				# deaths / year
		
		infectious_rate = E[i - 1] * incubation_factor			# infections / year
		infection_recovery_rate = I[i - 1] * recovery_factor  			# recoveries / year
		infection_death_rate = I[i - 1] * death_factor				# deaths / year

		# Primes.
		S_prime = -exposed_rate
		E_prime = exposed_rate - infectious_rate - quarantine_rate
		Q_prime = quarantine_rate - quarantine_recovery_rate - quarantine_death_rate
		I_prime = infectious_rate - infection_recovery_rate - infection_death_rate
		R_prime = infection_recovery_rate + quarantine_recovery_rate
		D_prime = infection_death_rate + infection_death_rate

		# Stocks.
		S[i] = S[i - 1] + S_prime * delta_t
		E[i] = E[i - 1] + E_prime * delta_t
		Q[i] = Q[i - 1] + Q_prime * delta_t
		I[i] = I[i - 1] + I_prime * delta_t
		R[i] = R[i - 1] + R_prime * delta_t
		D[i] = D[i - 1] + D_prime * delta_t

	if plot:
		plt.plot(time_values, S, color="blue", label="S")
		plt.plot(time_values, E, color="purple", label="E")
		plt.plot(time_values, Q, color="red", label="Q")
		plt.plot(time_values, I, color="green", label="I")
		plt.plot(time_values, R, color="orange", label="R")
		plt.plot(time_values, D, color="black", label="D")
		plt.xlabel("Time (Years)")
		plt.ylabel("Population (People)")
		plt.legend()
		plt.show()
		plt.savefig("plot1.png")

	return (total_pop - S[len(S) - 1]) / total_pop * 100


def main():
	mids = np.arange(.1, 10, .1)
	perc_infs = []

	#runsim(plot = True)
	runsim(plot = True)

	for mid in mids:
		perc_inf = runsim(mid, plot = False)
		perc_infs.append(perc_inf)

	plt.clf()
	plt.plot(mids, perc_infs, color="black", label="% Infected")
	plt.xlabel("Mean Infected Duration (Years)")
	plt.ylabel("% Ever Infected")
	plt.legend()
	plt.show()
	plt.savefig("plot2.png")

main()
