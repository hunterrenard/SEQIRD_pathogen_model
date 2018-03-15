import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def runsim(
	mean_infectious_duration = 10, 	         		# day
	transmissibility = .6,                         		# infection / contact
	contact_factor = .8,                          		# (contact / day) / person
	mean_incubation = 9,                               	# day
	mean_quarantine = 15,					# day
	mean_death = 30,                                 	# day
	plot = True):
	
	recovery_factor = 1 / mean_infectious_duration    	# 1 / day
	incubation_factor = 1 / mean_incubation                      	# 1 / day
	quarantine_factor = 1 / mean_quarantine                 # 1 / day
	death_factor = 1 / mean_death	                        # 1 / day

	delta_t = .1                                    	# day
	time_values = np.arange(0, 365 * 5, delta_t)

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
		
		SI_contact_rate = I[i - 1] * frac_susceptible * contact_factor 		# contacts / day
		
		exposed_rate = SI_contact_rate * transmissibility			# people / day
		
		quarantine_rate = E[i - 1] * quarantine_factor				# people / day
		quarantine_recovery_rate = Q[i - 1] * recovery_factor  			# recoveries / day
		quarantine_death_rate = Q[i - 1] * death_factor				# deaths / day
		
		infectious_rate = E[i - 1] * incubation_factor			# infections / day
		infection_recovery_rate = I[i - 1] * recovery_factor  			# recoveries / day
		infection_death_rate = I[i - 1] * death_factor				# deaths / day

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
		plt.xlabel("Time (Days)")
		plt.ylabel("Population (People)")
		plt.legend()
		plt.show()
		plt.savefig("plot1.png")

	return (total_pop - S[len(S) - 1]) / total_pop * 100


def main():
	mids = np.arange(.1, 10, .1)
	perc_infs = []

	runsim(plot=True)

	for mid in mids:
		perc_inf = runsim(mid, plot=False)
		perc_infs.append(perc_inf)

	plt.clf()
	plt.plot(mids, perc_infs, color="black", label="% Infected")
	plt.xlabel("Mean Infected Duration (Days)")
	plt.ylabel("% Ever Infected")
	plt.legend()
	plt.show()
	plt.savefig("plot2.png")

main()
