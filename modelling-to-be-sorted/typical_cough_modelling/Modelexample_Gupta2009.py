import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import os
import sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
print(parent_dir)
model_dir = os.path.join(parent_dir, 'cough-machine-control')
model_dir = os.path.join(model_dir,'functions')
print(model_dir)
sys.path.append(model_dir)
import Gupta2009 as Gupta
#initalizing
plt.style.use('tableau-colorblind10')
font = {'size'   : 14}
#linestyles: supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
plt.rc('font', **font)

script_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Script is located in: {script_dir}")

# Define the path for the new "results" folder
path = os.path.join(script_dir, "results")

# Create the "results" folder if it doesn't exist
os.makedirs(path, exist_ok=True)



#testing

Tau = np.linspace(0,10,101)

#person A lowest values of Male from Gupta et al 2009

CPFR_A = 3 #L/s
CEV_A = 0.400 #L
PVT_A = 57E-3 #s

cough_A = Gupta.M_model(Tau,PVT_A,CPFR_A,CEV_A)

#person B highest values of Male from Gupta et al 2009

CPFR_B = 8.5 #L/s
CEV_B = 1.6 #L
PVT_B = 96E-3 #s

cough_B = Gupta.M_model(Tau,PVT_B,CPFR_B,CEV_B)

#person C lowest values of feale from Gupta et al 2009

CPFR_C= 1.6 #L/s
CEV_C = 0.25 #L
PVT_C = 57E-3 #s

cough_C = Gupta.M_model(Tau,PVT_C,CPFR_C,CEV_C)

#person D highest values of Male from Gupta et al 2009

CPFR_D = 6 #L/s
CEV_D = 1.25 #L
PVT_D = 110E-3 #s

cough_D = Gupta.M_model(Tau,PVT_D,CPFR_D,CEV_D)

#person E, me based on Gupta et al
PVT_E, CPFR_E, CEV_E = Gupta.estimator("Male",70, 1.89)

cough_E = Gupta.M_model(Tau,PVT_E,CPFR_E,CEV_E)

plt.subplots(1,2,figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(Tau,cough_A,label= "Lower bound (LB) male", linestyle= "-")
plt.plot(Tau,cough_B,label= "Higher bound (HB) male", linestyle= "--")
plt.plot(Tau,cough_C,label= "LB female",linestyle= "-.")
plt.plot(Tau,cough_D,label= "HB female",linestyle= ":")
plt.plot(Tau,cough_E,label= "Our testcase", linestyle= ":", linewidth= 5)
plt.xlabel(r"$\tau$")
plt.ylabel("M")
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.xlim(0, 1.1)
plt.ylim(0, 9)
plt.plot(Tau*PVT_A,cough_A*CPFR_A,label= "LB male", linestyle= "-")
plt.plot(Tau*PVT_B,cough_B*CPFR_B,label= "HB male", linestyle= "--")
plt.plot(Tau*PVT_C,cough_C*CPFR_C,label= "LB female",linestyle= "-.")
plt.plot(Tau*PVT_D,cough_D*CPFR_D,label= "HB female",linestyle= ":")
plt.plot(Tau*PVT_E,cough_E* CPFR_E,label= "Our testcase ", linestyle= ":", linewidth= 5)
plt.legend()
plt.xlabel(r"Time (s)")
plt.ylabel("Q (L/s)")
plt.grid()

plt.savefig(path+ "\\Gupta2009_5modeledcases.png")
plt.figure(figsize=(6,6))
plt.xlim(0, 1.1)
plt.ylim(0, 9)
plt.plot(Tau*PVT_A,cough_A*CPFR_A,label= "Male minimum cough", linestyle= "-")
plt.plot(Tau*PVT_B,cough_B*CPFR_B,label= "Male maximum cough", linestyle= "--")
plt.plot(Tau*PVT_C,cough_C*CPFR_C,label= "Female minimum cough",linestyle= "-.")
plt.plot(Tau*PVT_D,cough_D*CPFR_D,label= "Female maximum cough",linestyle= ":")
plt.plot(Tau*PVT_E,cough_E* CPFR_E,label= "Our test case ", linestyle= ":", linewidth= 5)
plt.legend()
plt.xlabel(r"Time (s)")
plt.ylabel("Q (L/s)")
plt.grid()
plt.savefig(path+ "\\Gupta2009_modeledcasesphysicalparam.png")



"""
#Bernoulli law principle don't think it holds :-/

p_atm = 1.01325E5 #Pascal
pressure = np.linspace(p_atm,1.1*p_atm,100)
rho_air = 1.204 #kg/m^3 
A_trachea = 1E-2 * 2E-2 #m^2
print(30*A_trachea*1E3) #Acccording to article, free stream velocity = 30 m/s. -> 6 L/s flow rate
print(30**2 / 2 * rho_air)  #Gives dP = 542 Pa  ->Corresponds more or less to Bernouilli
Q = np.sqrt(2*(pressure-p_atm)/rho_air)*A_trachea #m^3/s

plt.figure()
plt.plot((pressure-p_atm)*1E-5,Q*1E3)
plt.xlabel(r"$\\Delta$P (bar)")
plt.ylabel("Q (L/s)")
plt.grid()
plt.show()
"""