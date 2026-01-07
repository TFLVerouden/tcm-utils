import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import os

import re
import pandas as pd
import sys
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
print(parent_dir)
model_dir = os.path.join(parent_dir, 'cough-machine-control')
model_dir = os.path.join(model_dir,'functions')
print(model_dir)
sys.path.append(model_dir)
import Gupta2009 as Gupta
def sorting_legend(handles,labels,suffix= ""):
    
    # Separate numeric and non-numeric labels
    numeric_labels = []
    non_numeric_labels = []

    for label, handle in zip(labels, handles):
        if label.isdigit():  # Check if the label is numeric
            numeric_labels.append((label, handle))
        else:
            non_numeric_labels.append((label, handle))

    # Sort numeric labels
    sorted_numeric_labels = sorted(numeric_labels, key=lambda x: int(x[0]))

    # Combine sorted numeric labels with non-numeric labels (non-numeric at the end)
    sorted_handles_labels = sorted_numeric_labels + non_numeric_labels

    # Unzip the sorted labels and handles
    sorted_labels, sorted_handles = zip(*sorted_handles_labels)

    # Optionally add " ms" to numeric labels
    sorted_labels = [f"{label} {suffix}" if label.isdigit() else label for label in sorted_labels]

    return sorted_handles,sorted_labels

def data_processer(df):
    """
    Gets in a measurement as dataframe (Pandas)
    calculates PVT, CFPR, CEV and sets the time index correct plus returns a time index.
    Returns:
        return df,t, PVT,CFPR, CEV

    """
    
    dt = np.diff(df.index)
    mask = df.loc[:,'Flow']>0 #finds the first time the flow rate is above 0
    t0 = df.index[mask][0]
    peak_ind = np.argmax(df.loc[:,'Flow'])
   
    PVT = df.index[peak_ind] - t0 #Peak velocity time
    CFPR = df.iloc[peak_ind, 1] #Critical flow pressure rate (L/s)
    CEV = np.sum(dt * df.iloc[1:,1]) #Cumulative expired volume
    df = df[mask]
    t = df.index - t0
    return df,t, PVT,CFPR, CEV

#initalizing
plt.style.use('tableau-colorblind10')
font = {'size'   : 14}
#linestyles: supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
plt.rc('font', **font)
colors = ["blue", "red", "green", "purple","black", "Orange"]
linestyles = ["-", "--", "-.", ":", "-", "--"]
#model

#person E, me based on Gupta et al
Tau = np.linspace(0,10,101)

PVT_E, CPFR_E, CEV_E = Gupta.estimator("Male",70, 1.89)

cough_E = Gupta.M_model(Tau,PVT_E,CPFR_E,CEV_E)


script_dir = os.path.dirname(os.path.abspath(__file__))

print(f"Script is located in: {script_dir}")

# Define the path for the new "results" folder
path = os.path.join(script_dir, "results")
datapath = os.path.join(script_dir, "data_experiments")

csv_files = [f for f in os.listdir(datapath) if f.endswith('.csv')]
extracted_parts = [re.search(r'[^_]+_[^_]+(?=\.csv)', f).group() for f in csv_files if re.search(r'[^_]+_[^_]+(?=\.csv)', f)]



# Create the "results" folder if it doesn't exist
os.makedirs(path, exist_ok=True)


#### plotting
fig, ax1 = plt.subplots(figsize= (8,8))
opening_durations = pd.Series([],dtype=float)
ax1.set_xlim(0, 1.1)
ax1.set_ylim(0, 8)


for i,csv_file in enumerate(csv_files):
    csv_path = os.path.join(datapath,csv_file)
    metadata = pd.read_csv(csv_path,nrows=5,header=None,index_col=0)
    df = pd.read_csv(csv_path,skiprows=7,names=["Time", "Pressure", "Flow"],index_col=0,header=None)

    opening_duration = metadata.loc['Opening duration (ms)']

    ###I want to sort them on plotting style

    if opening_durations.empty:
        label= list(opening_duration.values)[0]
        opening_durations = pd.concat([opening_durations,opening_duration],ignore_index=True)

    elif  opening_durations.isin(opening_duration).any():

        label= "_nolegend_"

    else:
        label= list(opening_duration.values)[0]
        opening_durations = pd.concat([opening_durations,opening_duration],ignore_index=True)

    plot_ind_mask = opening_durations.values == opening_duration.values
    plot_ind = list(opening_durations[plot_ind_mask].index)[0]
  

    #Processing the data for future purposes

    dt = np.diff(df.index)
    mask = df.loc[:,'Flow']>0 #finds the first time the flow rate is above 0
    t0 = df.index[mask][0]
    peak_ind = np.argmax(df.loc[:,'Flow'])
   
    PVT = df.index[peak_ind] - t0 #Peak velocity time
    CFPR = df.iloc[peak_ind, 1] #Critical flow pressure rate (L/s)
    CEV = np.sum(dt * df.iloc[1:,1]) #Cumulative expired volume
    df = df[mask]
    t = df.index - t0

    ax1.plot(t, df.loc[:,'Flow'],color= colors[plot_ind],linestyle = linestyles[plot_ind],label=label)
ax1.plot(Tau*PVT_E,cough_E* CPFR_E,label= "Me", linestyle= ":", linewidth= 5)

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Flow rate (L/s)')
ax1.set_title(f'Comparison of model and experimental based on opening time')
ax1.grid()
##sorting the legend
handles, labels = plt.gca().get_legend_handles_labels()
sorted_handles,sorted_labels = sorting_legend(handles,labels,suffix= "ms")

# Add the legend outside of the plot area (e.g., upper left, outside)
plt.legend(sorted_handles, sorted_labels, loc='upper right')
###
#plt.savefig(path+ "\Modelvsexp_openingtime.png")
#plt.show()
plt.close()

#####NOW for seperate times

time_color = ["r","g","b"]
pressure_linestyle = ["-", "--", "-.", ":",":"]
opening_duration_series =  pd.Series([],dtype=float)
pressure_series = pd.Series([],dtype=float)
plt.figure()

for i,csv_file in enumerate(csv_files):
    csv_path = os.path.join(datapath,csv_file)
    metadata = pd.read_csv(csv_path,nrows=5,header=None,index_col=0)
    df = pd.read_csv(csv_path,skiprows=7,names=["Time", "Pressure", "Flow"],index_col=0,header=None)
    
    
    wanted_opening = pd.Series([60,80,100])

    exp_name = str(metadata.loc['Experiment name'].values)
    
    match = re.search(r'([0-9]*\.?[0-9]+)bar', exp_name)


    
    opening_duration = float(metadata.loc['Opening duration (ms)'].values[0])
    if opening_duration in wanted_opening.values:
        #The only this whole piece does is correctly coloring my lines
        if opening_duration_series.empty:
            opening_duration_series = pd.concat([opening_duration_series,pd.Series([opening_duration])],ignore_index=True)

        elif not opening_duration_series.isin(pd.Series([opening_duration])).any():
            opening_duration_series = pd.concat([opening_duration_series,pd.Series([opening_duration])],ignore_index=True)

            
        plot_color_mask = opening_duration_series.values == opening_duration

        
        plot_color_ind = list(opening_duration_series[plot_color_mask].index)[0]
    #The next part the code only does give it a correct linestyle
            # Check if a match was found and store the result
        if match:
            system_pressure = match.group(1)
            print()
        else:
            system_pressure = None
        if pressure_series.empty:
            pressure_series = pd.concat([pressure_series,pd.Series([system_pressure])],ignore_index=True)

        elif not pressure_series.isin(pd.Series([system_pressure])).any():
            pressure_series = pd.concat([pressure_series,pd.Series([system_pressure])],ignore_index=True)

            
        plot_linestyle_mask = pressure_series.values == system_pressure

        
        plot_linestyle_ind = list(pressure_series[plot_linestyle_mask].index)[0]

        #### From here it is just regular plotting
        label = metadata.loc['Experiment name'].values[0]
        df,t, PVT,CPFR,CEV = data_processer(df)
        label = f"{label}"
        plt.plot(t,df.loc[:,"Flow"],label= label,c= time_color[plot_color_ind],linestyle= pressure_linestyle[plot_linestyle_ind],marker= "o",markeredgecolor= "k")
plt.plot(Tau*PVT_E,cough_E* CPFR_E,label= "Model", linestyle= ":",c="k", linewidth= 5)
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Flow rate (L/s)')
plt.legend(loc= 'upper right',fontsize=10)
plt.show()
#plt.savefig(path+ r"\Modelvsexp_smallopeningtimes.png")

   
