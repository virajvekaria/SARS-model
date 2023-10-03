import math
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the disease model
transmission_rate = 0.06  # Rate of disease transmission
contact_rate = 10  # Contact rate between individuals
recovery_rate = 0.0975  # Rate of recovery
total_population = 1e7  # Total population size
infection_probability = 0.2  # Probability of infection on contact
exposed_rate = 0.1  # Rate of progression from exposed to infected
quarantine_rate = 0.04  # Rate of quarantine
death_rate = 0.0625  # Rate of death due to the disease

# Function definitions for disease model
def susceptible_change(transmission_rate, contact_rate, infection_probability, susceptible, exposed, infected_quarantined, infected_unquarantined, deceased, recovered, total_population):
    return transmission_rate * exposed - contact_rate * infection_probability * (1 - susceptible / total_population) * infected_unquarantined * susceptible / total_population - contact_rate * infection_probability * infected_unquarantined * susceptible / total_population - contact_rate * (1 - infection_probability) * (1 - susceptible / total_population) * infected_unquarantined * susceptible / total_population

def exposed_change(transmission_rate, contact_rate, infection_probability, susceptible, exposed, infected_quarantined, infected_unquarantined, deceased, recovered, total_population):
    return contact_rate * infection_probability * (1 - susceptible / total_population) * infected_unquarantined * susceptible / total_population - exposed_rate * exposed

def infected_quarantined_change(transmission_rate, contact_rate, infection_probability, susceptible, exposed, infected_quarantined, infected_unquarantined, deceased, recovered, total_population):
    return contact_rate * infection_probability * infected_unquarantined * susceptible / total_population - quarantine_rate * infected_quarantined - exposed_rate * infected_quarantined - death_rate * infected_quarantined

def infected_unquarantined_change(transmission_rate, contact_rate, infection_probability, susceptible, exposed, infected_quarantined, infected_unquarantined, deceased, recovered, total_population):
    return exposed_rate * exposed - recovery_rate * infected_unquarantined - quarantine_rate * infected_unquarantined - death_rate * infected_unquarantined

def deceased_change(transmission_rate, contact_rate, infection_probability, susceptible, exposed, infected_quarantined, infected_unquarantined, deceased, recovered, total_population):
    return death_rate * (infected_quarantined + infected_unquarantined)

def recovered_change(transmission_rate, contact_rate, infection_probability, susceptible, exposed, infected_quarantined, infected_unquarantined, deceased, recovered, total_population):
    return recovery_rate * (infected_quarantined + infected_unquarantined)

# Streamlit app
st.title("SARS Model")

# Input sliders for parameters
infection_probability = st.slider("Infection Probability", 0.0, 0.2, 0.05)

start_time = 0
end_time = 150
time_step = 2

num_steps = int((end_time - start_time) / time_step)

susceptible = np.zeros(num_steps)
exposed = np.zeros(num_steps)
infected_quarantined = np.zeros(num_steps)
infected_unquarantined = np.zeros(num_steps)
deceased = np.zeros(num_steps)
recovered = np.zeros(num_steps)

susceptible[0] = total_population - 1e4
infected_unquarantined[0] = 1e4

data = {
    'Time (days)': [],
    'Susceptible': [],
    'Exposed': [],
    'Infected Quarantined': [],
    'Infected Unquarantined': [],
    'Deceased': [],
    'Recovered': []
}

for i in range(1, len(susceptible)):
    susceptible[i] = susceptible[i - 1] + susceptible_change(transmission_rate, contact_rate, infection_probability, susceptible[i - 1], exposed[i - 1], infected_quarantined[i - 1], infected_unquarantined[i - 1], deceased[i - 1], recovered[i - 1], total_population) * time_step
    exposed[i] = exposed[i - 1] + exposed_change(transmission_rate, contact_rate, infection_probability, susceptible[i - 1], exposed[i - 1], infected_quarantined[i - 1], infected_unquarantined[i - 1], deceased[i - 1], recovered[i - 1], total_population) * time_step
    infected_quarantined[i] = infected_quarantined[i - 1] + infected_quarantined_change(transmission_rate, contact_rate, infection_probability, susceptible[i - 1], exposed[i - 1], infected_quarantined[i - 1], infected_unquarantined[i - 1], deceased[i - 1], recovered[i - 1], total_population) * time_step
    infected_unquarantined[i] = infected_unquarantined[i - 1] + infected_unquarantined_change(transmission_rate, contact_rate, infection_probability, susceptible[i - 1], exposed[i - 1], infected_quarantined[i - 1], infected_unquarantined[i - 1], deceased[i - 1], recovered[i - 1], total_population) * time_step
    deceased[i] = deceased[i - 1] + deceased_change(transmission_rate, contact_rate, infection_probability, susceptible[i - 1], exposed[i - 1], infected_quarantined[i - 1], infected_unquarantined[i - 1], deceased[i - 1], recovered[i - 1], total_population) * time_step
    recovered[i] = recovered[i - 1] + recovered_change(transmission_rate, contact_rate, infection_probability, susceptible[i - 1], exposed[i - 1], infected_quarantined[i - 1], infected_unquarantined[i - 1], deceased[i - 1], recovered[i - 1], total_population) * time_step

    data['Time (days)'].append(i * time_step)
    data['Susceptible'].append(susceptible[i] / total_population)
    data['Exposed'].append(exposed[i] / total_population)
    data['Infected Quarantined'].append(infected_quarantined[i] / total_population)
    data['Infected Unquarantined'].append(infected_unquarantined[i] / total_population)
    data['Deceased'].append(deceased[i] / total_population)
    data['Recovered'].append(recovered[i] / total_population)

# Create a DataFrame
df = pd.DataFrame(data)

# Plotting using Streamlit with DataFrame
st.line_chart(df.set_index('Time (days)'), use_container_width=True)
