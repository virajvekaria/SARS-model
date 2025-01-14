import math
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

b = 0.06 
k = 10 
m = 0.0975
N = 1e7
p = 0.2
u = 0.1
v = 0.04
w = 0.0625

def s(u,k,q,b,v,w,m,p,S,Sq,E,Eq,Iq,Iu,Id,D,R,N):
  return u*Sq - k*q*(1-b)*Iu*S/N - k*q*Iu*S/N - k*(1-q)*b*Iu*S/N

def sq(u,k,q,b,v,w,m,p,S,Sq,E,Eq,Iq,Iu,Id,D,R,N):
  return k*q*(1-b)*Iu*S/N - u*Sq

def e(u,k,q,b,v,w,m,p,S,Sq,E,Eq,Iq,Iu,Id,D,R,N):
  return k*(1-q)*b*Iu*S/N - p*E

def eq(u,k,q,b,v,w,m,p,S,Sq,E,Eq,Iq,Iu,Id,D,R,N):
  return k*q*b*Iu*S/N - p*Eq

def iu(u,k,q,b,v,w,m,p,S,Sq,E,Eq,Iq,Iu,Id,D,R,N):
  return p*E - m*Iu - v*Iu - w*Iu

def iq(u,k,q,b,v,w,m,p,S,Sq,E,Eq,Iq,Iu,Id,D,R,N):
  return p*Eq - m*Iq - v*Iq - w*Iq

def ids(u,k,q,b,v,w,m,p,S,Sq,E,Eq,Iq,Iu,Id,D,R,N):
  return w*Iu + w*Iq - v*Id - m*Id

def d(u,k,q,b,v,w,m,p,S,Sq,E,Eq,Iq,Iu,Id,D,R,N):
  return m*(Iu + Iq + Id)

def r(u,k,q,b,v,w,m,p,S,Sq,E,Eq,Iq,Iu,Id,D,R,N):
  return v*(Iu + Iq + Id)

# Streamlit app
st.title("SARS Model")

# Create a sidebar for the q slider
q = st.slider("q Value", 0.0, 1.0, 0.01)

start_time = 0
end_time = 150
dt = 2

n = int((end_time - start_time) / dt)

S = np.zeros(n)
Sq = np.zeros(n)
E = np.zeros(n)
Eq = np.zeros(n)
Iu = np.zeros(n)
Id = np.zeros(n)
R = np.zeros(n)
D = np.zeros(n)
Iq = np.zeros(n)

S[0] = 1e7 - 1e4
Iu[0] = 1e4

data = {
    'Time (days)': [],
    'Susceptible(S)\n': [],
    'Susceptible in quarantine(Sq)\n': [],
    'Exposed(E)': [],
    'Exposed in\nquarantine\n(Eq)': [],
    'Infected\nunquarantined\n(Iu)': [],
    'Infected in\nquarantine\n(Iq)': [],
    'Infected and\ndeceased\n(Id)': [],
    'Deceased\n(D)': [],
    'Recovered\n(R)': []
}

for i in range(1, len(S)):
    S[i] = S[i - 1] + s(u, k, q, b, v, w, m, p, S[i - 1], Sq[i - 1], E[i - 1], Eq[i - 1], Iq[i - 1], Iu[i - 1], Id[i - 1], D[i - 1], R[i - 1], N) * dt
    Sq[i] = Sq[i - 1] + sq(u, k, q, b, v, w, m, p, S[i - 1], Sq[i - 1], E[i - 1], Eq[i - 1], Iq[i - 1], Iu[i - 1], Id[i - 1], D[i - 1], R[i - 1], N) * dt
    E[i] = E[i - 1] + e(u, k, q, b, v, w, m, p, S[i - 1], Sq[i - 1], E[i - 1], Eq[i - 1], Iq[i - 1], Iu[i - 1], Id[i - 1], D[i - 1], R[i - 1], N) * dt
    Eq[i] = Eq[i - 1] + eq(u, k, q, b, v, w, m, p, S[i - 1], Sq[i - 1], E[i - 1], Eq[i - 1], Iq[i - 1], Iu[i - 1], Id[i - 1], D[i - 1], R[i - 1], N) * dt
    Iu[i] = Iu[i - 1] + iu(u, k, q, b, v, w, m, p, S[i - 1], Sq[i - 1], E[i - 1], Eq[i - 1], Iq[i - 1], Iu[i - 1], Id[i - 1], D[i - 1], R[i - 1], N) * dt
    Id[i] = Id[i - 1] + ids(u, k, q, b, v, w, m, p, S[i - 1], Sq[i - 1], E[i - 1], Eq[i - 1], Iq[i - 1], Iu[i - 1], Id[i - 1], D[i - 1], R[i - 1], N) * dt
    Iq[i] = Iq[i - 1] + iq(u, k, q, b, v, w, m, p, S[i - 1], Sq[i - 1], E[i - 1], Eq[i - 1], Iq[i - 1], Iu[i - 1], Id[i - 1], D[i - 1], R[i - 1], N) * dt
    D[i] = D[i - 1] + d(u, k, q, b, v, w, m, p, S[i - 1], Sq[i - 1], E[i - 1], Eq[i - 1], Iq[i - 1], Iu[i - 1], Id[i - 1], D[i - 1], R[i - 1], N) * dt
    R[i] = R[i - 1] + r(u, k, q, b, v, w, m, p, S[i - 1], Sq[i - 1], E[i - 1], Eq[i - 1], Iq[i - 1], Iu[i - 1], Id[i - 1], D[i - 1], R[i - 1], N) * dt

    data['Time (days)'].append(i * dt)
    data['Susceptible(S)\n'].append(S[i] / N)
    data['Susceptible in quarantine(Sq)\n'].append(Sq[i] / N)
    data['Exposed(E)'].append(E[i] / N)
    data['Exposed in\nquarantine\n(Eq)'].append(Eq[i] / N)
    data['Infected\nunquarantined\n(Iu)'].append(Iu[i] / N)
    data['Infected in\nquarantine\n(Iq)'].append(Iq[i] / N)
    data['Infected and\ndeceased\n(Id)'].append(Id[i] / N)
    data['Deceased\n(D)'].append(D[i] / N)
    data['Recovered\n(R)'].append(R[i] / N)

# Create a DataFrame
df = pd.DataFrame(data)

# Create a figure and axis for Matplotlib
fig, ax = plt.subplots()

# Plotting the data using Matplotlib
ax.plot(df['Time (days)'], df['Susceptible(S)\n'], label='Susceptible(S)')
ax.plot(df['Time (days)'], df['Susceptible in quarantine(Sq)\n'], label='Susceptible in quarantine(Sq)')
ax.plot(df['Time (days)'], df['Exposed(E)'], label='Exposed(E)')
ax.plot(df['Time (days)'], df['Exposed in\nquarantine\n(Eq)'], label='Exposed in quarantine(Eq)')
ax.plot(df['Time (days)'], df['Infected\nunquarantined\n(Iu)'], label='Infected unquarantined(Iu)')
ax.plot(df['Time (days)'], df['Infected in\nquarantine\n(Iq)'], label='Infected in quarantine(Iq)')
ax.plot(df['Time (days)'], df['Infected and\ndeceased\n(Id)'], label='Infected and deceased(Id)')
ax.plot(df['Time (days)'], df['Deceased\n(D)'], label='Deceased(D)')
ax.plot(df['Time (days)'], df['Recovered\n(R)'], label='Recovered(R)')

# Add labels and title
ax.set_xlabel('Time (days)')
ax.set_ylabel('Normalized Population')
ax.set_title('SARS Model')

# Display the legend below the graph
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=1)

# Show the Matplotlib plot in Streamlit using st.pyplot
st.pyplot(fig)
