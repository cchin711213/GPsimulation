import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# --- 1. App Configuration ---
st.set_page_config(page_title="1D BEC Explorer", layout="wide")
st.title(fr"$\text{Cs}^{{133}}$ Bose-Einstein Condensate Explorer")

# --- 2. Sidebar Parameters ---
st.sidebar.header("Physics Parameters")
N_tot = st.sidebar.slider("Total Atoms (N)", 1000, 50000, 20000, step=1000)
a_3d_bohr = st.sidebar.slider("Scattering Length (Bohr)", 0, 500, 200)
f_perp = st.sidebar.number_input("Transverse Freq (Hz)", value=150.0)
f_x = st.sidebar.number_input("Axial Freq (Hz)", value=10.0)

# --- 3. Physics Logic (The math you verified in Colab) ---
hbar_si = 1.0545718e-34      
m_u = 1.6605390e-27          
a0 = 5.2917721e-11           
mass_cs = 133 * m_u          
a_3d = a_3d_bohr * a0              
um, ms = 1e-6, 1e-3
E_scale = mass_cs * (um/ms)**2 
hbar = hbar_si / (E_scale * ms) 

omega_perp = 2 * np.pi * f_perp
omega_x = 2 * np.pi * f_x
g1d = (2 * hbar_si * omega_perp * a_3d) / (E_scale * um)

# --- 4. Simulation Execution ---
if st.button("Run Simulation"):
    with st.spinner("Calculating Ground State..."):
        N, L = 512, 120.0 
        x = np.linspace(-L/2, L/2, N)
        dx = x[1] - x[0]
        k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
        V = 0.5 * (omega_x * ms)**2 * x**2 

        dt, steps = 0.02, 1000
        psi = np.exp(-x**2 / 50).astype(complex)
        psi *= np.sqrt(N_tot / trapezoid(np.abs(psi)**2, x))

        for _ in range(steps):
            psi = np.fft.ifft(np.exp(-0.5 * hbar * (k**2) * (dt/2)) * np.fft.fft(psi))
            psi *= np.exp(-(V + g1d * np.abs(psi)**2) * dt / hbar)
            psi = np.fft.ifft(np.exp(-0.5 * hbar * (k**2) * (dt/2)) * np.fft.fft(psi))
            psi *= np.sqrt(N_tot / trapezoid(np.abs(psi)**2, x))

        # Result calculation
        psi_k = np.fft.fft(psi)
        ke_dens = np.real(np.fft.ifft(0.5 * hbar**2 * k**2 * psi_k)) * np.conj(psi)
        mu = np.real(trapezoid(ke_dens + V*np.abs(psi)**2 + g1d*np.abs(psi)**4, x)) / N_tot
        mu_hz = (mu * (E_scale / ms)) / (2 * np.pi * hbar_si)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, np.abs(psi)**2, color='royalblue', lw=2)
        ax.fill_between(x, np.abs(psi)**2, color='royalblue', alpha=0.2)
        ax.set_title(fr"Ground State ($\mu = {mu_hz:.2f}$ Hz)")
        ax.set_xlabel(r"Position ($\mu$m)")
        ax.set_ylabel(r"Density (atoms/$\mu$m)")
        st.pyplot(fig)
else:
    st.info("Adjust settings and click 'Run Simulation'.")
