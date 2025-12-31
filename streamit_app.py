import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Page Config ---
st.set_page_config(page_title="3D BEC Profiler", layout="wide")
st.title("3D Thomas-Fermi BEC Density Explorer")
st.markdown("Calculate the density profile for Bose-Einstein Condensates of various atomic and molecular species.")

# --- 2. Sidebar UI ---
st.sidebar.header("Input Parameters")

species_option = st.sidebar.selectbox(
    "Particle Species", 
    ["Li-7", "Na-23", "K-39", "Cs-133", "Cs2-Molecule"]
)

n_particles = st.sidebar.slider("Total Particle Number (N)", 1000, 500000, 30000, step=1000)
a_bohr = st.sidebar.slider("Scattering Length (a_0)", 0, 1000, 200)

st.sidebar.subheader("Trap Frequencies (Hz)")
f_x = st.sidebar.number_input("fx (Transverse)", value=10.0)
f_y = st.sidebar.number_input("fy (Transverse)", value=10.0)
f_z = st.sidebar.number_input("fz (Axial)", value=150.0)

# --- 3. Physics Logic ---
hbar = 1.0545718e-34      
m_u = 1.6605390e-27       
a0 = 5.2917721e-11        

species_masses = {
    "Li-7": 7.01600,
    "Na-23": 22.98977,
    "K-39": 38.96370,
    "Cs-133": 132.90545,
    "Cs2-Molecule": 132.90545 * 2
}

mass = species_masses[species_option] * m_u
a_s = a_bohr * a0
omega = 2 * np.pi * np.array([f_x, f_y, f_z])
omega_mean = np.prod(omega)**(1/3)

# Thomas-Fermi Calculations
g = 4 * np.pi * hbar**2 * a_s / mass
a_ho = np.sqrt(hbar / (mass * omega_mean))
mu = (hbar * omega_mean / 2) * (15 * n_particles * a_s / a_ho)**(2/5)

R = np.sqrt(2 * mu / (mass * omega**2)) * 1e6
n0_um3 = (mu / g * 1e-18) if g != 0 else 0

# --- 4. Main Display & Plotting ---
if st.button("Run Simulation"):
    col1, col2 = st.columns([2, 1])

    # Generate Data for Plot and Download
    z_axis = np.linspace(-R[2]*1.2, R[2]*1.2, 1000)
    density_z = n0_um3 * np.maximum(0, 1 - (z_axis/R[2])**2)
    
    # Create a DataFrame for download
    df = pd.DataFrame({
        "z_position_um": z_axis,
        "particle_density_um3": density_z
    })

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(z_axis, density_z, lw=3, color='teal')
        ax.fill_between(z_axis, density_z, color='teal', alpha=0.2)
        ax.set_title(f"3D Density Profile along Z-axis: {species_option}")
        ax.set_xlabel(r"Position $z$ ($\mu$m)")
        ax.set_ylabel(r"3D Density $n$ (particles/$\mu$m$^3$)")
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
        
        # Download Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Density Data as CSV",
            data=csv,
            file_name=f"BEC_Density_{species_option}.csv",
            mime='text/csv',
        )

    with col2:
        st.subheader("Simulation Results")
        st.write(f"**Species:** {species_option}")
        st.write(f"**Chemical Potential:** {mu/(hbar*2*np.pi):.2f} Hz")
        st.write(f"**Peak 3D Density:** {n0_um3:.6f} particles/µm³")
        
        st.divider()
        st.write("**Thomas-Fermi Radii:**")
        st.write(f"Rx: {R[0]:.2f} µm")
        st.write(f"Ry: {R[1]:.2f} µm")
        st.write(f"Rz: {R[2]:.2f} µm")
        
        st.divider()
        st.write("**Input Summary**")
        st.write(f"Particle Count: {n_particles:,}")
        st.write(f"Aspect Ratio (fz/fx): {f_z/f_x:.2f}")

else:
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation'.")
