import uproot
import pandas as pd
import argparse
import numpy as np
#pd.set_option('display.max_columns', None)

# Set up the argument parser
parser = argparse.ArgumentParser(description="Load variables from a root file and save them to a .pkl file.")

# Add the root file name argument
parser.add_argument("-i","--input_file", type=str, help="The name of the input root file")

# Add the output file name argument
parser.add_argument("-o","--output_file", type=str, help="The name of the output file")

# Parse the arguments
args = parser.parse_args()

# Open the root file using uproot
file = uproot.open(args.input_file)

variables = [
  "pi_px_1", "pi_py_1", "pi_pz_1", "pi_E_1", "pi_px_2", "pi_py_2", "pi_pz_2", "pi_E_2",
  "pi2_px_1", "pi2_py_1", "pi2_pz_1", "pi2_E_1", "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
  "pi3_px_1", "pi3_py_1", "pi3_pz_1", "pi3_E_1", "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
  "pi0_px_1", "pi0_py_1", "pi0_pz_1", "pi0_E_1", "pi0_px_2", "pi0_py_2", "pi0_pz_2", "pi0_E_2",
  "sv_x_1", "sv_y_1", "sv_z_1", "sv_x_2", "sv_y_2", "sv_z_2",
  "ip_x_1", "ip_y_1", "ip_z_1", "ip_x_2", "ip_y_2", "ip_z_2",
  "metx", "mety",
  "tau1_charge", "tau2_charge", "dm_1", "dm_2",
  "taupos_px", "taupos_py", "taupos_pz", "taupos_m",
  "tauneg_px", "tauneg_py", "tauneg_pz", "tauneg_m",
  "taupos_polvec_x", "taupos_polvec_y", "taupos_polvec_z", "taupos_polvec_m",
  "tauneg_polvec_x", "tauneg_polvec_y", "tauneg_polvec_z", "tauneg_polvec_m"
]

# Load the variables into a pandas dataframe
df = file['ntuple'].pandas.df(variables)

# rearrange the names of the input variables so that the positive charge tau is always index 1
variables_to_switch = ["pi_px", "pi_py", "pi_pz", "pi_E", "pi2_px", "pi2_py", "pi2_pz", "pi2_E",
                       "pi3_px", "pi3_py", "pi3_pz", "pi3_E", "pi0_px", "pi0_py", "pi0_pz", "pi0_E",
                       "sv_x", "sv_y", "sv_z", "ip_x", "ip_y", "ip_z"]
mask = df['tau1_charge'] < df['tau2_charge']
for var in variables_to_switch:
  df.loc[mask, var + "_1"], df.loc[mask, var + "_2"] = df.loc[mask, var + "_2"], df.loc[mask, var + "_1"]

#print df[['tau1_charge','tau2_charge','pi_px_1','pi_px_2']][:10]

# In the case of a1 decays by convension the first pion has the opposite charge to the tau

# Drop rows containing any NaN values
df.dropna(inplace=True)

#print(df.columns)

# create 4-vectors to store opjects.
# store 4-vectors as numpy arrays with format: [E,px,py,pz]

# convert columns to a numpy array and stack them together to form a 2D array
# first we do this for pions where pi and E are all already defined
variables_to_convert = ['pi','pi2','pi3','pi0']
for var in variables_to_convert:
  for i in ['1','2']:
    np_array = np.column_stack((df[var+'_E_'+i].values, df[var+'_px_'+i].values, df[var+'_py_'+i].values, df[var+'_pz_'+i].values))

    # create a new column in the dataframe with the stacked array
    df[var+'_4vec_'+i] = np_array.tolist()

# Now for IPs and SV where only x, y, z are defined - we set the the E to equal the quadrature sum of the x, y, and z
#....

# Now for the MET we set the z component to 0 and set the energy as the quadrature sum of the x and y components 
#....

# Drop initial variables
df = df.drop(variables, axis=1)

print df[:10]

# Save the dataframe to a .pkl file
df.to_pickle(args.output_file)
