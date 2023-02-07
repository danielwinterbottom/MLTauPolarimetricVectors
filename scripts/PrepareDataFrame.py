import uproot3
import pandas as pd
import argparse
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option("display.precision", 2)

# Set up the argument parser
parser = argparse.ArgumentParser(description="Load variables from a root file and save them to a .pkl file.")

# Add the root file name argument
parser.add_argument("-i","--input_file", type=str, help="The name of the input root file")

# Add the output file name argument
parser.add_argument("-o","--output_file", type=str, help="The name of the output file")

# Parse the arguments
args = parser.parse_args()

# Open the root file using uproot
file = uproot3.open(args.input_file)

pi0_mass = float(134.9768/1000)
pi_mass  = float(139.57039/1000)
tau_mass = float(1776.86/1000)

variables = [
  "pi_px_1", "pi_py_1", "pi_pz_1", "pi_E_1", "pi_px_2", "pi_py_2", "pi_pz_2", "pi_E_2",
  "pi2_px_1", "pi2_py_1", "pi2_pz_1", "pi2_E_1", "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
  "pi3_px_1", "pi3_py_1", "pi3_pz_1", "pi3_E_1", "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
  "pi0_px_1", "pi0_py_1", "pi0_pz_1", "pi0_E_1", "pi0_px_2", "pi0_py_2", "pi0_pz_2", "pi0_E_2",
  "sv_x_1", "sv_y_1", "sv_z_1", "sv_x_2", "sv_y_2", "sv_z_2",
  "ip_x_1", "ip_y_1", "ip_z_1", "ip_x_2", "ip_y_2", "ip_z_2",
  "metx", "mety",
  "tau1_charge", "tau2_charge", "dm_1", "dm_2",
  "taupos_px", "taupos_py", "taupos_pz", "taupos_E",
  "tauneg_px", "tauneg_py", "tauneg_pz", "tauneg_E",
  "taupos_polvec_x", "taupos_polvec_y", "taupos_polvec_z", "taupos_polvec_E",
  "tauneg_polvec_x", "tauneg_polvec_y", "tauneg_polvec_z", "tauneg_polvec_E"
]

variables_nodm = [var for var in variables if not var.startswith('dm_')]

# Load the variables into a pandas dataframe
df = file['ntuple'].pandas.df(variables)

# rearrange the names of the input variables so that the positive charge tau is always index 1
variables_to_switch = ["pi_px", "pi_py", "pi_pz", "pi_E", "pi2_px", "pi2_py", "pi2_pz", "pi2_E",
                       "pi3_px", "pi3_py", "pi3_pz", "pi3_E", "pi0_px", "pi0_py", "pi0_pz", "pi0_E",
                       "sv_x", "sv_y", "sv_z", "ip_x", "ip_y", "ip_z","dm"]
mask = df['tau1_charge'] < df['tau2_charge']
for var in variables_to_switch:
  df.loc[mask, var + "_1"], df.loc[mask, var + "_2"] = df.loc[mask, var + "_2"], df.loc[mask, var + "_1"]

# drop all rows that don't include a tauhtauh decay

df = df[(df['dm_1'] == 0) | (df['dm_1'] == 1) | (df['dm_1'] == 10)]
df = df[(df['dm_2'] == 0) | (df['dm_2'] == 1) | (df['dm_2'] == 10)]

# In the case of a1 decays by convension the first pion has the opposite charge to the tau

# Drop rows containing any NaN values
df.dropna(inplace=True)

df = df.astype(float)

# create 4-vectors to store opjects.
# store 4-vectors as numpy arrays with format: [E,px,py,pz]
# convert columns to a numpy array and stack them together to form a 2D array

variables_to_convert = ['pi','pi2','pi3','pi0','sv','ip']

for var in variables_to_convert:
  for i in ['1','2']:
    if var == 'pi0':  mass = pi0_mass
    elif 'pi' in var: mass = pi_mass
    else: mass=0.
    # First we do this for pions where 3-momentum and E are all already defined, however we recompute E assuming the known pion masses
    if var.startswith('pi'):  np_array = np.column_stack(((np.sqrt(df[var+'_px_'+i]**2 + df[var+'_py_'+i]**2 + df[var+'_pz_'+i]**2 + mass**2)).values,  df[var+'_px_'+i].values, df[var+'_py_'+i].values, df[var+'_pz_'+i].values))
    # Now for IPs and SV where only x, y, z are defined - we set the the E to equal the quadrature sum of the x, y, and z
    if var.startswith('sv') or var.startswith('ip'):  np_array = np.column_stack(((np.sqrt(df[var+'_x_'+i]**2 + df[var+'_y_'+i]**2 + df[var+'_z_'+i]**2)).values,  df[var+'_x_'+i].values, df[var+'_y_'+i].values, df[var+'_z_'+i].values))
    # create a new column in the dataframe with the stacked array
    df[var+'_4vec_'+i] = np_array.tolist()


for var in ['taupos_polvec','tauneg_polvec']:
  np_array = np.column_stack((df[var+'_E'].values,  df[var+'_x'].values, df[var+'_y'].values, df[var+'_z'].values))
  df[var+'_4vec'] = np_array.tolist()
for var in ['taupos','tauneg']:
  np_array = np.column_stack((df[var+'_E'].values,  df[var+'_px'].values, df[var+'_py'].values, df[var+'_pz'].values))
  df[var+'_4vec'] = np_array.tolist()

# Now for the MET we set the z component to 0 and set the energy as the quadrature sum of the x and y components 
np_array = np.column_stack(((np.sqrt(df['metx']**2 + df['mety']**2)).values,  df['metx'].values, df['mety'].values, np.zeros((len(df), 1),dtype=float)))
df['met_4vec'] = np_array.tolist()
    
def set_empty_4vectors(df, var, mask):
  new = pd.Series(list(np.zeros((np.sum(mask), 4),dtype=float)), index=df.index[mask])
  df.loc[mask, var] = new

for i in ['1','2']:
  # set pi to 0 if tau is not a hadronic decay
  set_empty_4vectors(df, 'pi_4vec_%s' % i, df['dm_%s' % i] <0) 
  # set pi0 to 0 if tau does not have pi0 in decay final state
  set_empty_4vectors(df, 'pi0_4vec_%s' % i, (df['dm_%s' % i] !=1) & (df['dm_%s' % i] !=2) & (df['dm_%s' % i] !=11)) 
  # set pi's 2 and 3 to 0 if tau is 1-prong decay
  for j in ['2','3']: set_empty_4vectors(df, 'pi%s_4vec_%s' % (j,i), df['dm_%s' % i] <10) 
  # set sv to 0 for 1-prong taus (since we don't reconstruct this in the detector in such cases)
  set_empty_4vectors(df, 'sv_4vec_%s' % i, df['dm_%s' % i] <10)
  # set ip to 0 for 3-prong taus - we don't need this as we have a sv 
  set_empty_4vectors(df, 'ip_4vec_%s' % i, df['dm_%s' % i] >=10) 



# Drop initial variables
df = df.drop(variables_nodm, axis=1)

print(df[:5])
#print(df['met_4vec'][:5].tolist())

# Save the dataframe to a .pkl file
df.to_pickle(args.output_file)
