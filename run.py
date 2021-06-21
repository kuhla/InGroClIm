import warnings
warnings.filterwarnings("ignore")
import math
import sys
sys.path.insert(0, '../python_script')
import scripts
import parameter
import csv
import numpy as np
from netCDF4 import Dataset
from decimal import *
from tqdm import tqdm
import os
import argparse
import os

parser = argparse.ArgumentParser(description='') 
parser.add_argument('--outfile', type=str)
parser.add_argument('--simulation_number', type=int)
parser.add_argument('--insurance_coverage', type=float,default = 0.5)
parser.add_argument('--investment_cap', type=float,default = 0.01)
parser.add_argument('--total_damage', type=float,default = 0.0324)
parser.add_argument('--shock_number', type=int,default = 88)
parser.add_argument('--initial_growth', type=float,default = 2.6)
parser.add_argument('--sigma', type=float,default = 0.0010654)
parser.add_argument('--technology_growth', type=float,default = None)
parser.add_argument('--time_drawing_correct', type=int,default = 1,choices = [0,1])
parser.add_argument('--events_shuffle', type=int,default = 1,choices = [0,1])
parser.add_argument('--given_events', type=float,nargs = '+',default = None)
parser.add_argument('--given_times', type=int,nargs = '+',default = None)
parser.add_argument('--years', type=int,default = 35)
args = parser.parse_args()


if True:
    
    initial_production = 51638.10
    growth_initial_year = 1. + args.initial_growth/100.
    total_damage = args.total_damage
    total_shocks = args.shock_number
    
    
    
    time_steps_per_year = 52
    time_years = args.years
    time_steps = time_steps_per_year * time_years
    
    
    
    prefactor_insurance = 1.465*5.3/(1.*time_steps-1.)
    cobb_douglas_exponent = 0.7
    depreciation_rate = 0.1
    saving_rate = 0.2
    sigma = args.sigma
    delta_t = 1./(time_steps_per_year)
    production_growth = np.power(growth_initial_year,1./time_steps_per_year) -1.
    initial_capital = saving_rate*delta_t*initial_production/(depreciation_rate*delta_t + production_growth)
    initial_technology_factor = initial_production * np.power(initial_capital,-1.*cobb_douglas_exponent)
    if args.technology_growth is None:
        technology_growth = (np.power(1.+production_growth,1.-cobb_douglas_exponent)) - 1.
    else:
        technology_growth = args.technology_growth




time = np.array(range(time_steps))
capital_potential_unperturbed = np.full(shape=(time_steps),fill_value = np.nan)
capital_potential_unaffected = np.full(shape=(time_steps),fill_value = np.nan)
production_unperturbed = np.full(shape=(time_steps),fill_value = np.nan)
production_unaffected = np.full(shape=(time_steps),fill_value = np.nan)
capital_potential = np.full(shape=(time_steps),fill_value = np.nan)
capital_insurance = np.full(shape=(time_steps),fill_value = 0.)
xi = np.full(shape=(time_steps),fill_value = 1.)
production = np.full(shape=(time_steps),fill_value = np.nan)
payout = np.zeros(shape=(time_steps))
investment = np.full(shape=(time_steps),fill_value = np.nan)
investment_xi = np.full(shape=(time_steps),fill_value = np.nan)
investment_potential = np.full(shape=(time_steps),fill_value = np.nan)
technology = np.full(shape=(time_steps),fill_value = initial_technology_factor)

damages_abs = np.full(shape=(total_shocks),fill_value = np.nan)



if not os.path.isfile(args.outfile):
    writing = Dataset(args.outfile,'w', format='NETCDF4')
    writing.setncattr('saving_rate',saving_rate)
    writing.setncattr('growth_initial',production_growth)
    writing.setncattr('depreciation_rate',depreciation_rate)
    writing.setncattr('sigma',sigma)
    writing.setncattr('prefactor_insurance',prefactor_insurance)
    writing.setncattr('cobb_douglas_exponent',cobb_douglas_exponent)
    writing.setncattr('initial_production',initial_production)
    writing.setncattr('total_shocks',float(total_shocks))
    writing.setncattr('total_damage',total_damage)
    
    time_dim = writing.createDimension('time',time_steps)
    simulation_dim = writing.createDimension('simulation',None)
    shocks_dim = writing.createDimension('shocks',total_shocks)
        

        
    writing.createVariable('shocks',int,('shocks',))
    writing.createVariable('simulation',int,('simulation',))
    writing.createVariable('time',float,('time',))
    writing.createVariable('capital_potential_unperturbed',float,('simulation','time',),zlib=True)
    writing.createVariable('capital_potential_unaffected',float,('simulation','time',),zlib=True)
    writing.createVariable('production_unperturbed',float,('simulation','time',),zlib=True)
    writing.createVariable('production_unaffected',float,('simulation','time',),zlib=True)
    writing.createVariable('capital_potential',float,('simulation','time',),zlib=True)
    writing.createVariable('xi',float,('simulation','time',),zlib=True)
    writing.createVariable('production',float,('simulation','time',),zlib=True)
    writing.createVariable('investment',float,('simulation','time',),zlib=True)
    writing.createVariable('investment_xi',float,('simulation','time',),zlib=True)
    writing.createVariable('technology',float,('simulation','time',),zlib=True)
    writing.createVariable('investment_potential',float,('simulation','time',),zlib=True)
    writing.createVariable('payout',float,('simulation','time',),zlib=True)
    writing.createVariable('capital_insurance',float,('simulation','time',),zlib=True)
    writing.createVariable('shock_times',int,('simulation','shocks',),zlib=True)
    writing.createVariable('shock_sizes_relative',float,('simulation','shocks',),zlib=True)
    writing.createVariable('shock_sizes_absolute',float,('simulation','shocks',),zlib=True)
    
    writing.variables['time'][:] = time[:]
    writing.variables['shocks'][:] = np.array(range(total_shocks))[:]

else:
    writing = Dataset(args.outfile,'a', format='NETCDF4')
writing.variables['simulation'][args.simulation_number] = args.simulation_number
capital_potential[0] = initial_capital
production[0] = initial_production
investment_xi[0] = 0.
xi[0] = 1.
investment[0] = delta_t * saving_rate * technology[0] * xi[0] * np.power(capital_potential[0],cobb_douglas_exponent)
investment_potential[0] = investment[0]

capital_potential_unperturbed[0] = initial_capital
production_unperturbed[0] = initial_production
capital_potential_unaffected[0] = initial_capital
production_unaffected[0] = initial_production
for tt in time[1:]:
    technology[tt] = initial_technology_factor *np.power(1. + technology_growth,tt)
    capital_potential_unaffected[tt] =  delta_t * saving_rate * production_unaffected[tt-1] +(1. - delta_t * depreciation_rate)*capital_potential_unaffected[tt-1]
    production_unaffected[tt] = technology[tt]*np.power(capital_potential_unaffected[tt],cobb_douglas_exponent)
    

insurance_rate = prefactor_insurance * (saving_rate * initial_technology_factor - (depreciation_rate)) * args.insurance_coverage * total_damage * capital_potential_unaffected[-1] / ((capital_potential_unaffected[-1] - capital_potential_unaffected[0]));

damages = scripts.draw_damage(sigma = sigma, shock_number = total_shocks, total_damage = total_damage,given_events = args.given_events,events_shuffle = args.events_shuffle)

shock_times = scripts.draw_shock_times(shock_number = total_shocks, years = time_years,correct_drawing =args.time_drawing_correct,given_times = args.given_times)

for tt in time[1:]:    
    
    capital_potential[tt] = investment_potential[tt-1] / xi[tt-1]  + (1. - delta_t * (depreciation_rate + insurance_rate)) * capital_potential[tt-1]
    capital_potential_unperturbed[tt] = delta_t *saving_rate * production_unperturbed[tt-1]   + (1. - delta_t * (depreciation_rate + insurance_rate)) * capital_potential_unperturbed[tt-1]
    capital_insurance[tt] = capital_insurance[tt-1] + delta_t * insurance_rate * capital_potential[tt-1] - payout[tt-1]

    
    if tt not in shock_times:
        xi[tt] = investment_xi[tt -1] / capital_potential[tt - 1] + xi[tt-1]
    else:
        damage = damages[np.argwhere(shock_times ==tt)[0]][0]
        xi[tt] = investment_xi[tt -1] / capital_potential[tt - 1] + xi[tt-1] - damages[np.argwhere(shock_times ==tt)[0]]
        if xi[tt] < 0.:
            xi[tt] = 0.
        damages_abs[np.argwhere(shock_times ==tt)[0]] = damage * capital_potential[tt]
        payout = payout[:] + scripts.get_payout(time,tt,args.insurance_coverage*damage*capital_potential[tt],delta_t = delta_t)
    investment[tt] = delta_t * saving_rate * technology[tt] * xi[tt] * np.power(capital_potential[tt],cobb_douglas_exponent) + payout[tt]
    if xi[tt] >= 1.:
        investment_xi[tt] = 0.
    else:
        investment_xi[tt] = np.minimum(np.minimum(delta_t*args.investment_cap*technology[tt]*xi[tt]*np.power(capital_potential[tt],cobb_douglas_exponent) + payout[tt],investment[tt]),(1.-xi[tt])*capital_potential[tt])
    investment_potential[tt] = investment[tt] - investment_xi[tt]
    production[tt] = technology[tt] * xi[tt] * np.power(capital_potential[tt],cobb_douglas_exponent)
    production_unperturbed[tt] = technology[tt] * np.power(capital_potential_unperturbed[tt],cobb_douglas_exponent)
    

writing.variables['capital_potential_unperturbed'][args.simulation_number,:] = capital_potential_unperturbed[:]
writing.variables['capital_potential_unaffected'][args.simulation_number,:] = capital_potential_unaffected[:]
writing.variables['production_unaffected'][args.simulation_number,:] = production_unaffected[:]
writing.variables['production_unperturbed'][args.simulation_number,:] = production_unperturbed[:]
writing.variables['capital_potential'][args.simulation_number,:] = capital_potential[:]
writing.variables['production'][args.simulation_number,:] = production[:]
writing.variables['investment'][args.simulation_number,:] = investment[:]
writing.variables['investment_potential'][args.simulation_number,:] = investment_potential[:]
writing.variables['investment_xi'][args.simulation_number,:] = investment_xi[:]
writing.variables['xi'][args.simulation_number,:] = xi[:]
writing.variables['capital_insurance'][args.simulation_number,:] = capital_insurance[:]
writing.variables['payout'][args.simulation_number,:] = payout[:]
writing.variables['technology'][args.simulation_number,:] = technology[:]

writing.variables['shock_times'][args.simulation_number,:] = shock_times[:]
writing.variables['shock_sizes_relative'][args.simulation_number,:] = damages[:]
writing.variables['shock_sizes_absolute'][args.simulation_number,:] = damages_abs[:]


