import warnings
warnings.filterwarnings("ignore")
import math
import sys
import csv
import numpy as np
from netCDF4 import Dataset
from decimal import *
import os


def draw_damage(sigma=1., shock_number=1., total_damage=.01, given_events = None,events_shuffle = 1):
    if given_events is None:
        if shock_number == 1:
            return np.array([total_damage])
        else:
            if sigma <= 0.:
                return np.array([total_damage / shock_number for s in range(shock_number)])
            else:
                expected_value = total_damage / shock_number
                s_scale = np.sqrt(np.log(np.power(sigma / expected_value,2.)+1))
                m_local = np.log(expected_value) - .5 * np.power(s_scale,2.)
                side_calculation = 0.
                it = 0
                inner_counter = 0
                outer_counter = 0
                drawer = np.full(shape=(shock_number),fill_value = np.nan)
                while it < shock_number:
                    if it == shock_number - 1:
                        drawer[it] = total_damage - side_calculation
                    else:
                        drawer[it] = np.random.lognormal(m_local,s_scale)
                    side_calculation = side_calculation + drawer[it]
                    if side_calculation > total_damage:
                        inner_counter = inner_counter +1 
                        it = 0
                        side_calculation = 0.
                    else:
                        it = it +1
                        inner_counter = 0
                    if inner_counter == 1000:
                        it = 0
                        inner_counter = 0
                        outer_counter = outer_counter +1 
                        if outer_counter == 1000:
                            print("Run-Off-Draw-Damage")
                            return np.array([total_damage / shock_number for s in range(shock_number)])
    else:
        drawer = np.array(given_events)[:]
    if events_shuffle:
        np.random.shuffle(drawer)
        
    return drawer


def draw_shock_times(shock_number=88,years=35,correct_drawing = True,given_times=None):
    if given_times is None:
        shock_times = np.full(shape=shock_number,fill_value=0)
        if correct_drawing:
            if shock_number == 1:
                shock_times[0] = 12
                return shock_times
            helper_boolean = True
            while helper_boolean:
                shocks_per_season = np.array([np.random.poisson(shock_number/years) for y in range(years)])
                fixed_year = np.random.random_integers(low=0,high = years-1)
                difference = shock_number - np.nansum(shocks_per_season,axis=(0))
                shocks_per_season[fixed_year] = shocks_per_season[fixed_year] + difference
                if shocks_per_season[fixed_year] > 0 and shocks_per_season[fixed_year] < 26:
                    helper_boolean = False

            seasons_times = [[52*y + 22,52*y + 49] for y in range(years)]
            counter = 0
            for season,shocks in zip(seasons_times,shocks_per_season):
                boolean_helper = True
                finisher = 100000
                while boolean_helper:
                    time_season = np.random.random_integers(low=season[0],high=season[1],size=shocks)
                    u, c= np.unique(time_season, return_counts=True)
                    if len(u[c>1]) == 0:
                        boolean_helper = False
                    finisher = finisher - 1
                    if finisher == 0:
                        time_season = np.array([season[0]+i for i in range(shocks)])
                        boolean_helper = False
                shock_times[counter:counter+time_season.shape[0]] = time_season[:]
                counter = counter + shocks
        else:
            boolean_helper = True
            while boolean_helper:
                shock_times = np.random.random_integers(low=0,high=years*52,size=shock_number)
                u, c= np.unique(shock_times, return_counts=True)
                if len(u[c>1]) == 0:
                    boolean_helper = False
    else:
        shock_times = np.array(given_times)[:]
    return shock_times
    

def get_payout(time_original,time_shock,damage_absolute,delta_t = 52):
    a_x = 1./ delta_t
    a_y = 0.6
    b_x = 3./ delta_t
    b_y = 0.9
    f0 = .000000001
    f_helper = (1./f0-1.)
    beta = np.log((np.log(f_helper) - np.log(1./a_y -1.)) / (np.log(f_helper) - np.log(1./b_y -1.))) / (np.log(a_x / b_x))
    tau = a_x / (np.power((np.log(f_helper) - np.log(1. / a_y -1.)),1./beta))
    payout = np.zeros(shape=time_original.shape)
    time = time_original[time_shock+1:] - time_shock

    
    payout[time_shock+1:] = damage_absolute * beta * np.power(((1.*time) / tau),beta-1.) * (f_helper * np.exp(-1.*np.power(((1.*time) / tau),beta))) / (tau * np.power((1.+f_helper*np.exp(-1.*np.power(((1.*time)/tau),beta))),2.))

    return payout



def gini(x):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))
