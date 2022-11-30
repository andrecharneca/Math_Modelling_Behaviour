import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, log, exp
import biogeme.segmentation as seg
import numpy as np


if __name__ == '__main__':
    # load the data
    df = pd.read_csv('../data/lpmc01.dat', sep= '\t') 

    census = {
        'OLD_MEN': 1633263,
        'YG_MEN': 2676249,
        'OLD_WOMEN': 1765143,
        'YG_WOMEN': 2599058,
    }

    #define new filters for strata 
    filters = {
        'OLD_MEN': (df.age > 40) & (df.female == 0),
        'YG_MEN': (df.age <= 40) & (df.female == 0),
        'OLD_WOMEN': (df.age > 40) & (df.female == 1),
        'YG_WOMEN': (df.age <= 40) & (df.female == 1),
    }

    pop_total = sum(census.values())
    print("\nTotal population: ", pop_total)

    #count the number of observations in each strata
    sample_segments = {
        k: v.sum() for k, v in filters.items()
    }
    print("\nSample segments: ", sample_segments)

    total_sample = sum(sample_segments.values())
    print("\nTotal sample: ", total_sample)

    # strata weights
    weights = {
        k: census[k] * total_sample / (v * pop_total) 
        for k, v in sample_segments.items()
    }
    print("\nWeights: ", weights)

    # insert the weights into the database for each person
    for k, f in filters.items():
        df.loc[f, 'Weight'] = weights[k] 
    
    # check that the weights sum up to the sample size
    sum_weights = df['Weight'].sum()
    print("\nSum of weights (should be equal to sample size): ", sum_weights)

    ## Market Shares ##
    database = db.Database('LPMC', df)
    # create age_group column
    df['age_group'] = pd.cut(df['age'], [0, 16, 30, 60, 1000], labels=[0, 1, 2, 3])

    # import model_pref
    from model_pref import V_WALK, V_BIKE, V_PT, V_CAR, logprob
    prob_WALK = exp(V_WALK) / (exp(V_WALK) + exp(V_BIKE) + exp(V_PT) + exp(V_CAR))
    prob_BIKE = exp(V_BIKE) / (exp(V_WALK) + exp(V_BIKE) + exp(V_PT) + exp(V_CAR))
    prob_PT = exp(V_PT) / (exp(V_WALK) + exp(V_BIKE) + exp(V_PT) + exp(V_CAR))
    prob_CAR = 1 - prob_WALK - prob_BIKE - prob_PT

    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = 'lpmc_model'
    results = biogeme.estimate()

    # compute choice probability for each alternative, for each observation
    Weight = Variable('Weight')
    simulate = {
        'Weight': Weight,
        'prob_WALK': prob_WALK,
        'prob_BIKE': prob_BIKE,
        'prob_PT': prob_PT,
        'prob_CAR': prob_CAR,
    }
    biosim = bio.BIOGEME(database, simulate)
    simulated_values = biosim.simulate(results.getBetaValues())
    print("\nSimulated values (head): ", simulated_values.head())

    # compute market shares with weighted means of individual probabilities
    for mode in ['WALK', 'BIKE', 'PT', 'CAR']:
        simulated_values['Weighted ' + mode] = (
            simulated_values['Weight'] *
            simulated_values['prob_' + mode]
        )

    # use a loop to compute the market shares for all modes
    for mode in ['WALK', 'BIKE', 'PT', 'CAR']:
        market_share = simulated_values['Weighted ' + mode].mean()
        print(f'Market share for {mode}: {100*market_share:.1f}%')

