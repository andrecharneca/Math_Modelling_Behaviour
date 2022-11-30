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


    ## Question 1: Weights ##


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


    ## Question 2: Predicted Market Shares  ##
    # We want to compute the market share of each mode, by using the model
    # to predict the probability of choosing each mode. We multiply this by
    # the weight of each person, and sum over all persons. This gives the 
    # predicted market share of each mode.
    # The confidence interval is computed with bootstrapping.

    
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

    # get market shares
    simulated_values = biosim.simulate(results.getBetaValues())

    for mode in ['WALK', 'BIKE', 'PT', 'CAR']:
        simulated_values[f'Weighted {mode}'] = (
            simulated_values['Weight'] * 
            simulated_values[f'prob_{mode}']
        )

    # bootstrap the data
    N_boot = 200
    print(f"Bootstrapping {N_boot} times...")
    results_bootstrapping = biogeme.estimate(bootstrap=N_boot)
    betas = biogeme.freeBetaNames()
    b = results_bootstrapping.getBetasForSensitivityAnalysis(betas)

    # confidence interval of 90%
    left, right = biosim.confidenceIntervals(b, 0.9)

    for mode in ['WALK', 'BIKE', 'PT', 'CAR']:
        left['Weighted ' + mode] = (
            left['Weight'] * 
            left['prob_' + mode]
        )
        right['Weighted ' + mode] = (
            right['Weight'] * 
            right['prob_' + mode]
        )
 
    for mode in ['WALK', 'BIKE', 'PT', 'CAR']:
        left_str = '{:.2f}'.format(left["Weighted " + mode].mean())
        right_str = '{:.2f}'.format(right["Weighted " + mode].mean())
        print(f'Market share for {mode}: {100*simulated_values[f"Weighted {mode}"].mean():.2f}% ')
        print(f'90% Confidence interval: [-{left_str}, +{right_str}]%')
        print()

    ## Question 3: Actual Market Shares ##
    # We want to compute the weighted market share of each mode, by using the data
    # We just need pandas for this

    actual_market_shares = df[['travel_mode', 'Weight']].groupby('travel_mode').sum() / df['Weight'].sum()

    # use the command above to print in loop
    print("\n Actual market shares from data:")
    for i, mode in enumerate(['WALK', 'BIKE', 'PT', 'CAR']):
        print(f'Market share for {mode}: {100*actual_market_shares["Weight"].loc[i+1]:.2f}%')