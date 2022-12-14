import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, log, exp, Derive, DefineVariable
import biogeme.segmentation as seg
import numpy as np

if __name__ == '__main__':
    # load the data
    df = pd.read_csv('lpmc01.dat', sep='\t')
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

    #count the number of observations in each strata
    sample_segments = {
        k: v.sum() for k, v in filters.items()
    }

    total_sample = sum(sample_segments.values())

    # strata weights
    weights = {
        k: census[k] * total_sample / (v * pop_total) 
        for k, v in sample_segments.items()
    }

    # insert the weights into the database for each person
    for k, f in filters.items():
        df.loc[f, 'Weight'] = weights[k] 
    
    # check that the weights sum up to the sample size
    sum_weights = df['Weight'].sum()

    database = db.Database('LPMC', df)

    # define the variables
    WEIGHT=Variable('Weight')
    TRAVEL_MODE = Variable('travel_mode')
    FARETYPE = Variable('faretype')
    BUS_SCALE = Variable('bus_scale')
    AGE = Variable('age')
    DRIVING_LICENSE = Variable('driving_license')
    CAR_OWNERSHIP = Variable('car_ownership')
    DUR_WALKING = Variable('dur_walking')
    DUR_CYCLING = Variable('dur_cycling')
    DUR_PT_ACCESS = Variable('dur_pt_access')
    DUR_PT_RAIL = Variable('dur_pt_rail')
    DUR_PT_BUS = Variable('dur_pt_bus')
    DUR_PT_INT = Variable('dur_pt_int')
    PT_INTERCHANGES = Variable('pt_interchanges')
    DUR_DRIVING = Variable('dur_driving')
    COST_TRANSIT = Variable('cost_transit')
    COST_DRIVING_FUEL = Variable('cost_driving_fuel')
    COST_DRIVING_CCHARGE = Variable('cost_driving_ccharge')

    DRIVING_TRAFFIC_PERCENT = Variable('driving_traffic_percent')


    # Parameters to be estimated
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_PT = Beta('ASC_PT', 0, None, None, 0)
    ASC_WALK = Beta('ASC_WALK', 0, None, None, 0)
    ASC_BIKE = Beta('ASC_BIKE', 0, None, None, 1) #fixed to 0
    B_TIME_CAR = Beta('B_TIME_CAR', 0, None, None, 0)
    B_TIME_PT = Beta('B_TIME_PT', 0, None, None, 0)
    B_TIME_WALK = Beta('B_TIME_WALK', 0, None, None, 0)
    B_TIME_BIKE = Beta('B_TIME_BIKE', 0, None, None, 0)
    B_COST = Beta('B_COST', 0, None, None, 0)
    B_DRIVING_TRAFFIC_PERCENT = Beta('B_DRIVING_TRAFFIC_PERCENT', 0, None, None, 0)
    LAMBDA = Beta('LAMBDA', 1, None, None, 0)

    # Auxiliary variables
    COST_DRIVING = database.DefineVariable('cost_driving',(COST_DRIVING_FUEL + COST_DRIVING_CCHARGE))
    DUR_PT = database.DefineVariable('dur_pt',DUR_PT_ACCESS + DUR_PT_RAIL + DUR_PT_BUS + DUR_PT_INT)

    # changing the variables in accordance to model
    # Divide into age groups (0-16, 16-30, 30-60, 60+) and create dummy variables
    df['age_group'] = pd.cut(df['age'], [0, 16, 30, 60, 1000], labels=[0, 1, 2, 3])
    AGE_GROUP = Variable('age_group')
    segmentation_age = seg.DiscreteSegmentationTuple(variable=AGE_GROUP, mapping={0: 'young', 1: 'young_adult', 2: 'adult', 3: 'senior'})
    segmented_B_TIME_WALK = seg.segment_parameter(B_TIME_WALK, [segmentation_age])

    boxcox_dur_walking = models.boxcox(DUR_WALKING, LAMBDA)
    boxcox_dur_cycling = models.boxcox(DUR_CYCLING, LAMBDA)
    boxcox_dur_driving = models.boxcox(DUR_DRIVING, LAMBDA)
    boxcox_dur_pt = models.boxcox(DUR_PT, LAMBDA)

    V_WALK = ASC_WALK + segmented_B_TIME_WALK * boxcox_dur_walking
    V_BIKE = ASC_BIKE + B_TIME_BIKE * boxcox_dur_cycling
    V_PT = ASC_PT + B_TIME_PT * boxcox_dur_pt + B_COST * COST_TRANSIT
    V_PT_SCENARIO =  ASC_PT  + B_COST * COST_TRANSIT * 0.85 +  B_TIME_PT * boxcox_dur_pt
    V_CAR = ASC_CAR + B_TIME_CAR * boxcox_dur_driving + B_COST * COST_DRIVING + B_DRIVING_TRAFFIC_PERCENT * DRIVING_TRAFFIC_PERCENT
    V_CAR_SCENARIO = ASC_CAR  + B_COST * COST_DRIVING * 1.15 + B_DRIVING_TRAFFIC_PERCENT * DRIVING_TRAFFIC_PERCENT + B_TIME_CAR * boxcox_dur_driving
    

    # Associate utility functions with the numbering of alternatives
    V = {1: V_WALK, 2: V_BIKE, 3: V_PT, 4: V_CAR}

    # Associate the availability conditions with the alternatives
    av = {1: 1, 2: 1, 3: 1, 4: 1}

    # The choice model is a logit, with availability conditions
    logprob = models.loglogit(V, av, TRAVEL_MODE)

    # Create the Biogeme object
    biogeme = bio.BIOGEME(database, logprob, numberOfThreads=4)
    biogeme.modelName = 'model3'

    # Calculate the null log likelihood for reporting
    nullLogLikelihood = biogeme.calculateNullLoglikelihood(av)

    # Estimate the parameters
    results = biogeme.estimate()
    
    # Define the estimated probabilities
    PROB_WALK = exp(V_WALK)/(exp(V_WALK)+exp(V_BIKE)+exp(V_PT)+exp(V_CAR))
    PROB_BIKE = exp(V_BIKE)/(exp(V_WALK)+exp(V_BIKE)+exp(V_PT)+exp(V_CAR))
    PROB_PT = exp(V_PT)/(exp(V_WALK)+exp(V_BIKE)+exp(V_PT)+exp(V_CAR))
    PROB_CAR = exp(V_CAR)/(exp(V_WALK)+exp(V_BIKE)+exp(V_PT)+exp(V_CAR))
    
    PROB_WALK_IF_PT = exp(V_WALK)/(exp(V_WALK)+exp(V_BIKE)+exp(V_CAR)+exp(V_PT_SCENARIO))
    PROB_BIKE_IF_PT = exp(V_BIKE)/(exp(V_WALK)+exp(V_BIKE)+exp(V_CAR)+exp(V_PT_SCENARIO))
    PROB_PT_IF_PT = exp(V_PT_SCENARIO)/(exp(V_WALK)+exp(V_BIKE)+exp(V_CAR)+exp(V_PT_SCENARIO))
    PROB_CAR_IF_PT = exp(V_CAR)/(exp(V_WALK)+exp(V_BIKE)+exp(V_CAR)+exp(V_PT_SCENARIO))
    
    PROB_WALK_IF_CAR = exp(V_WALK)/(exp(V_WALK)+exp(V_BIKE)+exp(V_PT)+exp(V_CAR_SCENARIO))
    PROB_BIKE_IF_CAR = exp(V_BIKE)/(exp(V_WALK)+exp(V_BIKE)+exp(V_PT)+exp(V_CAR_SCENARIO))
    PROB_PT_IF_CAR = exp(V_PT)/(exp(V_WALK)+exp(V_BIKE)+exp(V_PT)+exp(V_CAR_SCENARIO))
    PROB_CAR_IF_CAR = exp(V_CAR_SCENARIO)/(exp(V_WALK)+exp(V_BIKE)+exp(V_PT)+exp(V_CAR_SCENARIO))
    
    # Calculate the desired values: elasticities, values of time, market shares.
    simulation1 = {
        'Weight': WEIGHT,
        'Prob. walk': PROB_WALK,
        'Prob. bike': PROB_BIKE,
        'Prob. PT': PROB_PT,
        'Prob. car': PROB_CAR
    }
    simulation2 = {
        'Weight': WEIGHT,
        'Prob. walk': PROB_WALK_IF_PT,
        'Prob. bike': PROB_BIKE_IF_PT,
        'Prob. PT': PROB_PT_IF_PT,
        'Prob. car': PROB_CAR_IF_PT
    }
    simulation3 = {
        'Weight': WEIGHT,
        'Prob. walk': PROB_WALK_IF_CAR,
        'Prob. bike': PROB_BIKE_IF_CAR,
        'Prob. PT': PROB_PT_IF_CAR,
        'Prob. car': PROB_CAR_IF_CAR
    }
    simulation_of_utilities = {
        'Weight': WEIGHT,
        'VOT PT': Derive(V_PT, 'dur_pt')/B_COST,
        'VOT car': Derive(V_CAR, 'dur_driving')/B_COST,
        'Elast. dir. PT': Derive(PROB_PT, 'cost_transit')*COST_TRANSIT/PROB_PT,
        'Elast. dir. car': Derive(PROB_CAR, 'cost_driving')*COST_DRIVING/PROB_CAR,
        'Elast. cross PT-car': Derive(PROB_CAR, 'cost_transit')*COST_TRANSIT/PROB_CAR,
        'Elast. cross PT-bike': Derive(PROB_BIKE, 'cost_transit')*COST_TRANSIT/PROB_BIKE,
        'Elast. cross PT-walk': Derive(PROB_WALK, 'cost_transit')*COST_TRANSIT/PROB_WALK,
        'Elast. cross car-PT': Derive(PROB_PT, 'cost_driving')*COST_DRIVING/PROB_PT,
        'Elast. cross car-bike': Derive(PROB_PT, 'cost_driving')*COST_DRIVING/PROB_PT,
        'Elast. cross car-walk': Derive(PROB_PT, 'cost_driving')*COST_DRIVING/PROB_PT,
        'Weighted prob. car': WEIGHT*PROB_CAR,
        'Weighted prob. PT': WEIGHT*PROB_PT,
        'Weighted prob. walk': WEIGHT*PROB_WALK,
        'Weighted prob. bike': WEIGHT*PROB_BIKE
    }
    biosim1 = bio.BIOGEME(database, simulation1)
    simulated_values1 = biosim1.simulate(results.getBetaValues())
    
    biosim2 = bio.BIOGEME(database, simulation2)
    simulated_values2 = biosim2.simulate(results.getBetaValues())
    
    biosim3 = bio.BIOGEME(database, simulation3)
    simulated_values3 = biosim3.simulate(results.getBetaValues())
    
    biosim4 = bio.BIOGEME(database, simulation_of_utilities)
    simulated_values4 = biosim4.simulate(results.getBetaValues())
    
    simulated_values1['Weighted PT'] = (
    simulated_values1['Weight'] * 
    simulated_values1['Prob. PT']
    )
    simulated_values1['Weighted car'] = (
    simulated_values1['Weight'] * 
    simulated_values1['Prob. car']
    )
    simulated_values1['Weighted walk'] = (
    simulated_values1['Weight'] * 
    simulated_values1['Prob. walk']
    )
    simulated_values1['Weighted bike'] = (
    simulated_values1['Weight'] * 
    simulated_values1['Prob. bike']
    )
    
    simulated_values2['Weighted PT'] = (
    simulated_values2['Weight'] * 
    simulated_values2['Prob. PT']
    )
    simulated_values2['Weighted car'] = (
    simulated_values2['Weight'] * 
    simulated_values2['Prob. car']
    )
    simulated_values2['Weighted walk'] = (
    simulated_values2['Weight'] * 
    simulated_values2['Prob. walk']
    )
    simulated_values2['Weighted bike'] = (
    simulated_values2['Weight'] * 
    simulated_values2['Prob. bike']
    )
    
    simulated_values3['Weighted PT'] = (
    simulated_values3['Weight'] * 
    simulated_values3['Prob. PT']
    )
    simulated_values3['Weighted car'] = (
    simulated_values3['Weight'] * 
    simulated_values3['Prob. car']
    )
    simulated_values3['Weighted walk'] = (
    simulated_values3['Weight'] * 
    simulated_values3['Prob. walk']
    )
    simulated_values3['Weighted bike'] = (
    simulated_values3['Weight'] * 
    simulated_values3['Prob. bike']
    )
    
    simulated_values4['Weighted VOT PT'] = (
    simulated_values4['Weight'] *
    simulated_values4['VOT PT']
    )
    simulated_values4['Weighted VOT car'] = (
    simulated_values4['Weight'] *
    simulated_values4['VOT car']
    )    
    
    # Printing the desired values for the "Forecasting" part
    for mode in ['walk', 'bike', 'PT', 'car']:
        print(f'Market share for {mode}: {100*simulated_values1[f"Weighted {mode}"].mean():.2f}% ')
        print()
    for mode in ['walk', 'bike', 'PT', 'car']:
        print(f'Market share with decrease of PT cost for {mode}: {100*simulated_values2[f"Weighted {mode}"].mean():.2f}% ')
        print()
    for mode in ['walk', 'bike', 'PT', 'car']:
        print(f'Market share with increase of car cost for {mode}: {100*simulated_values3[f"Weighted {mode}"].mean():.2f}% ')
        print()
    print('Average value of time in PT: ', round(100*simulated_values4['Weighted VOT PT'].mean())/100, ' GBP/hour')
    print()
    print('Average value of time in car: ', round(100*simulated_values4['Weighted VOT car'].mean())/100, ' GBP/hour','\n')
    
    for mode in ['walk', 'bike', 'PT', 'car']:
        print(f'Normalizing factor of {mode} prob. elast.:',(simulated_values4[f'Weighted prob. {mode}']).sum(),'\n')
    for mode in ['walk', 'bike', 'PT', 'car']:
        for prob in ['PT', 'car']:
            if prob==mode:
                print(f'Direct aggregate elasticity of {mode} cost: ', (simulated_values4[f'Elast. dir. {mode}']*simulated_values4[f'Weighted prob. {mode}']).sum()/(simulated_values4[f'Weighted prob. {mode}']).sum(),'\n')
            else:
                print(f'Cross aggregate elasticity of {prob} cost and {mode} prob.: ', (simulated_values4[f'Elast. cross {prob}-{mode}']*simulated_values4[f'Weighted prob. {mode}']).sum()/(simulated_values4[f'Weighted prob. {mode}']).sum(),'\n')

    for mode in ['walk', 'bike', 'PT', 'car']:
        print(f'Difference of {mode} logprob. for PT cost divided by log(1/1.15):', np.log(simulated_values2[f"Weighted {mode}"].mean()/simulated_values1[f"Weighted {mode}"].mean())/(np.log(1/1.15)))
        print()
    for mode in ['walk', 'bike', 'PT', 'car']:
        print(f'Difference of {mode} logprob. for car cost divided by log(1/0.85):', np.log(simulated_values3[f"Weighted {mode}"].mean()/simulated_values1[f"Weighted {mode}"].mean())/(np.log(1/0.85)))
        print()
