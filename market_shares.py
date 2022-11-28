import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, log
import biogeme.segmentation as seg
import numpy as np

if __name__ == '__main__':
    # load the data
    df = pd.read_csv('Dataset.dat', sep= '\t')  
    #df = pd.read_csv('../data/lpmc01.dat', sep='\t')
    database = db.Database('LPMC', df)
    
    #define Variables 
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
    FEMALE = Variable('female')
    
    #define new Variables for strata 
    YG_WOMEN = database.DefineVariable(
        'YG_WOMEN', ((FEMALE>0) * (AGE < 41))
    )

    OLD_WOMEN = database.DefineVariable(
        'OLD_WOMEN', ((FEMALE>0) * (AGE > 40))
    )

    YG_MEN = database.DefineVariable(
        'YG_MEN', ((1-(FEMALE>0)) * (AGE < 41))
    )
    
    OLD_MEN = database.DefineVariable(
        'OLD_MEN', ((1-(FEMALE>0)) * (AGE > 40))
    )   
    
    #nb in the population
    POP = np.array([2599058,1765143,2676249,1633263])
    POP_TOT=sum(POP)
    
    #fraction in the population ie FRAP
    FRAP=POP/POP_TOT

    #nb in the sample
    SAMPLE_SIZE = 5000
    NB = np.array([sum(df['YG_WOMEN']),sum(df['OLD_WOMEN']),sum(df['YG_MEN']),sum(df['OLD_MEN'])])

    #fraction of group in the sample ie FRAS
    FRAS = NB/SAMPLE_SIZE
    
    #weights ; prob of being drawn = R(i,x)
    WGHT = (FRAS*SAMPLE_SIZE)/(FRAP*POP_TOT)
    
    #calculate table with size and weights of the sample
    d = {'Strata': ['YG_WOMEN', 'OLD_WOMEN', 'YG_MEN','OLD_MEN'],
    'Size': NB,
    'Weight': WGHT
    }
    table = pd.DataFrame(data=d)
    
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
    COST_DRIVING = (COST_DRIVING_FUEL + COST_DRIVING_CCHARGE)
    DUR_PT = DUR_PT_ACCESS + DUR_PT_RAIL + DUR_PT_BUS + DUR_PT_INT

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
    V_CAR = ASC_CAR + B_TIME_CAR * boxcox_dur_driving + B_COST * COST_DRIVING + B_DRIVING_TRAFFIC_PERCENT * DRIVING_TRAFFIC_PERCENT

    # Associate utility functions with the numbering of alternatives
    V = {1: V_WALK, 2: V_BIKE, 3: V_PT, 4: V_CAR}

    # Associate the availability conditions with the alternatives
    av = {1: 1, 2: 1, 3: 1, 4: 1}

    
    #calculate predicted market share of each mode
    prob_WALK = models.loglogit(V, av, 0)
    prob_BK = models.loglogit(V, av, 1)
    prob_PT = models.loglogit(V, av, 2)
    prob_CAR = models.loglogit(V, av, 3)
    
    #create a dictionnary to simulate quantities
    simulate = {
        'weight': WGHT,
        'Prob. WALK': prob_WALK,
        'Prob. BK': prob_BK,
        'Prob. PT': prob_PT,
        'Prob. CAR': prob_CAR,
    }

    biosim = bio.BIOGEME(database, simulate)
    simulated_values = biosim.simulate(results.getBetaValues())
    
    #perform simulation + confidence intervals
    betas = biogeme.freeBetaNames()
    b = results.getBetasForSensitivityAnalysis(betas)
    left, right = biosim.confidenceIntervals(b, 0.9)
    
    #calculate means of weights and then the Market shares linked
    #Walk
    simulated_values['Weighted prob. WALK'] = (
        simulated_values['weight'] * simulated_values['Prob. WALK']
    )
    left['Weighted prob. WALK'] = left['weight'] * left['Prob. WALK']
    right['Weighted prob. WALK'] = (
                right['weight'] * right['Prob. WALK']
    )

    marketShare_walk = IndicatorTuple(
    value=simulated_values['Weighted prob. WALK'].mean(),
    lower=left['Weighted prob. WALK'].mean(),
    upper=right['Weighted prob. WALK'].mean()
    )
    #print the result 
    print(
        f'Market share for walk: {100*marketShare_walk.value:.1f}% '
        f'[{100*marketShare_walk.lower:.1f}%, '
        f'{100*marketShare_walk.upper:.1f}%]'
    ) 
    #calculate means of weights and then the Market shares linked
    #Bike
     simulated_values['Weighted prob. BK'] = (
        simulated_values['weight'] * simulated_values['Prob. BK']
    )
    left['Weighted prob. BK'] = left['weight'] * left['Prob. BK']
    right['Weighted prob. BK'] = (
                right['weight'] * right['Prob. BK']
    )

    marketShare_bike = IndicatorTuple(
    value=simulated_values['Weighted prob. BK'].mean(),
    lower=left['Weighted prob. BK'].mean(),
    upper=right['Weighted prob. BK'].mean()
    )
    #print the result 
    print(
        f'Market share for bike: {100*marketShare_bike.value:.1f}% '
        f'[{100*marketShare_bike.lower:.1f}%, '
        f'{100*marketShare_bike.upper:.1f}%]'
    )
    #calculate means of weights and then the Market shares linked
    #PT
    simulated_values['Weighted prob. PT'] = (
        simulated_values['weight'] * simulated_values['Prob. PT']
    )
    left['Weighted prob. PT'] = left['weight'] * left['Prob. PT']
    right['Weighted prob. PT'] = (
                right['weight'] * right['Prob. PT']
    )

    marketShare_pt = IndicatorTuple(
    value=simulated_values['Weighted prob. PT'].mean(),
    lower=left['Weighted prob. PT'].mean(),
    upper=right['Weighted prob. PT'].mean()
    )
    #print the result 
    print(
        f'Market share for pt: {100*marketShare_pt.value:.1f}% '
        f'[{100*marketShare_pt.lower:.1f}%, '
        f'{100*marketShare_pt.upper:.1f}%]'
    )
    #calculate means of weights and then the Market shares linked
    #CAR
    simulated_values['Weighted prob. CAR'] = (
        simulated_values['weight'] * simulated_values['Prob. CAR']
    )
    left['Weighted prob. CAR'] = left['weight'] * left['Prob. CAR']
    right['Weighted prob. CAR'] = (
                right['weight'] * right['Prob. CAR']
    )

    marketShare_car = IndicatorTuple(
    value=simulated_values['Weighted prob. CAR'].mean(),
    lower=left['Weighted prob. CAR'].mean(),
    upper=right['Weighted prob. CAR'].mean()
    )
    #print the result 
    print(
        f'Market share for car: {100*marketShare_car.value:.1f}% '
        f'[{100*marketShare_car.lower:.1f}%, '
        f'{100*marketShare_car.upper:.1f}%]'
    )
    
    #market shares using actual choices
    