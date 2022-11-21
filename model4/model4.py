import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, log
import biogeme.segmentation as seg
import numpy as np

if __name__ == '__main__':
    # load the data
    df = pd.read_csv('../data/lpmc01.dat', sep='\t')
    database = db.Database('LPMC', df)

    # define the variables
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

    # Nesting: separate by motorized and non-motorized
    mu = Beta('mu', 1, 1, 10, 0) # min=1, max=10
    non_motorized = 1.0, [1,2]
    motorized = mu, [3,4]
    nests = motorized, non_motorized

    # The choice model is a logit, with availability conditions
    logprob = models.lognested(V, av, nests, TRAVEL_MODE)

    # Create the Biogeme object
    biogeme = bio.BIOGEME(database, logprob, numberOfThreads=4)
    biogeme.modelName = 'model4'

    # Calculate the null log likelihood for reporting
    nullLogLikelihood = biogeme.calculateNullLoglikelihood(av)

    # Estimate the parameters
    results = biogeme.estimate()

    # Get the results in a pandas table
    pandasResults = results.getEstimatedParameters()
    likelihood = results.data.logLike
    print(pandasResults)
    print(f'Null log likelihood: {nullLogLikelihood}')
    print(f'Likelihood: {likelihood}')

    #print(results.getLaTeX())
