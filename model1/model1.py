import pandas as pd
import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable

if __name__ == '__main__':
    # load the data
    df = pd.read_csv('../data/lpmc01.dat', sep='\t')
    database = db.Database('LPMC', df)

    print(database.data.head(0))

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

    # Parameters to be estimated
    ASC_CAR = Beta('ASC_CAR', 0, None, None, 0)
    ASC_PT = Beta('ASC_PT', 0, None, None, 0)
    ASC_WALK = Beta('ASC_WALK', 0, None, None, 0)
    ASC_BIKE = Beta('ASC_BIKE', 0, None, None, 1)
    B_TIME_CAR = Beta('B_TIME_CAR', 0, None, None, 0)
    B_TIME_PT = Beta('B_TIME_PT', 0, None, None, 0)
    B_TIME_WALK = Beta('B_TIME_WALK', 0, None, None, 0)
    B_TIME_BIKE = Beta('B_TIME_BIKE', 0, None, None, 0)
    B_COST = Beta('B_COST', 0, None, None, 0)

    # Auxiliary variables
    COST_DRIVING = COST_DRIVING_FUEL + COST_DRIVING_CCHARGE
    DUR_PT = DUR_PT_ACCESS + DUR_PT_RAIL + DUR_PT_BUS + DUR_PT_INT * PT_INTERCHANGES

    # Definition of utility functions
    V_WALK = ASC_WALK + B_TIME_WALK * DUR_WALKING
    V_BIKE = ASC_BIKE + B_TIME_BIKE * DUR_CYCLING
    V_PT = ASC_PT + B_TIME_PT * DUR_PT + B_COST * COST_TRANSIT
    V_CAR = ASC_CAR + B_TIME_CAR * DUR_DRIVING + B_COST * COST_DRIVING

    # Associate utility functions with the numbering of alternatives
    V = {1: V_WALK, 2: V_BIKE, 3: V_PT, 4: V_CAR}

    # Associate the availability conditions with the alternatives
    av = {1: 1, 2: 1, 3: 1, 4: 1}

    # The choice model is a logit, with availability conditions
    logprob = models.loglogit(V, av, TRAVEL_MODE)

    # Create the Biogeme object
    biogeme = bio.BIOGEME(database, logprob)
    biogeme.modelName = 'model0'

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
