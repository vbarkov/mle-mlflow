from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, 
    SplineTransformer, 
    QuantileTransformer, 
    RobustScaler,
    PolynomialFeatures,
    KBinsDiscretizer,
)

def create_preprocessor():
    cat_columns = ["type", "payment_method", "internet_service", "gender"]
    num_columns = ["monthly_charges", "total_charges"]
    n_knots = 3
    degree_spline = 4
    n_quantiles=100
    degree = 3
    n_bins = 5
    encode = 'ordinal'
    strategy = 'uniform'
    subsample = None

    # SplineTransformer
    encoder_spl = SplineTransformer(n_knots=n_knots,
                                   degree=degree_spline
    )

    # QuantileTransformer
    encoder_q = QuantileTransformer(n_quantiles=n_quantiles)

    # RobustScaler
    encoder_rb = RobustScaler()

    # PolynomialFeatures
    encoder_pol = PolynomialFeatures(degree=degree)

    # KBinsDiscretizer
    encoder_kbd = KBinsDiscretizer(n_bins=n_bins,
                                   encode=encode,
                                   strategy=strategy,
                                   subsample=subsample)
    
    encoder_oh = OneHotEncoder(categories="auto",
                               handle_unknown="ignore",
                               max_categories=10,
                               drop="first",
                               sparse_output=False
    )

    numeric_transformer = ColumnTransformer(
        transformers=[
            ('spl', encoder_spl, num_columns), 
            ('q', encoder_q, num_columns), 
            ('rb', encoder_rb, num_columns), 
            ('pol', encoder_pol, num_columns), 
            ('kbd', encoder_kbd, num_columns)
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", encoder_oh)
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_columns), 
            ('cat', categorical_transformer, cat_columns)], 
        n_jobs=-1
    )

    return preprocessor