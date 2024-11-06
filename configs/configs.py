import os

PDF_PATH=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data','MINT05-Mint-123-issuu.pdf')

BREAKPOINT_DEFAULTS = {
    "percentile": {
        "default": 90,        # Default threshold for outlier detection
        "range_min": 75,      # Minimum reasonable threshold for flexibility
        "range_max": 99       # Maximum threshold for stricter outliers
    },
    "standard_deviation": {
        "default": 2,         # Default deviation threshold
        "range_min": 1,       # Minimum for lighter outlier detection
        "range_max": 4        # Maximum for stricter outliers
    },
    "interquartile": {
        "default": 1.5,       # Standard IQR multiplier for moderate outliers
        "range_min": 0.5,     # Minimum for a more lenient range
        "range_max": 2.5      # Maximum for stricter outliers
    },
    "gradient": {
        "default": 95,        # Default threshold for gradient adjustments
        "range_min": 10,      # Broadened range for capturing smaller gradients
        "range_max": 99       # Upper limit for stricter gradients
    }
}