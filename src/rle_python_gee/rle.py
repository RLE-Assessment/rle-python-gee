# https://iucnrle.org/rle-categ-and-criteria
rle_criteria = {
    "A": {
        "description": "Reduction in geographic distribution",
        "threshold": 0.5
    },
    "B": {
        "description": "Restricted geographic distribution",
        "threshold": 0.5
    },
    "C": {
        "description": "Environmental degradation",
        "threshold": 0.5
    },
    "D": {
        "description": "Disruption of biotic processes and interactions",
        "threshold": 0.5
    },
    "E": {
        "description": "Quantitative analysis that estimates the probability of ecosystem collapse",
        "threshold": 0.5
    }
}

# Source: https://iucnrle.org/rle-categ-and-criteria
rle_categories = [
    {
        "name": "Collapsed",
        "abbreviation": "CO",
        "threatened": True,
        "background_color": "black",
    },
    {
        "name": "Critically Endangered",
        "abbreviation": "CR",
        "threatened": True,
        "background_color": "red",
    },
    {
        "name": "Endangered",
        "abbreviation": "EN",
        "threatened": True,
        "background_color": "orange",
    },
    {
        "name": "Vulnerable",
        "abbreviation": "VU",
        "threatened": True,
        "background_color": "yellow",
    },
    {
        "name": "Near Threatened",
        "abbreviation": "NT",
        "background_color": "green",
    },
    {
        "name": "Least Concern",
        "abbreviation": "LC",
        "background_color": "darkgreen",
    },
    {
        "name": "Data Deficient",
        "abbreviation": "DD",
        "background_color": "lightgray",    
    },
    {
        "name": "Not Evaluated",
        "abbreviation": "NE",
        "background_color": "white",
    }
]