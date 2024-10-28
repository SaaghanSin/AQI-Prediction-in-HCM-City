from sklearn.preprocessing import StandardScaler

AQI_BREAKPOINTS = {
    "PM2.5": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500.0, 301, 500),
    ],
    "O3": [
        (0.0, 54.0, 0, 50),
        (55.0, 70.0, 51, 100),
        (71.0, 85.0, 101, 150),
        (86.0, 105.0, 151, 200),
        (106.0, 200.0, 201, 300),
    ],
    "CO": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
    ],
    "SO2": [
        (0.0, 35.0, 0, 50),
        (36.0, 75.0, 51, 100),
        (76.0, 185.0, 101, 150),
        (186.0, 304.0, 151, 200),
        (305.0, 604.0, 201, 300),
    ],
    "NO2": [
        (0.0, 53.0, 0, 50),
        (54.0, 100.0, 51, 100),
        (101.0, 360.0, 101, 150),
        (361.0, 649.0, 151, 200),
        (650.0, 1249.0, 201, 300),
    ],
}


def calculate_aqi(concentration, breakpoints):
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (
                concentration - c_low
            ) + i_low
            return aqi
    return None


def preprocess_data(data):
    data["AQI_PM2.5"] = data["PM2.5"].apply(
        lambda x: calculate_aqi(x, AQI_BREAKPOINTS["PM2.5"])
    )
    data["AQI_O3"] = data["O3"].apply(lambda x: calculate_aqi(x, AQI_BREAKPOINTS["O3"]))
    data["AQI_CO"] = data["CO"].apply(lambda x: calculate_aqi(x, AQI_BREAKPOINTS["CO"]))
    data["AQI_SO2"] = data["SO2"].apply(
        lambda x: calculate_aqi(x, AQI_BREAKPOINTS["SO2"])
    )
    data["AQI_NO2"] = data["NO2"].apply(
        lambda x: calculate_aqi(x, AQI_BREAKPOINTS["NO2"])
    )
    data["AQI"] = data[["AQI_PM2.5", "AQI_O3", "AQI_CO", "AQI_SO2", "AQI_NO2"]].max(
        axis=1
    )
    data = data.drop(columns=["AQI_PM2.5", "AQI_O3", "AQI_CO", "AQI_SO2", "AQI_NO2"])

    features = data[
        ["TSP", "PM2.5", "O3", "CO", "NO2", "SO2", "Temperature", "Humidity"]
    ]
    target = data["AQI"]

    return features, target
