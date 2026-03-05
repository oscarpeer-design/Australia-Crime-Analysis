from ETL import UseExtract_Excel, Load_NeuralNetwork
#from Neural_Network_Application import NeuralNetwork

def analyse_crime_data():
    data = UseExtract_Excel("Crime Stats Per Suburb", "Australia_Crime_Statistics.xlsx", ["Suburb", "Total Crimes Per Population 2025", "High Risk Alcohol Consumption", "Education Levels Year 9 and Below", "Proportion Low Income Households", "Non Workforce Participation"], True, True)
    total_crimes = data[0]

    daily_alcohol = data[1]
    education_levels = data[2]
    low_income = data[3]
    unemployment = data[4]

    daily_alcohol_X = [[x] for x in daily_alcohol]
    education_X = [[x] for x in education_levels]
    low_income_X = [[x] for x in low_income]
    unemployment_X = [[x] for x in unemployment]
    max_crime = max(total_crimes)
    all_factors_X = [
    [daily_alcohol[i], education_levels[i], low_income[i], unemployment[i]]
    for i in range(len(total_crimes))
    ]
    total_crimes_scaled = [y / max_crime for y in total_crimes]

    lr = 0.001
    epochs = 5000
    verbose = True
    size_output = 1
    size_hidden = 4
    #Load individually
    alcohol_impact = Load_NeuralNetwork(daily_alcohol_X, total_crimes_scaled, lr, size_hidden, size_output)
    education_impact = Load_NeuralNetwork(education_X, total_crimes_scaled, lr, size_hidden, size_output)
    poverty_impact = Load_NeuralNetwork(low_income_X, total_crimes_scaled, lr, size_hidden, size_output)
    unemployment_impact = Load_NeuralNetwork(unemployment_X, total_crimes_scaled, lr, size_hidden, size_output) #Note that this should use tanh activation function instead of ReLU
    #Load all variables
    lr = 0.0005
    size_hidden = 6
    all_factors = Load_NeuralNetwork(all_factors_X, total_crimes_scaled, lr, size_hidden, size_output)
    #Compare variables
    print("==== Impact of the percentage of the population who drink alcohol at risky levels on crimes: ====\n")
    alcohol_impact.train_and_observe_results(epochs, verbose)
    print("==== Impact of the percentage of the population that didn't finish year 9 on crimes: ====\n")
    education_impact.train_and_observe_results(epochs, verbose)
    print("==== Impact of the percentage of households that earn below $650 per week on crime: ====\n")
    poverty_impact.train_and_observe_results(epochs, verbose)
    print("==== Impact of a suburb's non-work participation rate, (excluding those who do not declare that they are in the workforce), on crime: ====\n")
    unemployment_impact.train_and_observe_results(epochs, verbose)
    #Look at all variables collectively
    epochs = 8000
    print("==== Impact of all variables collectively on crime in Australia's highest crime suburbs: ====\n")
    all_factors.train_and_observe_results(epochs, verbose)

if __name__ == '__main__':
    analyse_crime_data()
