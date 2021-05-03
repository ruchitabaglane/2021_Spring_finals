import pandas as pd


def cleanData(dataSet: pd.DataFrame) -> pd.DataFrame:
    dataSet_copy = dataSet.copy()
    dataSet_copy['State'].replace({"Jammu and Kashmir": "Jammu & Kashmir", "Delhi": "NCT OF Delhi", \
                                   "Dadra and Nagar Haveli and Daman and Diu": "Dadra & Nagar Haveli and Daman & Diu",\
                                   "Andaman and Nicobar Islands": "Andaman & Nicobar Islands",\
                                   "A.& N.Islands": "Andaman & Nicobar Islands"}, inplace=True)

    return dataSet_copy


def getStateGovernments(stateElect: pd.DataFrame) -> pd.DataFrame:
    groupByStateParty = stateElect.groupby(['State', 'Party']).agg({'Votes': 'sum'})
    groupByStateParty.columns = ['Votes']
    groupByStateParty = groupByStateParty.reset_index()

    stateRulingParty = groupByStateParty.sort_values(['Votes'], ascending=False)
    stateRulingParty = stateRulingParty.drop_duplicates(subset=['State'], keep='first')

    return stateRulingParty


def getStateVaccineRecords(stateVaccines: pd.DataFrame) -> pd.DataFrame:
    stateVaccines = cleanData(stateVaccines)

    stateVaccines = stateVaccines.loc[stateVaccines['State'] != 'India']
    latestVaccineInfo = stateVaccines.sort_values(['State'], ascending=False)
    latestVaccineInfo = latestVaccineInfo.drop_duplicates(subset=['State'], keep='first')

    return latestVaccineInfo


def calcRateOfVaccination(vaccineRecords: pd.DataFrame, statePop: pd.DataFrame) -> pd.DataFrame:
    vaccinePopulation = statePop[['State', 'Population_2019']].merge(vaccineRecords[['Updated On', 'State', 'Total Individuals Vaccinated']], on='State')
    print(vaccinePopulation)


if __name__ == '__main__':
    stateElections = pd.read_csv("./StateElectionData.csv")
    stateVaccinations = pd.read_csv("./covid_vaccine_statewise.csv")
    statePopulation = pd.read_csv("./statePopulationIndia.csv")
    statePopulation = cleanData(statePopulation)

    stateGovernments = getStateGovernments(stateElections)
    stateVaccineRecords = getStateVaccineRecords(stateVaccinations)

    rateOfVaccination = calcRateOfVaccination(stateVaccineRecords, statePopulation)

    #print(stateGovernments.State)
    #print(stateVaccineRecords.columns)
    #print(statePopulation.State)
