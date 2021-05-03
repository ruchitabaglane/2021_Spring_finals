import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def cleanData(dataSet: pd.DataFrame) -> pd.DataFrame:
    dataSet_copy = dataSet.copy()
    dataSet_copy['State'].replace({"Jammu and Kashmir": "Jammu & Kashmir", "Delhi": "NCT OF Delhi", \
                                   "Dadra and Nagar Haveli and Daman and Diu": "Dadra & Nagar Haveli and Daman & Diu", \
                                   "Andaman and Nicobar Islands": "Andaman & Nicobar Islands", \
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
    stateVaccines['Updated On'] = pd.to_datetime(stateVaccines['Updated On'], infer_datetime_format=True)
    latestVaccineInfo = stateVaccines.sort_values(['Updated On'], ascending=False)
    latestVaccineInfo = latestVaccineInfo.drop_duplicates(subset=['State'], keep='first')
    latestVaccineInfo = latestVaccineInfo.reset_index()
    return latestVaccineInfo


def calcVaccinationRate(Total_Individuals_Vaccinated: float, Population_2019: float):
    VaccinationRate = (Total_Individuals_Vaccinated / Population_2019) * 100
    return VaccinationRate


def calcRateOfVaccination(vaccineRecords: pd.DataFrame, statePop: pd.DataFrame,
                          stateGovernments: pd.DataFrame) :
    vaccinePopulation = statePop[['State', 'Population_2019']].merge(
                        vaccineRecords[['Updated On', 'State', 'Total Individuals Vaccinated']], on='State')
    vaccinePopulation['VaccinationRate'] = vaccinePopulation.apply(
        lambda row: calcVaccinationRate(row['Total Individuals Vaccinated'], row['Population_2019']), axis=1)

    vaccinePopulation1 = statePop[['State', 'Population_2019']].merge(
                         vaccineRecords[['Updated On', 'State', 'Total Individuals Vaccinated']], on='State').merge(
                         stateGovernments[['State', 'Party']], on='State')

    grouped = vaccinePopulation1.groupby('Party').agg({'Population_2019': 'sum', 'Total Individuals Vaccinated': 'sum'})
    grouped = grouped.reset_index()
    grouped['VaccinationRate'] = grouped.apply(
        lambda row: calcVaccinationRate(row['Total Individuals Vaccinated'], row['Population_2019']), axis=1)

    return (grouped, vaccinePopulation)


def StateClassificationByRulingParty(stateGovernments: pd.DataFrame, rateOfVaccination: pd.DataFrame) -> pd.DataFrame:
    vaccinationRate_by_rulingParty = stateGovernments[['State', 'Party']].merge(
        rateOfVaccination[['State', 'Population_2019', 'Total Individuals Vaccinated', 'VaccinationRate']], on='State')
    return vaccinationRate_by_rulingParty


if __name__ == '__main__':
    # Loading all the input files
    stateElections = pd.read_csv("./StateElectionData.csv")
    stateVaccinations = pd.read_csv("./covid_vaccine_statewise.csv")
    statePopulation = pd.read_csv("./statePopulationIndia.csv")
    # CLeaning the data to have uniform names of states across the input files.
    statePopulation = cleanData(statePopulation)

    # Fetching the currently ruling parties in each state of India.
    stateGovernments = getStateGovernments(stateElections)
    # Fetching the statewise vaccination records for the latest dates.
    stateVaccineRecords = getStateVaccineRecords(stateVaccinations)

    # Calculating the rate of vaccination in each state.
    rateOfVaccination_vs_party = calcRateOfVaccination(stateVaccineRecords, statePopulation, stateGovernments)

    # Plotting bar plot for States within each party vs vaccination rate.
    rateOfVaccination_vs_party[0].plot(x='Party', y='VaccinationRate', kind='bar')
    plt.title(label="Vaccination Rates per Political Party",
              fontsize=15)
    plt.xlabel('Political Parties')
    plt.ylabel('Vaccination Rates')
    plt.show()

    # Classifying the states by ruling parties.
    vaccinationRate_by_rulingParty = StateClassificationByRulingParty(stateGovernments, rateOfVaccination_vs_party[1])

    # plotting bar plot for States vs Vaccination rate and coloring the bar plot to represent the ruling party.
    fig, ax = plt.subplots(figsize=(100, 30))
    for key, grp in vaccinationRate_by_rulingParty.groupby(['Party']):
        ax.bar(grp['State'], grp['VaccinationRate'], label=key)

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(55)
    ax.legend(loc=2, prop={'size': 40})
    ax.tick_params(axis='x', labelrotation=90)
    plt.title(label="States Vs Vaccination Rates",
              fontsize=95)
    plt.xlabel('States', fontsize=55)
    plt.ylabel('Vaccination Rates', fontsize=55)
    plt.show()


