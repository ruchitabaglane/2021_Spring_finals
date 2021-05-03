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
                          stateGovernments: pd.DataFrame) -> pd.DataFrame:
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
    stateElections = pd.read_csv("./StateElectionData.csv")
    stateVaccinations = pd.read_csv("./covid_vaccine_statewise.csv")
    statePopulation = pd.read_csv("./statePopulationIndia.csv")
    statePopulation = cleanData(statePopulation)

    stateGovernments = getStateGovernments(stateElections)
    stateVaccineRecords = getStateVaccineRecords(stateVaccinations)

    rateOfVaccination_vs_party = calcRateOfVaccination(stateVaccineRecords, statePopulation, stateGovernments)

    # print (rateOfVaccination_vs_party[1])

    rateOfVaccination_vs_party[0].plot(x='Party', y='VaccinationRate', kind='bar')
    plt.show()

    vaccinationRate_by_rulingParty = StateClassificationByRulingParty(stateGovernments, rateOfVaccination_vs_party[1])

    fig, ax = plt.subplots(figsize=(100, 30))
    for key, grp in vaccinationRate_by_rulingParty.groupby(['Party']):
        ax.bar(grp['State'], grp['VaccinationRate'], label=key)
    ax.legend()
    plt.show()


