import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def cleanData(dataSet: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to clean the data and make it consistent for further processing.
    :param dataSet: Dataframe to be cleaned
    :return: Dataframe with appropriate transformations.
    """
    dataSet_copy = dataSet.copy()
    dataSet_copy['State'].replace({"Jammu and Kashmir": "Jammu & Kashmir", "NCT OF Delhi": "Delhi", \
                                   "Dadra and Nagar Haveli and Daman and Diu": "Dadra & Nagar Haveli and Daman & Diu", \
                                   "Andaman and Nicobar Islands": "Andaman & Nicobar Islands", \
                                   "A.& N.Islands": "Andaman & Nicobar Islands"}, inplace=True)

    return dataSet_copy


def getStateGovernments(stateElect: pd.DataFrame) -> pd.DataFrame:
    """
    This function aims to find the Ruling Party in each state and the votes it received.
    :param stateElect: Dataframe with election data
    :return: Dataframe with states and the corresponding political party in power.
    """
    groupByStateParty = stateElect.groupby(['State', 'Party']).agg({'Votes': 'sum'})
    groupByStateParty.columns = ['Votes']
    groupByStateParty = groupByStateParty.reset_index()

    stateRulingParty = groupByStateParty.sort_values(['Votes'], ascending=False)
    stateRulingParty = stateRulingParty.drop_duplicates(subset=['State'], keep='first')

    return stateRulingParty


def getStateVaccineRecords(stateVaccines: pd.DataFrame) -> pd.DataFrame:
    """
    This function is used to find the vaccine data for each state based on the latest updated date.
    :param stateVaccines: Dataframe with vaccine data statewise.
    :return: Dataframe with sates and corresponding number of individuals vaccinated as per latest updated date.
    """
    stateVaccines = cleanData(stateVaccines)
    stateVaccines = stateVaccines.loc[stateVaccines['State'] != 'India']
    stateVaccines['Updated On'] = pd.to_datetime(stateVaccines['Updated On'], infer_datetime_format=True)
    latestVaccineInfo = stateVaccines.sort_values(['Updated On'], ascending=False)
    latestVaccineInfo = latestVaccineInfo.drop_duplicates(subset=['State'], keep='first')
    latestVaccineInfo = latestVaccineInfo.reset_index()
    return latestVaccineInfo


def calcVaccinationRate(Total_Individuals_Vaccinated: float, Population_2019: float)-> float:
    """
    The purpose of this function is to calculate the VaccineRate in each state by finding the ratio of Total Individuals Vaccinated and Population.
    :param Total_Individuals_Vaccinated: Number of individuals vaccinated
    :param Population_2019: Population of the state
    :return: VaccineRate(float)
    """
    VaccinationRate = (Total_Individuals_Vaccinated / Population_2019) * 100
    return VaccinationRate


def calcRateOfVaccination(vaccineRecords: pd.DataFrame, statePop: pd.DataFrame,
                          stateGovernments: pd.DataFrame) -> tuple:
    """
    This function is used to calculate the VaccineRate and add it as a new column in the dataframe.
    :param vaccineRecords: Vaccine data
    :param statePop: Population data
    :param stateGovernments: Election data
    :return: Tuple with two dataframes consisting of vaccination rates.
    """
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


def stateClassificationByRulingParty(stateGovernments: pd.DataFrame, rateOfVaccination: pd.DataFrame) -> pd.DataFrame:
    """
    This function aims at merging the election dataframe and VaccineRate dataframe for further processing.
    :param stateGovernments: Election data
    :param rateOfVaccination: VaccineRate data
    :return: Dataframe with states classified based on the currently ruling party.
    """
    vaccinationRate_by_rulingParty = stateGovernments[['State', 'Party']].merge(
        rateOfVaccination[['State', 'Population_2019', 'Total Individuals Vaccinated', 'VaccinationRate']], on='State')
    return vaccinationRate_by_rulingParty



def hypothesis1(electionData, vaccineData, populationData):
    """
    This function consists of all the steps required for Hypothesis 1.
    All the function calls for Hypothesis 1 are made from this function.
    :param electionData: Dataframe with results for 2019 elections in India.
    :param vaccineData: Dataframe with vaccination records for each state in India.
    :param populationData: Dataframe with population in each state as of 2019.
    """

    # CLeaning the data to have uniform names of states across the input files.
    statePopulation = cleanData(populationData)

    # Fetching the currently ruling parties in each state of India.
    stateGovernments = getStateGovernments(electionData)

    # Fetching the statewise vaccination records for the latest dates.
    stateVaccineRecords = getStateVaccineRecords(vaccineData)

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
    vaccinationRate_by_rulingParty = stateClassificationByRulingParty(stateGovernments, rateOfVaccination_vs_party[1])

    # plotting bar bar graph for States vs Vaccination rate and coloring the bar plot to represent the ruling party.
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


def getTestGrouped(testData: pd.DataFrame) -> pd.DataFrame:
    """
    Combines number of tests conducted and positive cases for each day recorded in dataset for every state.
    :param testData: Data frame consisting of the COVID-19 testing data for each state.
    :return: Dataframe having total number of tests conducted and positive cases for each day recorded.
    """
    testData['Updated On'] = pd.to_datetime(testData['Updated On'], infer_datetime_format=True)
    testData.drop(testData[testData['Updated On'].dt.year != 2021].index, inplace=True)
    testingTotal = testData.groupby(['Updated On']).agg({'Total Tested': 'sum'})
    testingTotal = testingTotal.reset_index()

    testingPositive = testData.groupby(['Updated On']).agg({'Positive': 'sum'})
    testingPositive = testingPositive.reset_index()

    testGrouped = testingTotal.merge(testingPositive, on='Updated On')

    return testGrouped


def getTotalDailyVaccinated(vaccination: pd.DataFrame) -> pd.DataFrame:
    """
    Combines number of individuals vaccinated for each day across all the states.
    :param vaccination: Dataframe consisting of vaccination records for each state.
    :return: Dataframe having total number of individuals vaccinated for each day recorded.
    """
    vaccination['Updated On'] = pd.to_datetime(vaccination['Updated On'], infer_datetime_format=True)
    vaccineCombined = vaccination.groupby(['Updated On']).agg({'Total Individuals Vaccinated': 'sum'})
    vaccineCombined = vaccineCombined.reset_index()

    return vaccineCombined


def hypothesis2(testingData: pd.DataFrame, vaccinesData: pd.DataFrame):
    """
    This function consists of all the steps required for Hypothesis 1.
    All the function calls for Hypothesis 1 are made from this function.
    :param testingData: Data frame consisting of the COVID-19 testing data for each state.
    :param vaccinesData: Dataframe consisting of vaccination records for each state.
    """
    testingGrouped = getTestGrouped(testingData)
    vaccineGrouped = getTotalDailyVaccinated(vaccinesData)

    testingVsVaccinated = vaccineGrouped.merge(testingGrouped, on='Updated On')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=testingVsVaccinated['Updated On'], y=testingVsVaccinated['Total Tested'],
                             mode='lines',
                             name='Total Tests Conducted'))

    fig.add_trace(go.Scatter(x=testingVsVaccinated['Updated On'], y=testingVsVaccinated['Total Individuals Vaccinated'],
                             mode='lines',
                             name='Total Individuals Vaccinated'))

    fig.add_trace(go.Scatter(x=testingVsVaccinated['Updated On'], y=testingVsVaccinated['Positive'],
                             mode='lines',
                             name='Total Positive Cases'))
    fig.show()


if __name__ == '__main__':
    # Loading all the input files for Hypothesis 1
    stateElections = pd.read_csv("./StateElectionData.csv")
    stateVaccinations = pd.read_csv("./covid_vaccine_statewise.csv")
    statePopulation = pd.read_csv("./statePopulationIndia.csv")
    stateVaccinations = cleanData(stateVaccinations)
    stateVaccineRecords = getStateVaccineRecords(stateVaccinations)
    # Cleaning all the file for Hypothesis 1
    stateElections = cleanData(stateElections)

    statePopulation = cleanData(statePopulation)

    # Loading all the input files for Hypothesis 2
    stateTesting = pd.read_csv("./statewise_tested_numbers_data.csv")

    # Performing analysis for Hypothesis 1
    hypothesis1(stateElections, stateVaccinations, statePopulation)

    # CLeaning all the files for Hypothesis 2
    stateTesting = cleanData(stateTesting)

    # Performing analysis for Hypothesis 2
    hypothesis2(stateTesting, stateVaccinations)
